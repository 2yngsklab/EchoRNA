from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.data import Batch
import numpy as np
import functools
import math
import fm
from source.dataloader import *
from source.lora import *
from source.diffusion import our_diffusion
from source.util import RDMCrossEntropyLoss
from tqdm.auto import tqdm
import argparse, yaml, pathlib, pprint
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn
)

parser = argparse.ArgumentParser(
    description="EchoRNA training script")
parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
args = parser.parse_args()

with open(args.config, "r") as f:
    C = yaml.safe_load(f)

print("Loaded config:")
pprint.pprint(C, compact=True)


device = torch.device(C["device"])
seed = C["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_, alphabet = fm.pretrained.rna_fm_t12()
dataset = C["dataset"]
data_dir = pathlib.Path(C["data_root"]) / dataset / "data"
bs = C["batch_size"]
batch_size=bs
train_dataset = RBPDataset(alphabet = alphabet, batch_size=bs, path=data_dir,
                             dataset_dir = data_dir,
                             split="train",
                             device=device,
                             max_pad = 6, pad_thr = 40, pad_pr = 0.75)

valid_dataset = RBPDataset(alphabet = alphabet, batch_size=bs, path=data_dir,
                 dataset_dir = data_dir, 
                 split="valid", device=device)

test_dataset = RBPDataset(alphabet = alphabet, batch_size=bs, path=data_dir,
                 dataset_dir = data_dir, 
                 split="test", device=device)

lora_config = LoRA_Config(**C["lora_config"])

adaptor_config = {**C["adaptor_config"], "lora_config": lora_config, "device": device}

net = our_diffusion(adaptor_config).to(device)

class EchoRNA(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.AdamW(self.model.parameters(), **C["optimizer"])
        self.lr_scheduler = None        
        self.scaler = torch.cuda.amp.GradScaler()
    
    def step(self, batch, attention_bais=None, preservation_loss_ratio=0):
        with torch.cuda.amp.autocast():
            logits, target, loss_mask, weight = self.model.compute_loss(batch[1], batch[0],
                                                                        attention_bias=attention_bais,
                                                                        weighting="linear")
            loss, logging_output = self.criterion(logits, target, label_mask = loss_mask, weights = weight)
            if preservation_loss_ratio!=0:
                prs_loss, _ = self.criterion(logits, target, label_mask = ~loss_mask)
                prs_loss = prs_loss/bs
            else:
                prs_loss = 0
            loss = loss/bs
        if torch.isnan(loss):
            print("Loss NAN on step ", self.global_step, "total mask :", loss_mask.sum())
            loss = loss * 0
            logging_output['nll_loss'] = logging_output['nll_loss'] * 0
            logging_output['fullseq_loss'] = logging_output['fullseq_loss'] * 0
            logging_output['fullseq_nll_loss'] = logging_output['fullseq_nll_loss'] * 0
            logging_output['ppl'] = logging_output['ppl'] * 0
            logging_output["accuracy"] = 0
        else:
            logging_output["accuracy"] = ((logits.argmax(-1) == target) & loss_mask).sum().item() / (loss_mask.sum().item() + 1e-9)

        logging_output["preservation"] = ((logits.argmax(-1) == target) & ~loss_mask).sum().item() / ((~loss_mask).sum().item() + 1e-9)
        return loss, logging_output, logits, target, loss_mask, prs_loss

    def training_step(self, batch, attention_bais=None, step=None,
                      	accumulation_steps=C["accumulation_steps"], step_counter=0,
                      preservation_loss_ratio=0):
        self.model.train()
        loss, logging_output, logits, target, loss_mask, prs_loss = self.step(batch, attention_bais, preservation_loss_ratio)
        if preservation_loss_ratio != 0:
            loss = (1-preservation_loss_ratio)*loss + preservation_loss_ratio*prs_loss
        self.scaler.scale(loss / accumulation_steps).backward()
        
        if step_counter % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        logging_output["lr"] = self.optimizer.param_groups[0]["lr"]
        metric = [loss.item(), logging_output['ppl'].item(), logging_output['accuracy'], logging_output['preservation']]
        return logging_output, metric

    
    def eval_step(self, batch, attention_bais):
        self.model.eval()
        with torch.no_grad():
            loss, logging_output, _, _, _, _ = self.step(batch)
        metric = [loss.item(), logging_output['ppl'].item(), logging_output['accuracy'], logging_output['preservation']]
        return logging_output, metric

model = EchoRNA(net, RDMCrossEntropyLoss()).to(device)

save_dir = pathlib.Path(C["save_root"]) / C["experiment_name"]
save_dir.mkdir(parents=True, exist_ok=True)

step_loss = {}
best_loss = -1
epochs = C["epochs"]
console = Console(highlight=False)

for e in range(epochs):
    # -------------------------------------------------
    # TRAIN ────────────────────────────────────────────
    # -------------------------------------------------
    train_loss = train_ppl = train_acc = train_prs = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Epoch {}/{}".format(e + 1, epochs)),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(compact=True),
        console=console,
        transient=True,            # clear bar when done
    ) as prog:

        train_task = prog.add_task("train", total=len(train_dataset))
        train_dataset.shuffle_dataset()

        for step, (batch, _) in enumerate(train_dataset):
            if e > 25 and random.random()  < 0.5:
                attention_bais = None
            else:
                if 'attention_bias' not in batch[1].keys():
                    attention_bais = None
                else:
                    attention_bais = batch[1]['attention_bias']
                    if attention_bais.max().item() == 0:
                        attention_bais = None
            logging_output, metric = model.training_step(batch,
                                                         attention_bais=attention_bais,
                                                         step_counter=step)
            train_loss += metric[0]
            train_ppl += metric[1]
            train_acc += metric[2]
            train_prs += metric[3]
            prog.advance(train_task)

    if model.lr_scheduler is not None and e > 10:
        model.lr_scheduler.step()
    lr = logging_output["lr"]

    # -------------------------------------------------
    # VALID ───────────────────────────────────────────
    # -------------------------------------------------
    eval_loss = eval_ppl = eval_acc = eval_prs = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]valid"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:

        val_task = prog.add_task("valid", total=len(valid_dataset))
        for batch, _ in valid_dataset:
            _, metric = model.eval_step(batch, None)
            eval_loss += metric[0]
            eval_ppl  += metric[1]
            eval_acc  += metric[2]
            eval_prs  += metric[3]
            prog.advance(val_task)

    # -------------------------------------------------
    # OPTIONAL TEST-GEN ───────────────────────────────
    # -------------------------------------------------
    test_gen_acc = None
    if (e + 1) % 10 == 0 or e == 0:
        correct_tok = total_tok = 0
        for batch, _ in test_dataset:
            tok, _ = model.model.generate_RDMSampling(
                batch,
                sampling_strategy="vanilla",
                decoding_strategy="reparam-uncond-deterministic-cosine",
            )
            correct_tok += (tok == batch[0]["input_ids"]).sum().item() - 2 * bs
            total_tok   += tok.numel() - 2 * bs
        test_gen_acc = round(correct_tok / total_tok, 4) if total_tok else 0

    # -------------------------------------------------
    # NEAT SUMMARY TABLE ──────────────────────────────
    # -------------------------------------------------
    table = Table(
        title=f"[bold white]Results | Epoch {e+1}/{epochs}",
        header_style="bold magenta",
        show_edge=False, pad_edge=False,
    )
    table.add_column("Split",  justify="left")
    table.add_column("CE loss", justify="right")
    table.add_column("PPL",     justify="right")
    table.add_column("Acc",     justify="right")
    table.add_column("Prs",     justify="right")
    table.add_column("LR",      justify="right")

    table.add_row(
        "Train",
        f"{train_loss/len(train_dataset):.4f}",
        f"{train_ppl/len(train_dataset):.4f}",
        f"{train_acc/len(train_dataset):.4f}",
        f"{train_prs/len(train_dataset):.4f}",
        f"{lr:.2e}",
    )
    table.add_row(
        "Valid",
        f"{eval_loss/len(valid_dataset):.4f}",
        f"{eval_ppl/len(valid_dataset):.4f}",
        f"{eval_acc/len(valid_dataset):.4f}",
        f"{eval_prs/len(valid_dataset):.4f}",
        "—",
    )
    if test_gen_acc is not None:
        table.add_row(
            "Test-gen",
            "—", "—",
            f"{test_gen_acc:.4f}",
            "—",
            "—",
            style="bold cyan"
        )

    console.print(table)
    step_loss[str(e + 1)] = {
        "lr": lr,
        "train_loss": round(train_loss / len(train_dataset), 4),
        "train_ppl":  round(train_ppl  / len(train_dataset), 4),
        "train_acc":  round(train_acc  / len(train_dataset), 4),
        "train_prs":  round(train_prs  / len(train_dataset), 4),
        "eval_loss":  round(eval_loss  / len(valid_dataset), 4),
        "eval_ppl":   round(eval_ppl   / len(valid_dataset), 4),
        "eval_acc":   round(eval_acc   / len(valid_dataset), 4),
        "eval_prs":   round(eval_prs   / len(valid_dataset), 4),
        "test_gen_acc": test_gen_acc
            if test_gen_acc is not None else
            step_loss[str(max((e // 10) * 10, 1))]["test_gen_acc"],
    }

    torch.save(model.model.state_dict(),  os.path.join(save_dir, f"model_epoch_{e+1}.pth"))
    torch.save(step_loss,                os.path.join(save_dir, "step_loss.pth"))