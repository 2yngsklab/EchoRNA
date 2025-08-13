import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric
import pickle
import random
import os

LIST_NODE_FEAT={"one-hot", "esm-2", "esm-if"}


def process_distance_matrix(complex):
    padded_matrix = torch.cat([
        complex[0].unsqueeze(0),  # First row at the beginning
        complex,
        complex[-1].unsqueeze(0)  # Last row at the end
    ])
    
    return padded_matrix



class RBPDataset(Dataset):
    def __init__(self, alphabet, batch_size=1, torch_geometric = True, node_feat = ["one-hot", "esm-2", "esm-if"],
                 path="/home/skfskfl9898/new-RNAdiffusion/seal/data",
                 dataset_dir = "/home/skfskfl9898/new-RNAdiffusion/seal/dataset/selex", 
                 split="train", device="cuda:0",
                max_pad = 6,
                pad_thr = 40,
                pad_pr = 0.75):

        """
        Args:
            alphabet : RNA-FM tokenizer
            batch_size (int) : Input batch size. Can learn multiple time step at once per complex
            torg_geometric (bool) : True if using torch_geometric.data.Data
            node_feat (list) : list of node feature to use, default = ["one_hot", "esm-2", "esm-if"]
            path (str) : path to data directory (where pkl, esm embeddings are saved) 
            split (str) : select train/valid/test dataset (where txt is saved)
            device (str) : use gpu(cuda) or cpu(cpu)
        """

        self.batch_converter = alphabet.get_batch_converter()
        self.torch_geometric = torch_geometric
        self.node_feat = node_feat
        self.path = path
        self.batch_size = batch_size
        self.device = device
        self.dataset_dir = os.path.join(dataset_dir, split+".txt")
        self.data = self.load_data()
        self.max_pad = max_pad
        self.pad_thr = pad_thr
        self.pad_pr = pad_pr
        self.random_elements = [4, 5, 6, 7]
        self.split = split
        
    def load_data(self):
        with open(self.dataset_dir, "r") as f:
            txt = f.readlines()
            
        return [i.strip() for i in txt]        
    
    def __len__(self):
        return len(self.data)

    def shuffle_dataset(self):
        """Shuffle the dataset entries"""
        random.shuffle(self.data)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        with open(os.path.join(self.path, "complex_pkl", data+".pickle"), "rb") as f:
            protein = pickle.load(f)
        
        
        invalid_feats = set(self.node_feat) - LIST_NODE_FEAT
        if invalid_feats:
            raise ValueError(f"Invalid node feature(s): {invalid_feats}. Must be a subset of {LIST_NODE_FEAT}")

        features_to_concat = []
        if "one-hot" in self.node_feat:
            features_to_concat.append(protein["node_s"])
        if "esm-2" in self.node_feat:
            features_to_concat.append(torch.load(os.path.join(self.path, "esm2_rep", data+".pt")).squeeze())
        if "esm-if" in self.node_feat:
            features_to_concat.append(torch.load(os.path.join(self.path, "esmIF_rep", data+".pt")))

        protein["node_s"] = torch.cat(features_to_concat, dim=1)
        # protein["node_s"] = torch.cat([protein["node_s"], esm2], dim=1)
        
        rna = [("rna", protein["rna_seq"])]
               # ("padding", "A"*254)]
        
        # print(rna)
        _, _, batch_tokens = self.batch_converter(rna)
        # batch_tokens = batch_tokens[0].reshape(1, -1)

        pad_mask = (
            batch_tokens.ne(1)
        )
        # protein['attention_bias'] = protein['attention_bias'].unsqueeze(0)
        # protein['attention_bias'] = protein['attention_bias'].expand(self.batch_size,
        #                                                              protein['attention_bias'].size(1),
        #                                                              protein['attention_bias'].size(2))
        if protein['attention_bias'] == None:
            protein['attention_bias'] = torch.zeros(len(rna[0][1]),protein['coords'].size(0), device = self.device)
        if self.split=="train" or self.split=="allpdbs":
            augment = True
        else:
            augment = False
        if augment and len(rna[0][1]) <= self.pad_thr and random.random() < self.pad_pr:
            b = []
            b_pad_mask = []
            b_atten_bias = []
            for b_idx in range(self.batch_size):
                rand_s = random.randint(0, self.max_pad)
                rand_e = random.randint(0, self.max_pad)
                b.append(torch.cat((batch_tokens[:,:1],
                                          torch.tensor([random.choice(self.random_elements) for _ in range(rand_s)],
                                                       dtype=batch_tokens.dtype, 
                                                       device=batch_tokens.device).unsqueeze(0),
                                          batch_tokens[:,1:-1],
                                          torch.tensor([random.choice(self.random_elements) for _ in range(rand_e)],
                                                       dtype=batch_tokens.dtype, 
                                                       device=batch_tokens.device).unsqueeze(0),
                                          batch_tokens[:,-1:]),dim = -1).squeeze(0))
                b_pad_mask.append(torch.cat((pad_mask[:,:1],
                                          torch.tensor([False for _ in range(rand_s)]).unsqueeze(0),
                                          pad_mask[:,1:-1],
                                          torch.tensor([False for _ in range(rand_e)]).unsqueeze(0),
                                          pad_mask[:,-1:]), dim=-1).squeeze(0))
                b_atten_bias.append(torch.cat([
                                            protein['attention_bias'][0, :].repeat(rand_s+1, 1),
                                            protein['attention_bias'][:, :],
                                            protein['attention_bias'][-1, :].repeat(rand_e+1, 1)
                                        ], dim=0))
            max_len = max(seq.size(0) for seq in b)
            padded_sequences = [F.pad(seq, (0, max_len - seq.size(0)), value=1) for seq in b]
            batch_tokens = torch.stack(padded_sequences, dim=0)
            padded_sequences = [F.pad(seq, (0, max_len - seq.size(0)), value=False) for seq in b_pad_mask]
            pad_mask = torch.stack(padded_sequences, dim=0)
            padded_sequences = [F.pad(seq, (0, 0, 0, max_len - seq.size(0)), value=0) for seq in b_atten_bias]
            protein['attention_bias'] = torch.stack(padded_sequences, dim=0)
        else:
            batch_tokens = batch_tokens.expand(self.batch_size, -1)
            pad_mask = pad_mask.expand(self.batch_size, -1)
            protein['attention_bias'] = torch.cat([
                                            protein['attention_bias'][0, :].unsqueeze(0),
                                            protein['attention_bias'][:, :],
                                            protein['attention_bias'][-1, :].unsqueeze(0)
                                        ], dim=0)
            protein['attention_bias'] = protein['attention_bias'].unsqueeze(0).expand(self.batch_size,
                                                                                      protein['attention_bias'].size(0),
                                                                                      protein['attention_bias'].size(1))
                                                                                      
        target = batch_tokens.clone()
        target[target==1]=2
        
        rna = {}
        rna["input_ids"] = batch_tokens
        rna["pad_mask"] = pad_mask.bool()
        rna["targets"] = target
            
        for key, value in rna.items():
            rna[key] = value.to(self.device)
        for key, value in protein.items():
            if isinstance(value, torch.Tensor):  # Ensures only tensors are moved to device
                protein[key] = value.to(self.device)
        if self.torch_geometric:
            protein = torch_geometric.data.Data.from_dict(protein).to(self.device)
            rna = torch_geometric.data.Data.from_dict(rna).to(self.device)
        
                
        return (rna, protein), data[:4].upper()
