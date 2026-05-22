import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch_geometric
from torch_geometric.data import Batch
import numpy as np
import functools
import math
import fm
import os
from source.model import RNP_adapter
from source.sampling import topk_masking, sample_from_categorical

class Echo_diffusion(nn.Module):
    """
    Discrete diffusion model for RNA sequence generation conditioned on protein structures.
    
    This class implements a discrete diffusion process where RNA tokens are gradually
    masked and then denoised using the RNP_adapter model. The diffusion process can
    use either linear or cosine noise schedules.
    
    Args:
        adaptor_config: Configuration dictionary for RNP_adapter model
        num_diff_step: Number of diffusion timesteps (default: 100)
    """
    def __init__(self, adaptor_config, num_diff_step=100):
        super(Echo_diffusion, self).__init__()
        self.model = RNP_adapter(**adaptor_config)
        self.pad_id=1
        self.bos_id=0
        self.eos_id=2
        self.x_id = 3
        self.mask_id=24
        self.num_diff_step=num_diff_step
    
        
    def get_non_special_sym_mask(self, rna_ids, partial_masks=None):
        """
        Create a mask for non-special tokens (excludes PAD, BOS, EOS).
        
        Args:
            rna_ids: RNA token IDs tensor
            partial_masks: Optional additional mask to apply
            
        Returns:
            Boolean mask where True indicates non-special tokens
        """
        non_special_sym_mask = (
            rna_ids.ne(self.pad_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask

    def sample_noise(self, x_0, t1, maskable_mask, scheduler='cosine'):
        """
        Sample noised sequence conditioned to time t with optional cosine scheduler.
        
        Args:
            x_0 (torch.Tensor): RNA input indices (bs * L)
            t1 (torch.Tensor): Time step tensor
            maskable_mask (torch.Tensor): Bool tensor indicating maskable tokens
            scheduler (str): Noise scheduler, either 'linear' (default) or 'cosine'
        
        Returns:
            dict: {'x_t': torch.Tensor, 't': torch.Tensor, 'mask_mask': torch.Tensor}
        """
        u = torch.rand_like(x_0, dtype=torch.float)
        
        if scheduler == 'cosine':
            alpha_t = 0.5 * (1 - torch.cos(math.pi * t1 / self.num_diff_step))  # Cosine schedule
        else:
            alpha_t = t1 / self.num_diff_step  # Linear schedule (default)
        t1_mask = (u < alpha_t[:, None]) & self.get_non_special_sym_mask(x_0)
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)
        
        return {
            "x_t": x_t1,
            "t": t1,
            "mask_mask": t1_mask & maskable_mask,
        }

    
    def forward(self, protein, rna, t, 
                mask_x=None, mask_gvp=None, 
                attention_bias=None, bias_scale=1.0,
                return_attn=False):
        """
        Forward pass through the diffusion model.
        
        Args:
            protein: Protein structure data (PyTorch Geometric batch)
            rna: RNA token IDs tensor
            t: Current diffusion timestep
            mask_x: Optional mask for RNA tokens
            mask_gvp: Optional mask for protein nodes
            attention_bias: Optional distance-based attention bias
            bias_scale: Scaling factor for attention bias (default: 1.0)
            return_attn: Whether to return attention weights (default: False)
            
        Returns:
            Output from RNP_adapter model (logits, fm_logits, attention_weights)
        """
        return self.model(protein, {"input_ids":rna}, t,
                          mask_x=mask_x, mask_gvp=mask_gvp,
                          num_timesteps=self.num_diff_step,
                          distance_bias=attention_bias, 
                          distance_bias_scale=bias_scale,
                          return_attn=return_attn)
    
    def compute_loss(self, protein, rna, attention_bias=None, 
                     bias_scale=1.0, return_attn=False, weighting="constant"):
        """
        Compute diffusion training loss.
        
        Args:
            protein: Protein structure data
            rna: Dictionary containing 'input_ids' and 'pad_mask'
            attention_bias: Optional distance-based attention bias
            bias_scale: Scaling factor for attention bias (default: 1.0)
            return_attn: Whether to return attention weights (default: False)
            weighting: Loss weighting scheme - 'linear' or 'constant' (default: 'constant')
            
        Returns:
            logits: Model predictions
            target: Ground truth RNA tokens
            loss_mask: Mask indicating which tokens to compute loss on
            weight: Per-sample loss weights
        """
        target = rna['input_ids']

        t1 = torch.randint(
            1, self.num_diff_step + 1,
            (target.size(0),),
            device=target.device
        )
        # print('t1', t1)

        # x_t, t, loss_mask = list(
        #     self.sample_noise(
        #         target, t1,
        #         maskable_mask=self.get_non_special_sym_mask(target)
        #     ).values()
        # )
        x_t, t, loss_mask = list(
            self.sample_noise(
                target, t1,
                maskable_mask=rna['pad_mask']
            ).values()
        )

        # print('t', t)
        # print('x_t', x_t)
        # print('x_t', x_t[0])
        logits, _, _ = self.forward(protein, x_t, t,
                                    mask_x=None, mask_gvp=None,
                                    attention_bias=attention_bias, bias_scale=bias_scale,
                                    return_attn=return_attn)
        weight = {
            "linear": torch.clamp(1 - loss_mask.sum(dim=-1) / rna['pad_mask'].sum(dim=-1), min=0.05),
            "constant": torch.ones_like(t) # 1
        }[weighting][:, None].float() 
        
        return logits, target, loss_mask, weight

    def initialize_output_tokens(self, batch, partial_masks=None, **kwargs):
        """
        Initialize output tokens for generation by masking all non-special tokens.
        
        Args:
            batch: Dictionary containing 'input_ids'
            partial_masks: Optional mask indicating tokens to preserve
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            output_tokens: Initialized tokens with maskable positions set to MASK token
            output_scores: Zero-initialized scores tensor
        """
        tokens = batch['input_ids']
        if tokens is None:
            raise NotImplementedError
        else:
            output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)
            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)
            return output_tokens, output_scores

            
    def resample_conditional(self, protein, _tokens, _scores, ratio, scale, step):
        """
        Conditional resampling to fix tokens that appear too frequently.
        
        This function identifies sequences where a single token appears more frequently
        than the specified ratio and resamples those positions to increase diversity.
        
        Args:
            protein: Protein structure data
            _tokens: Current token predictions
            _scores: Current prediction scores
            ratio: Maximum allowed frequency for any single token (e.g., 0.8 = 80%)
            scale: Noise scale for stochastic sampling
            step: Current generation step
            
        Returns:
            None (modifies _tokens and _scores in-place)
        """
        to_be_resample_idx = []
        resample_input = []
        resample_input_mask = []
        resample_input_scores = []
        for i, seq in enumerate(_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * ratio:#max(0.3/(step+1) ** 0.2, 0.1):
                to_be_resample_idx.append(i)
                resample_input_scores.append(_scores[i])
                mask = torch.zeros_like(seq).bool()
                for k, v in most_token_dict.items():
                    if len(v) > len(seq) * ratio:#max(0.3/(step+1) ** 0.2, 0.1):
                        mask |= seq.eq(k)
                resample_input_mask.append(mask)
                resample_input.append(seq.masked_fill(mask, self.mask_id))
                #resample_input.append(seq.masked_scatter(mask, xt[i][mask]))
            
        if len(to_be_resample_idx) > 0:
            resample_input = torch.stack(resample_input, dim=0).type_as(_tokens)
            resample_input_scores = torch.stack(resample_input_scores, dim=0).type_as(_scores)
            resample_input_mask = torch.stack(resample_input_mask, dim=0).type_as(_tokens).bool()
            

            resample_logits, _, _ = self.model(protein, {"input_ids":resample_input}, self.num_diff_step-step, num_timesteps=self.num_diff_step, mask_x=None, mask_gvp=None, distance_bias=None, distance_bias_scale=1, return_attn=False)
            
            if resample_logits.dtype != _scores.dtype:
                resample_logits = resample_logits.type_as(_scores)
            resample_logits[..., self.mask_id] = -math.inf
            resample_logits[..., self.x_id] = -math.inf
            resample_logits[..., self.pad_id] = -math.inf
            resample_logits[..., self.bos_id] = -math.inf
            resample_logits[..., self.eos_id] = -math.inf
            
            resample_logits = top_k_top_p_filtering(resample_logits, top_p=0.95)
            #noise_scale = 1.5 - 0.2 * ((step + 1) / max_step)
            noise_scale = scale
            assert resample_logits.size(0) == len(to_be_resample_idx)
            resample_tokens, resample_scores = stochastic_sample_from_categorical(resample_logits, temperature=0.0, noise_scale=noise_scale)
            resample_input.masked_scatter_(resample_input_mask, resample_tokens[resample_input_mask])
            resample_input_scores.masked_scatter_(resample_input_mask, resample_scores[resample_input_mask])
            _tokens[to_be_resample_idx], _scores[to_be_resample_idx] = resample_input, resample_input_scores


    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
        pretrain=False
    ):
        """
            This function is used to perform reparameterized decoding.
        """
        # output_tokens: [B, N]
        # output_scores: [B, N]
        # cur_tokens: [B, N]
        # cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)
        # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"

        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        if schedule == "linear":
            rate = 1 - (max_step - t) / max_step
        elif schedule == "cosine":
            rate = np.cos((max_step - t) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()


        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        to_be_resample = []
        for i, seq in enumerate(cur_tokens):
            most_token_dict = {}
            most_token = None
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token == self.pad_id:
                    continue
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token = token
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * 0.25:
                to_be_resample.append(i)
                
        # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
            if len(to_be_resample) > 0:
                noise_scale = 1.5
                #print(lowest_k_mask[to_be_resample[0]])
                lowest_k_mask[to_be_resample] = topk_masking(_scores_for_topk[to_be_resample], cutoff_len[to_be_resample], 
                                                             stochastic=True, temp=noise_scale * rate)
        else:
            raise NotImplementedError

        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.

        # #### the `uncond` mode ####
        # In our reparameterized decoding,
        # both v1_t and v2_t can be fully determined by the current token scores .

        # #### the `cond` mode ####
        # However, we can also impose some conditional constraints on v1_t so that
        # the decoding can be performed in a more conservative manner.
        # For example, we can set v1_t = 0 only when
        # (the newly output tokens are the same as previous denoised results, AND
        # the current token score becomes lower, AND
        # the current token score is not in the top-k share among all tokens).
        if condition == "cond":
            not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError

        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        last_mask_position = xt_neq_x0
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
        # b_{t} = (b_{t+1} & u_t) | v_t
        # For convenience, save the NOT of b_t for the next iteration
        # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
        #
        # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        assert (new_xt_neq_x0 == not_v2_t).all()
        return new_xt_neq_x0, output_tokens, output_scores

    def forward_decoder(self, prev_decoder_out, encoder_out=None, need_attn_weights=False, partial_masks=None,
                        sampling_strategy='gumbel_argmax', resample=True):
        
        protein = prev_decoder_out['protein']
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']
        score_history = prev_decoder_out['score_history']
        output_masks = self.get_non_special_sym_mask(output_tokens, partial_masks=partial_masks)


        logits, _, _ = self.model(protein, {"input_ids":output_tokens}, step, num_timesteps=self.num_diff_step, mask_x=None, mask_gvp=None, distance_bias=None, distance_bias_scale=1, return_attn=False)
        
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        logits[..., self.mask_id] = -math.inf
        # logits[..., self.x_id] = -math.inf
        # logits[..., self.pad_id] = -math.inf
        # logits[..., self.bos_id] = -math.inf
        # logits[..., self.eos_id] = -math.inf
        
        if sampling_strategy == 'vanilla':
            _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        elif sampling_strategy == 'argmax':
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == 'gumbel_argmax':
            noise_scale = 1.0
            _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=noise_scale)
            if resample:
                self.resample_conditional(protein, _tokens, _scores, ratio=0.8, scale=1.0, step=step)# 0.25 migh be high for prot by maybe not rna -> 0.4
        else:
            raise NotImplementedError
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
    
        # history.append(output_tokens.clone())
    
        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=None, # [B, L, H, T, T]
            step=step-1,
            max_step=max_step,
            history=history,
            score_history = score_history
            # hidden_states=hidden_state,
        )

    def generate_RDMSampling(self, batch, 
                             use_tqdm = False,
                             temperature=None, 
                             partial_masks=None,
                             sampling_strategy='gumbel_argmax',
                             decoding_strategy='reparam-uncond-stochastic1.0-linear',
                             resample=True
                            ):
        
        max_iter = self.num_diff_step
        temperature = temperature
        rna, protein = batch

        encoder_out = {}

        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            rna, encoder_out=encoder_out, partial_masks=partial_masks)
        
        prev_decoder_out = dict(
            protein=batch[1], # protein
            output_tokens=initial_output_tokens, # all maksed
            # output_scores=initial_output_scores, # all-zeros #SHOULD NOT WE START WITH ALL -INF?
            output_scores=torch.full_like(initial_output_scores, float('-inf')),
            output_masks=None,
            attentions=None,
            step=max_iter,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            score_history=[torch.full_like(initial_output_scores, float('-inf'))],
            temperature=temperature,
        )

        prev_decoder_out['output_masks'] = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
            
        for step in range(max_iter-1, -1, -1):
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    resample=resample
                )
            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']
            non_special_sym_mask = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
            output_masks, result_tokens, result_scores = self._reparam_decoding(
                output_tokens=prev_decoder_out['output_tokens'].clone(),
                output_scores=prev_decoder_out['output_scores'].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy='reparam-uncond-stochastic1.0-linear',#'reparam-uncond-stochastic1.0-linear'
                xt_neq_x0=prev_decoder_out['output_masks'],
                non_special_sym_mask=non_special_sym_mask,
                t=step,
                max_step=max_iter,
                noise=self.mask_id,
            ) 
            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores
            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step-1,
                history=decoder_out['history'],
                score_history=decoder_out['score_history'],
            )
            prev_decoder_out['history'].append(output_tokens)
            prev_decoder_out['score_history'].append(output_scores)
        decoder_out = prev_decoder_out
        return decoder_out['output_tokens'], {'output_scores': decoder_out['output_scores'], 'history': decoder_out['history'], 'score_history': decoder_out['score_history']}
