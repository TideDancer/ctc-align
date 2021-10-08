import warnings
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers.modeling_outputs import CausalLMOutput
from torch import nn
from ctc_ent import ctc_ent_cost
from noskip_ctc import noskip_ctc_loss
import re


class Wav2Vec2CTCTokenizerPho(Wav2Vec2CTCTokenizer):
    def __init__(self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        **kwargs):
        super().__init__(
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        **kwargs)

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        return (text, kwargs)

    def _tokenize(self, text, **kwargs):
        if self.do_lower_case:
            text = text.upper()
        t = text.split(" ")
        t = list(filter(lambda e: e, t)) # filter out empty token ""
        return t


class Wav2Vec2ForPCTC(Wav2Vec2ForCTC):
    def __init__(self, config, ctc_type=None, prior_type=None):
        super().__init__(config)
        self.ctc_type = ctc_type
        self.prior_type = prior_type
        if prior_type:
            self.prior_list = [] # prior list = [ tensor of counts, size = vocab_size ], e.g. can be a uniform prior, a tensor whose elements are all 1
            self.prior = torch.ones(config.vocab_size, requires_grad=False)

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # if prior: p_t(y_t | x_1^T) / p(s_t)
            if self.prior_type == 'posterior' or self.prior_type == 'uniform':
                T, b, v = log_probs.shape
                if self.prior.device != log_probs.device:
                    self.prior = self.prior.to(log_probs.device)
                if self.prior_type == 'posterior':
                    to_add = log_probs.detach().exp().sum(dim=0).sum(dim=0) # sum over T and b, obtain prob of each symbol as prior
                    self.prior_list.append(to_add) 
                    self.prior += to_add
                    if len(self.prior_list) > 100:
                        to_pop = self.prior_list.pop(0)
                        self.prior -= to_pop
                prior = (self.prior / self.prior.sum()).repeat(T, b, 1)
                log_probs = log_probs - prior.log()
                # # re-normalize, seems not useful
                # log_probs = log_probs.log_softmax(dim=-1)

            # ctc loss
            with torch.backends.cudnn.flags(enabled=False):
                if self.ctc_type == 'no-skip':
                    loss = noskip_ctc_loss(log_probs, labels, input_lengths, target_lengths)
                elif self.prior_type == 'entropy': # entropy-regularized = a prior
                    H, cost = ctc_ent_cost(log_probs, flattened_targets, input_lengths, target_lengths, sumed=False)
                    H, cost = torch.mean(H), torch.mean(cost)
                    loss = 0.9 * cost - 0.1 * H
                else:
                    loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

