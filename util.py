from typing import Any, Callable, Dict, List, Optional, Set, Union
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Trainer, is_apex_available
import numpy as np
from fastdtw import fastdtw
import torch
from torch import nn
from dataclasses import dataclass, field
if is_apex_available():
    from apex import amp
from packaging import version
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward() 

        return loss.detach()


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# convert boundary to seconds
def convert_to_s(delim, gt):
    d = list(map(lambda e: e*0.02, delim))
    g = list(map(lambda e: e/16000, gt))
    return d, g


# compute DTW cost, return dist, correct, len(path), bias
def calc_cost(delim, gt, dist_func, frame_length=0.02):
    delim, gt = convert_to_s(delim, gt)
    if len(delim) == 0: # since fastdtw(list, empty_list) outputs 0, which is not correct
        return sum([e**2 for e in gt]), 0, len(gt), sum([-e for e in gt])
    else:
        distance, path = fastdtw(delim, gt, dist=dist_func)
        correct, bias = 0, 0.0
        for i,j in path:
            if abs(delim[i]-gt[j]) < frame_length:
                correct += 1
            bias += delim[i] - gt[j]
        return distance, correct, len(path), bias


# obtain boundary for regular vocab
def dist_regular(pred_ids, val_dataset, dist_func, return_all=False): # center of blank
    total_dist, total_corr, total_t, total_bias, return_tuple = 0, 0, 0, 0, []
    for i in range(len(pred_ids)):
        ids = pred_ids[i]
        j, cnt, starts, stops = 0, 0, [], []
        while j < len(ids):
            if ids[j] > 0:
                start = j
                while j < len(ids)-1 and ids[j] == ids[j+1]:
                    j += 1
                end = j
                starts.append(start)
                stops.append(end)
            j += 1

        delim = []
        for j in range(len(stops)-1):
            delim.append( (stops[j] + starts[j+1])/2 + 0.5 )
        gt = val_dataset['stops'][i][:-1]
        dist, corr, t, bias = calc_cost(delim, gt, dist_func)
        total_dist += dist
        total_corr += corr
        total_t += t
        total_bias += bias
        if return_all:
            return_tuple.append((dist, corr, t, bias))
    if return_all:
        return total_dist/total_t, total_corr/total_t, total_bias/total_t return_tuple
    else:
        return total_dist/total_t, total_corr/total_t, total_bias/total_t


def dist_boundary(pred_ids, val_dataset, processor, boundary_list, dist_func, mid_boundary=False, return_all=False): # use phone_end boundary token
    total_dist, total_corr, total_t, total_bias, return_tuple = 0, 0, 0, 0, []
    b_list = processor.tokenizer.convert_tokens_to_ids(boundary_list)
    b_list = dict(zip(b_list, [1]*len(b_list)))
    for i in range(len(pred_ids)):
        ids = pred_ids[i]
        j, delim = 0, []
        while j < len(ids):
            if ids[j] in b_list:
                k = j+1
                while k < len(ids) and ids[k] == ids[j]:
                    k += 1
                if mid_boundary:
                    delim.append((j+k)/2) # mid of split_symbol
                else:
                    delim.append(k) # end of end_symbol
                j = k
            elif ids[j] == 0:
                k = j+1
                while k < len(ids) and ids[k] == 0:
                    k += 1
                if k == len(ids): # the 0s are ending 0s
                    break
                if ids[k] not in b_list: # 0 ... 0 + regular token, therefore 0s are boundaries. Otherwise 0 ... 0 + boundary_token: ignore 0
                    if j == 0 or (ids[j-1] not in b_list): # b_token + 0 .. 0 + regular token: ignore 0
                        delim.append((j+k)/2)
                j = k
            elif j > 0 and ids[j] != 0 and ids[j-1] != 0 and (ids[j-1] not in b_list) and ids[j] != ids[j-1]: # if no boundary predicted, but consecutive tokens are different
                delim.append(j)
                j += 1
            else:
                j += 1
        gt = val_dataset['stops'][i][1::2]
        dist, corr, t, bias = calc_cost(delim, gt, dist_func)
        total_dist += dist
        total_corr += corr
        total_t += t
        total_bias += bias
        if return_tuple:
            return_tuple.append((dist, corr, t, bias))
    if return_all:
        return total_dist/total_t, total_corr/total_t, total_bias/total_t, return_tuple
    else:
        return total_dist/total_t, total_corr/total_t, total_bias/total_t


# remove boundaries from predictions
def clean_str(s_list, boundary_list):
    res = []
    for s in s_list:
        res.append(' '.join(c for c in s.split(' ') if c not in boundary_list))
    return res


