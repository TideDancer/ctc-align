import math
import torch
import torch.nn.functional as F

def noskip_ctc_loss(log_probs : torch.Tensor, targets : torch.Tensor, input_lengths : torch.Tensor, target_lengths : torch.Tensor, blank : int = 0, reduction : str = 'none', finfo_min_fp32: float = torch.finfo(torch.float32).min, finfo_min_fp16: float = torch.finfo(torch.float16).min, alignment : bool = False):
    input_time_size, batch_size = log_probs.shape[:2]
    B = torch.arange(batch_size, device = input_lengths.device)

    if len(targets.shape) == 1: # only 1 dimension concatednated targets
        max_len = max(target_lengths)
        targets_copy = torch.full((batch_size, max_len), 0, device=log_probs.device, dtype=torch.long)
        i = 0
        cnt = 0
        for l in target_lengths:
            targets_copy[cnt, :l] = targets[i:i+l]
            i += l
            cnt += 1
    else:
        targets_copy = targets
    
    targets_t = torch.cat([targets_copy, targets_copy[:, :1]], dim = -1)
    targets_t[targets_t < 0] = 0
    
    zero_padding, zero = 1, torch.tensor(finfo_min_fp16 if log_probs.dtype == torch.float16 else finfo_min_fp32, device = log_probs.device, dtype = log_probs.dtype)
    log_probs_ = log_probs.gather(-1, targets_t.expand(input_time_size, -1, -1))
    log_alpha = torch.full((input_time_size, batch_size, zero_padding + targets_t.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)
    log_alpha[0, :, zero_padding + 0] = log_probs[0, B, targets_t[:, 0]]

    for t in range(1, input_time_size):
        log_alpha[t, :, 1:] = log_probs_[t] + logadd(log_alpha[t - 1, :, 1:], log_alpha[t - 1, :, :-1])

    loss = -log_alpha[input_lengths - 1, B].gather(-1, torch.stack([zero_padding + target_lengths-1], dim = -1)) 
    return torch.mean(loss)


def logadd(x0, x1, x2=None):
    # produces nan gradients in backward if -inf log-space zero element is used https://github.com/pytorch/pytorch/issues/31829
    x0 = x0.clone()
    x1 = x1.clone()
    if x2 is None:
        return torch.logsumexp(torch.stack([x0, x1]), dim = 0)
    else:
        x2 = x2.clone()
        return torch.logsumexp(torch.stack([x0, x1, x2]), dim = 0)
    

class LogsumexpFunction(torch.autograd.function.Function):
    @staticmethod
    def forward(self, x0, x1, x2):
        m = torch.max(torch.max(x0, x1), x2)
        m = m.masked_fill_(torch.isinf(m), 0)
        e0 = (x0 - m).exp_()
        e1 = (x1 - m).exp_()
        e2 = (x2 - m).exp_()
        e = (e0 + e1).add_(e2).clamp_(min = 1e-16)
        self.save_for_backward(e0, e1, e2, e)
        return e.log_().add_(m)

    @staticmethod
    def backward(self, grad_output):
        e0, e1, e2, e = self.saved_tensors
        g = grad_output / e
        return g * e0, g * e1, g * e2

