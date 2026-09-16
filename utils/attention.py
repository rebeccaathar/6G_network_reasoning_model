import torch

def build_causal_mask(seq_len, dtype, device=None):
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
 
 
def suppress_span(causal_mask, suppress_start, suppress_end, dtype):
    mask = causal_mask.clone()
    min_val = torch.finfo(dtype).min
    mask[:, :, suppress_end:, suppress_start:suppress_end] = min_val
    return mask


def mask_hide_columns_from_row(causal_mask, from_row, hide_start, hide_end, dtype):
    mask = causal_mask.clone()
    min_val = torch.finfo(dtype).min
    mask[:, :, from_row:, hide_start:hide_end] = min_val
    return mask