import torch
from torch.nn import functional as F
from .token_utils import get_sentence_token_spans
from .attention import build_causal_mask
import matplotlib.pyplot as plt
import os
import numpy as np

def compute_gradient_x_input_flow(tokenizer, model, prompt_text, generated_sentences,
                                   answer_chunk=None, use_abs=True, aggregate="mean"):
   
    all_chunks = [prompt_text] + generated_sentences

    if answer_chunk is not None:
        all_chunks = all_chunks + [answer_chunk]

    input_ids, spans = get_sentence_token_spans(tokenizer, all_chunks, device=model.device)
    seq_len = input_ids.shape[1]
    n = len(spans) 
    dtype = next(model.parameters()).dtype

    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        inputs_embeds = embed_layer(input_ids)
    inputs_embeds = inputs_embeds.clone().detach().requires_grad_(True)

    causal_mask = build_causal_mask(seq_len, dtype, device=model.device)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=causal_mask)
    logits = outputs.logits[0].float()  # (seq_len, vocab)
    logp = F.log_softmax(logits, dim=-1)

    target_ids = input_ids[0]
    token_logprob = logp[:-1, :].gather(1, target_ids[1:].unsqueeze(1)).squeeze(1)

    flow_matrix_grad = torch.zeros((n, n), dtype=torch.float32)
    
    for target_idx in range(1, n):
        s, e = spans[target_idx]
        if s == 0 or e <= s:
            continue 

        f_target = token_logprob[s - 1: e - 1].sum() # --> F

    
        retain = target_idx < (n - 1)
        grad = torch.autograd.grad(f_target, inputs_embeds, retain_graph=retain)[0][0]
     
        token_attr = torch.norm(grad * inputs_embeds[0].detach(), dim=-1)  # (seq_len,)
      
        for source_idx in range(0, target_idx):
            ss, se = spans[source_idx]
            if se <= ss:
                continue
            source_attr = token_attr[ss:se]
            value = source_attr.mean() if aggregate == "mean" else source_attr.sum()
            flow_matrix_grad[source_idx, target_idx] = value.item()
            
        del grad, token_attr

    final_answer_idx = n - 1
    n_reasoning = (n - 2) if answer_chunk is not None else (n - 1)

    n_targets = torch.tensor(
        [max(n - 1 - i, 1) for i in range(n)], dtype=torch.float32
    )
    sentence_relevance = flow_matrix_grad[:, final_answer_idx]
    downstream_importance = flow_matrix_grad.sum(dim=1) / n_targets

    return {
        "all_chunks": all_chunks,
        "spans": spans,
        "input_ids": input_ids,
        "flow_matrix": flow_matrix_grad,  
        "sentence_relevance": sentence_relevance,
        "downstream_importance": downstream_importance,
        "final_answer_idx": final_answer_idx,
        "n_reasoning": n_reasoning,
    }


def plot_gradient_input_heatmap(analysis, title="Gradient x Input", signed=True):
    n = analysis["flow_matrix"].shape[0]
    n_reasoning = analysis["n_reasoning"]
    mat = analysis["flow_matrix"][1:n, 1:n].numpy()

    labels = [f"s{i}" for i in range(n_reasoning)]
    if analysis["final_answer_idx"] == n - 1 and n - 1 > n_reasoning:
        labels = labels + ["ANSWER"]

    cmap = "viridis" if signed else "magma"
    vmax = float(np.abs(mat).max()) if mat.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    vmin = -vmax if signed else 0.0

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), max(5, 0.55 * len(labels))))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Target sentence s'' (effect)")
    ax.set_ylabel("Source sentence s' (cause)")
    ax.set_title(title)
    cbar_label = "Gradient x Input Attribution" + (" (signed)" if signed else " (magnitude |.|)")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()

    os.makedirs("./figures", exist_ok=True)
    plt.savefig("./figures/gradient_x_input_heatmap.png")
    plt.show()


