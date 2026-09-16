import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from .token_utils import get_sentence_token_spans
from .attention import build_causal_mask, suppress_span
import numpy as np

def compute_step_nll(logits, input_ids, span):
    s, e = span
    s_eff = max(s, 1) 
    if e <= s_eff:
        return None
    logp = F.log_softmax(logits, dim=-1)
    token_ids = input_ids[0, s_eff:e]
    token_logp = logp[s_eff - 1:e - 1, :].gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return -token_logp.mean()


def compute_conditional_perplexity_flow(tokenizer, model, prompt_text, generated_sentences, answer_chunk=None):
    all_chunks = [prompt_text] + generated_sentences
    if answer_chunk is not None:
        all_chunks = all_chunks + [answer_chunk]

    input_ids, spans = get_sentence_token_spans(tokenizer, all_chunks, device=model.device)
    seq_len = input_ids.shape[1]
    dtype = next(model.parameters()).dtype
    n = len(spans)

    baseline_mask = build_causal_mask(seq_len, dtype, device=model.device)
    with torch.no_grad():
        logits_base = model(input_ids=input_ids, attention_mask=baseline_mask).logits[0]

    nll_base = torch.full((n,), float("nan"), dtype=torch.float32)
    for j in range(1, n):
        nll = compute_step_nll(logits_base, input_ids, spans[j])
        if nll is not None:
            nll_base[j] = nll.item()

    delta_nll = torch.zeros((n, n), dtype=torch.float32)

    ##supress the attention of each source step s_i (i=1..n-2) and measure the effect on all subsequent steps s_j (j=i+1..n-1)
    for i in range(1, n - 1): 
        suppress_start, suppress_end = spans[i]
        suppressed_mask = suppress_span(baseline_mask, suppress_start, suppress_end, dtype)

        with torch.no_grad():
            logits_supp = model(input_ids=input_ids, attention_mask=suppressed_mask).logits[0]

        for j in range(i + 1, n):
            nll_ablated = compute_step_nll(logits_supp, input_ids, spans[j])
            if nll_ablated is None or torch.isnan(nll_base[j]):
                continue
            delta_nll[i, j] = nll_ablated.item() - nll_base[j].item()

    ppl_ratio = torch.exp(delta_nll)

    final_answer_idx = n - 1
    n_reasoning = (n - 2) if answer_chunk is not None else (n - 1)
    n_targets = torch.tensor([max(n - 1 - i, 1) for i in range(n)], dtype=torch.float32)

    necessity_to_answer = delta_nll[:, final_answer_idx]
    downstream_necessity = delta_nll.sum(dim=1) / n_targets

    return {
        "all_chunks": all_chunks, "spans": spans, "input_ids": input_ids,
        "flow_matrix": delta_nll,        
        "ppl_ratio": ppl_ratio,
        "nll_base": nll_base,
        "sentence_relevance": necessity_to_answer,
        "downstream_importance": downstream_necessity,
        "final_answer_idx": final_answer_idx,
        "n_reasoning": n_reasoning,
    }


def plot_conditional_perplexity_heatmap(cp_analysis, title="Causal necessity between steps (ΔNLL)"):
    n = cp_analysis["flow_matrix"].shape[0]
    n_reasoning = cp_analysis["n_reasoning"]
    mat = cp_analysis["flow_matrix"][1:n, 1:n].numpy()

    labels = [f"s{i}" for i in range(n_reasoning)] + ["ANSWER"]
    vmax = max(abs(mat.min()), abs(mat.max())) if mat.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), max(5, 0.55 * len(labels))))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("target sentence s'' (effect)")
    ax.set_ylabel("source sentence s' (cause)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="ΔNLL (nats) = NLL_wo_s' - NLL_base")
    fig.tight_layout()
    plt.savefig("./figures/conditional_ppl_heatmap.png")
    plt.show()