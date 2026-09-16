import torch.nn.functional as F
from .token_utils import get_sentence_token_spans
from .attention import build_causal_mask, mask_hide_columns_from_row
import matplotlib.pyplot as plt
import torch
import numpy as np


def compute_entropy(logits):
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(-1)


def compute_predictive_entropy_curve(tokenizer, model, prompt_text, generated_sentences, answer_chunk):
    if answer_chunk is None:
        raise ValueError(
            "compute_predictive_entropy_curve requires an explicit answer_chunk ('Final answer: <number>'); " \
            "the model did not follow the requested format."
        )

    all_chunks = [prompt_text] + generated_sentences + [answer_chunk]
    #get span[i] = (start, end) -> the tokens range each chunk occupies in that single sequence. 
    input_ids, spans = get_sentence_token_spans(tokenizer, all_chunks, model.device) 
    seq_len = input_ids.shape[1]
    #spans[0] = prompt
    #spans[1..n-2] = reasoning steps
    #spans[n-1] = answer_chunk
    dtype = next(model.parameters()).dtype

    n = len(spans)
    n_reasoning = n - 2 
    final_answer_idx = n - 1
    ans_s, ans_e = spans[final_answer_idx] #the token range of the answer_chunk in the sequence

    baseline_mask = build_causal_mask(seq_len, dtype, device=model.device)

    entropies = torch.zeros(n_reasoning + 1, dtype=torch.float32)

    for k in range(n_reasoning + 1): #ranges from 0 (no reasoninng steps visible) to n_reasoning (all reasoning steps visible)
        if k == n_reasoning:
            mask_k = baseline_mask
        else:
            from_row = spans[k][1] if k > 0 else spans[0][1]
            hide_start = spans[k + 1][0]
            hide_end = spans[final_answer_idx][0] 
            mask_k = mask_hide_columns_from_row(baseline_mask, from_row, hide_start, hide_end, dtype)

        with torch.no_grad():
            logits_k = model(input_ids=input_ids, attention_mask=mask_k).logits[0]

        entropy_k = compute_entropy(logits_k)
        entropies[k] = entropy_k[ans_s:ans_e].mean().item()

    return {
        "entropies": entropies,         
        "n_reasoning": n_reasoning,
        "all_chunks": all_chunks,
        "spans": spans,
    }


def plot_entropy_suppression_delta(analysis, title="Effect of suppression on the entropy of the answer"):
    n_reasoning = analysis["n_reasoning"]
    idxs = np.arange(n_reasoning)
    delta = analysis["entropy_delta"][1:1 + n_reasoning].numpy()

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n_reasoning), 4))
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in delta]
    ax.bar(idxs, delta, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(idxs)
    ax.set_xticklabels([f"s{i}" for i in idxs])
    ax.set_xlabel("Supressed reasoning step (sentence)")
    ax.set_ylabel("dH = H_after_supression - H_before_supression (nats)")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_predictive_entropy_curve(entropy_curve, title="Predictive entropy over the answer vs. steps revealed"):
    entropies = entropy_curve["entropies"].numpy()
    n_reasoning = entropy_curve["n_reasoning"]
    ks = np.arange(n_reasoning + 1)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * (n_reasoning + 1)), 4))
    ax.plot(ks, entropies, marker="o", color="#2ca02c")
    ax.set_xticks(ks)
    ax.set_xticklabels(["0\n(only the question)"] + [f"{k}" for k in ks[1:]])
    ax.set_xlabel("Reasoning steps revealed (k)")
    ax.set_ylabel("Predicted entropy H(Y | s_0, ..., s_k) - avg (nats)")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig("./figures/entropy_curve.png")
    plt.show()