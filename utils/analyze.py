
from .entropy import compute_entropy
from .kl_divergency import compute_kl_for_suppression
from .token_utils import get_sentence_token_spans
from .attention import build_causal_mask, suppress_span
import torch
import numpy as np

def analyze_sentence_flows(tokenizer, model, prompt_text, generated_sentences, answer_chunk=None):
    all_chunks = [prompt_text] + generated_sentences
    if answer_chunk is not None:
        all_chunks = all_chunks + [answer_chunk]

    input_ids, spans = get_sentence_token_spans(tokenizer, all_chunks, device=model.device)
    seq_len = input_ids.shape[1]
    dtype = next(model.parameters()).dtype

    baseline_mask = build_causal_mask(seq_len, dtype, device=model.device)
    with torch.no_grad():
        logits_base = model(input_ids=input_ids, attention_mask=baseline_mask).logits[0]

    entropy_base = compute_entropy(logits_base)

    n = len(spans)
    flow_matrix = torch.zeros((n, n), dtype=torch.float32)
    entropy_delta = torch.zeros(n, dtype=torch.float32)

    final_answer_idx = n - 1
    ans_s, ans_e = spans[final_answer_idx]
    entropy_before_answer = entropy_base[ans_s:ans_e].mean().item()

    for target_idx in range(1, n - 1):
        suppress_start, suppress_end = spans[target_idx]
        suppressed_mask = suppress_span(baseline_mask, suppress_start, suppress_end, dtype)

        with torch.no_grad():
            logits_supp = model(input_ids=input_ids, attention_mask=suppressed_mask).logits[0]

        kl_per_token = compute_kl_for_suppression(logits_base, logits_supp)
        for j in range(target_idx + 1, n):
            s, e = spans[j]
            flow_matrix[target_idx, j] = kl_per_token[s:e].mean().item()

        entropy_supp = compute_entropy(logits_supp)
        entropy_after_answer = entropy_supp[ans_s:ans_e].mean().item()
        entropy_delta[target_idx] = entropy_after_answer - entropy_before_answer

    n_reasoning = n - 2

    sentence_relevance = flow_matrix[:, final_answer_idx]
    n_targets = torch.tensor(
        [max(n - 1 - i, 1) for i in range(n)], dtype=torch.float32
    )
    downstream_importance = flow_matrix.sum(dim=1) / n_targets

    alpha = 0.5
    relevance_score = alpha * sentence_relevance + (1 - alpha) * downstream_importance
    

    for i in range(1, n - 1):
        text = all_chunks[i].strip().replace("\n", " ")
        print(
            f"{i - 1:<5}{text[:55]:<60}{sentence_relevance[i]:>12.6f}"
            f"{downstream_importance[i]:>12.6f}{relevance_score[i]:>10.6f}{entropy_delta[i]:>12.6f}"
        )

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            if flow_matrix[i, j] > 0.0:
                label_j = "Answer" if j == final_answer_idx else f"ep[{j - 1}]"
                print(f"  ep[{i - 1}] -> {label_j}: {flow_matrix[i, j]:.6f}")

    return {
        "all_chunks": all_chunks,
        "spans": spans,
        "input_ids": input_ids,
        "flow_matrix": flow_matrix,
        "sentence_relevance": sentence_relevance,
        "downstream_importance": downstream_importance,
        "relevance_score": relevance_score,
        "final_answer_idx": final_answer_idx,
        "n_reasoning": n_reasoning,
        "entropy_base": entropy_base,
        "entropy_before_answer": entropy_before_answer,
        "entropy_delta": entropy_delta,
    }

def rank_sentence_relevance(analysis, top_k=None):
    n_reasoning = analysis["n_reasoning"]
    scores = analysis["relevance_score"][1:1 + n_reasoning]
    chunks = analysis["all_chunks"][1:1 + n_reasoning]

    order = torch.argsort(scores, descending=True).tolist()
    if top_k is not None:
        order = order[:top_k]

    for rank, idx in enumerate(order, start=1):
        text = chunks[idx].strip().replace("\n", " ")
        print(f"  #{rank:<3} ep[{idx}] score={scores[idx]:.6f}  {text[:80]}")

    return order

