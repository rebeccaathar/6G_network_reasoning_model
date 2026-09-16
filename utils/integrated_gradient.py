import torch
import torch.nn.functional as F
from .attention import build_causal_mask
import matplotlib.pyplot as plt
import numpy as np 

def compute_target_score(logits, input_ids, span, mode="logprob"):
    s, e = span
    logp = F.log_softmax(logits, dim=-1)  # (1, seq_len, vocab)

    if mode == "logprob":
        s_eff = max(s, 1)
        total = logp.new_zeros(())
        for pos in range(s_eff, e):
            token_id = input_ids[0, pos]
            total = total + logp[0, pos - 1, token_id]
        return total
    elif mode == "neg_entropy":
        p = logp.exp()
        entropy = -(p * logp).sum(-1)  # (1, seq_len)
        return -entropy[0, s:e].mean()
    else:
        raise ValueError(f"mode desconhecido: {mode!r}")


def integrated_gradients(
    tokenizer, model, input_ids, spans, source_idx, target_idxs,
    ig_steps=16, baseline="zero", target_mode="logprob",
):
    
    device = input_ids.device
    embed_layer = model.get_input_embeddings()

    with torch.no_grad():
        input_embeds = embed_layer(input_ids)

    seq_len = input_ids.shape[1]
    dtype = input_embeds.dtype
    causal_mask = build_causal_mask(seq_len, dtype, device=device)

    src_s, src_e = spans[source_idx]

    if baseline == "zero":
        baseline_region = torch.zeros_like(input_embeds[:, src_s:src_e, :])
    elif baseline == "pad":
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        pad_id_t = torch.tensor([[pad_id]], device=device)
        with torch.no_grad():
            pad_embed = embed_layer(pad_id_t)
        baseline_region = pad_embed.expand(1, src_e - src_s, -1).clone()
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}")

    real_region = input_embeds[:, src_s:src_e, :].detach()
    #vector: (x-x') the distance in ebedding space between the real value and the baseline value for each token in the source span.
    diff_region = real_region - baseline_region

    alphas = torch.linspace(0.0, 1.0, ig_steps + 1)[1:]

    grad_accum = {t: torch.zeros_like(real_region) for t in target_idxs}

    for alpha in alphas:
        #x' + alpha*(x - x')
        region_alpha = (baseline_region + alpha * diff_region).clone().requires_grad_(True)

        interp_embeds = torch.cat(
            [input_embeds[:, :src_s, :].detach(),
             region_alpha,
             input_embeds[:, src_e:, :].detach()],
            dim=1,
        )

        logits = model(inputs_embeds=interp_embeds, attention_mask=causal_mask).logits

        for k, target_idx in enumerate(target_idxs):
            target_score = compute_target_score(logits, input_ids, spans[target_idx], mode=target_mode)
            retain = k < (len(target_idxs) - 1) 
            grad, = torch.autograd.grad(target_score, region_alpha, retain_graph=retain)

            grad_accum[target_idx] = grad_accum[target_idx] + grad.detach()

        del logits, region_alpha, interp_embeds

    results = {}
    for target_idx in target_idxs:
        avg_grad = grad_accum[target_idx] / ig_steps
        attributions = avg_grad * diff_region
        results[target_idx] = {
            "ig_score": attributions.sum().item(),
            "ig_score_abs": attributions.abs().sum().item(),
            "attributions": attributions.detach().cpu(),
        }
    return results



def compute_ig_flow_matrix(
    tokenizer, model, input_ids, spans, ig_steps=16, baseline="zero",
    target_mode="logprob", only_to_answer=False,
):
  
    n = len(spans)
    final_answer_idx = n - 1
    ig_matrix = torch.zeros((n, n), dtype=torch.float32)

    for i in range(1, n - 1):
        targets = [final_answer_idx] if only_to_answer else list(range(i + 1, n))
        results = integrated_gradients(
            tokenizer, model, input_ids, spans, source_idx=i, target_idxs=targets,
            ig_steps=ig_steps, baseline=baseline, target_mode=target_mode,
        )
        for j, res in results.items():
            ig_matrix[i, j] = res["ig_score_abs"]
        print(f" IG completed for source step s{i - 1} ({len(targets)} target(s))")

    return {
        "ig_matrix": ig_matrix,
        "final_answer_idx": final_answer_idx,
        "n_reasoning": n - 2,
        "ig_steps": ig_steps,
        "baseline": baseline,
        "target_mode": target_mode,
    }

def plot_ig_heatmap(ig_analysis, title="Flow of influence between steps (Integrated Gradients)"):
   
    ig_matrix = ig_analysis["ig_matrix"]
    n_reasoning = ig_analysis["n_reasoning"]
    n = ig_matrix.shape[0]
    mat = ig_matrix[1:n, 1:n].numpy()

    labels = [f"s{i}" for i in range(n_reasoning)] + ["ANSWER"]

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), max(5, 0.55 * len(labels))))
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Target step (effect)")
    ax.set_ylabel("Source step (cause)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|IG attribution| sum")
    fig.tight_layout()
    plt.savefig("../figures/ig_heatmap.png")
    plt.show()