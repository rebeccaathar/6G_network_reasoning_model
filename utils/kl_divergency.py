import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def compute_kl_for_suppression(logits_base, logits_supp):
    logp_base = F.log_softmax(logits_base, dim=-1)
    logp_supp = F.log_softmax(logits_supp, dim=-1)
    return F.kl_div(logp_supp, logp_base, log_target=True, reduction="none").sum(-1)

def plot_flow_heatmap(analysis, title="Information Flow Heatmap "):
    n = analysis["flow_matrix"].shape[0]
    n_reasoning = analysis["n_reasoning"]
    mat = analysis["flow_matrix"][1:n, 1:n].numpy()

    labels = [f"s{i}" for i in range(n_reasoning)] + ["ANSWER"]

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), max(5, 0.55 * len(labels))))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Target sentence s'' (effect)")
    ax.set_ylabel("Source sentence s' (cause)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=" Mean KL (logits base vs. supressed)")
    fig.tight_layout()
    plt.savefig("./figures/heatmap.png")
    plt.show()

def plot_sentence_relevance(analysis, title="Relevance fo each step"):
    n_reasoning = analysis["n_reasoning"]
    idxs = np.arange(n_reasoning)
    ans_rel = analysis["sentence_relevance"][1:1 + n_reasoning].numpy()
    down_imp = analysis["downstream_importance"][1:1 + n_reasoning].numpy()
    score = analysis["relevance_score"][1:1 + n_reasoning].numpy()

    width = 0.27
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n_reasoning), 4))
    ax.bar(idxs - width, ans_rel, width, label="Relevance to the answer")
    ax.bar(idxs, down_imp, width, label="Downstream influence")
    ax.bar(idxs + width, score, width, label="Score (weighted combination)")
    ax.set_xticks(idxs)
    ax.set_xticklabels([f"s{i}" for i in idxs])
    ax.set_xlabel("reasoning step (sentence)")
    ax.set_ylabel("Mean KL divergence (nats)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig("./figures/sentence_relevance.png")
    plt.show()