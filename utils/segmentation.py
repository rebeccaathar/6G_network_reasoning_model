import re
import torch

SEGMENTATION_MARKER = "<step>"


def build_segmentation_prompt(reasoning_text: str) -> str:
    return rf"""You will be given a chain of mathematical reasoning written by another model.

Your task: identify the boundaries between distinct logical steps (a
definition, a formula derivation, a substitution, a transformation, a case
split, or a self-contained computation block -- e.g. converting several
fractions to a common denominator counts as ONE step, not one per
fraction). Do not split an introductory phrase (one ending in ':' or a
short connector like 'So,', 'Then,') away from the content that completes
it -- keep them in the same step.

For EACH step, quote VERBATIM the first 5 to 10 words of where that step
begins in the text below -- copy them exactly, character for character,
including punctuation and symbols. Do not paraphrase the snippet.

Return ONLY a JSON array of these starting snippets, in the order they
appear, one per step. No explanation, no markdown fences, nothing else.

Example output format:
["Let me denote", "Substituting equation 1 into", "Now, simplifying the left side"]

Reasoning to segment:
\"\"\"
{reasoning_text}
\"\"\"
"""


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def segment_with_llm(seg_tokenizer, seg_model, reasoning_text,
                      fallback_fn=None, max_new_tokens=2048):

    reasoning_text = reasoning_text.strip()
    if not reasoning_text:
        return []

    messages = [{"role": "user", "content": build_segmentation_prompt(reasoning_text)}]
    prompt_text = seg_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = seg_tokenizer(prompt_text, return_tensors="pt").to(seg_model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = seg_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy 
            pad_token_id=seg_tokenizer.eos_token_id,
        )
    segmented_raw = seg_tokenizer.decode(
        out[0][prompt_len:], skip_special_tokens=True
    ).strip()

    steps = [s.strip() for s in segmented_raw.split(SEGMENTATION_MARKER) if s.strip()]

    reconstructed = " ".join(steps)
    if _normalize(reconstructed) != _normalize(reasoning_text):
        if fallback_fn is not None:
            return fallback_fn(reasoning_text)
        return [reasoning_text]

    return steps