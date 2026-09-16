import torch
import re
from torch.nn import functional as F
from .token_utils import find_boxed
import numpy as np

def extract_final_answer_text(text, marker="Answer:"):
    idx = text.lower().find(marker.lower())
    if idx == -1:
        return None
    after = text[idx + len(marker):]
    end_match = re.search(r'\n\s*\n', after)
    candidate = after[: end_match.start()] if end_match else after
    return candidate.strip()

def extract_reasoning_and_answer(generated_text):
    after_prompt = generated_text.strip()
    think_close_match = re.search(r"</think>", after_prompt, re.IGNORECASE)

    marker_idx = after_prompt.lower().find("Answer:")
    if marker_idx != -1:
        content = extract_final_answer_text(after_prompt)
        reasoning_text = after_prompt[: think_close_match.end()] \
            if (think_close_match and think_close_match.end() <= marker_idx) \
            else after_prompt[:marker_idx]
        answer_chunk = f"Answer: {content} "
        return reasoning_text.strip(), answer_chunk

    boxed = find_boxed(after_prompt)
    if boxed:
        start, end, content = boxed
        reasoning_text = after_prompt[: think_close_match.end()] \
            if (think_close_match and think_close_match.end() <= start) \
            else after_prompt[:start]
        answer_chunk = f"Answer: {content.strip()} "
        return reasoning_text.strip(), answer_chunk

    return after_prompt, None




import re

TRANSITION_MARKERS = (
    r"(?:First|Next|Then|Now|So|Therefore|Thus|Finally|Also|Similarly|"
    r"Substituting|Simplifying|Since|Given that|Note that|Alternatively)\b"
)


def _is_bare_computation(line):
    """True if a line is mostly a standalone equation/value continuing a
    prior computation, rather than a new sentence-level reasoning move."""
    core = re.sub(r'^(?:and|or)\s+', '', line.strip(), flags=re.IGNORECASE)
    words = re.findall(r'[A-Za-z]{3,}', core)
    filler = {'and', 'the', 'are', 'for', 'with', 'now'}
    meaningful_words = [w for w in words if w.lower() not in filler]
    has_math = bool(re.search(r'[=><]|\d+\s*/\s*\d+|\\frac', core))
    return has_math and len(meaningful_words) <= 1


def _is_open_continuation(prev_line):
    """True if the previous line ends in a way that clearly demands a
    continuation: a trailing colon, or a short dangling connector ending
    in a comma (e.g. 'So,', 'Then,', 'Substituting,')."""
    if prev_line.endswith(':'):
        return True
    if prev_line.endswith(','):
        # only treat as an open connector if it's short (a lead-in phrase,
        # not a long clause that just happens to end mid-sentence)
        word_count = len(re.findall(r'[A-Za-z]+', prev_line))
        return word_count <= 4
    return False

DANGLING_CONNECTOR = re.compile(
    r'^(?:First|Next|Then|Now|So|Therefore|Thus|Finally|Also|Similarly|'
    r'Substituting|Simplifying|Since|Given that|Note that|Alternatively|Hence)'
    r'\s*[,:]?\s*$',
    re.IGNORECASE
)

def _premerge_dangling_connectors(lines):
    """Force-join any line that is ONLY a bare connector word with the
    line that follows it, regardless of any other merge logic."""
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if DANGLING_CONNECTOR.match(line) and i + 1 < len(lines):
            merged.append(f"{line} {lines[i + 1]}")
            i += 2
        else:
            merged.append(line)
            i += 1
    return merged

def split_into_sentences(text):
    text = text.strip()
    if not text:
        return []

    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE).strip()
    if not text:
        return []
    text = text.replace('\r\n', '\n')

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    lines = _premerge_dangling_connectors(lines)   # <-- guarantees the "So," case is fixed

    steps = []
    current_step = []
    for line in lines:
        is_unordered_bullet = bool(re.match(r'^(?:[-*\u2022])\s+', line))
        prev = current_step[-1] if current_step else None

        prev_open = bool(prev) and _is_open_continuation(prev)
        bare_computation = bool(current_step) and _is_bare_computation(line)
        bullet_continuation = is_unordered_bullet and bool(current_step)

        should_merge = prev_open or bare_computation or bullet_continuation

        if should_merge:
            current_step.append(line)
        else:
            if current_step:
                steps.append(" ".join(current_step))
            current_step = [line]

    if current_step:
        steps.append(" ".join(current_step))

    final_steps = []
    for step in steps:
        sub_steps = re.split(fr'(?<=[.?!])\s+(?={TRANSITION_MARKERS})', step)
        for sub in sub_steps:
            sub_clean = sub.strip()
            if sub_clean:
                final_steps.append(sub_clean)

    deduped = []
    for s in final_steps:
        if not deduped or s != deduped[-1]:
            deduped.append(s)

    return deduped

def split_into_steps_by_tag(text):
    pattern = re.compile(r"<episode_(\d+)>(.*?)</episode_\1>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    steps = [re.sub(r"\s+", " ", m[1]).strip() + " " for m in matches if m[1].strip()]
    return steps


def split_into_steps(text):
    tagged = split_into_steps_by_tag(text)
    if tagged:
        return tagged
    return split_into_sentences(text)



def save_cot_relevance_txt(
    analysis, filepath, generated_text=None, question=None, gt_answer=None
):
    import os
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_reasoning = analysis["n_reasoning"]
    all_chunks = analysis["all_chunks"]
    sentence_relevance = analysis["sentence_relevance"]
    downstream_importance = analysis["downstream_importance"]
    relevance_score = analysis["relevance_score"]
    entropy_delta = analysis["entropy_delta"]

    with open(filepath, "w", encoding="utf-8") as f:
        if question is not None:
            f.write(f"Pergunta: {question}\n")
        if gt_answer is not None:
            f.write(f"Gabarito: {gt_answer}\n")
        if question is not None or gt_answer is not None:
            f.write("\n")

        if generated_text is not None:
            f.write("=" * 80 + "\n")
            f.write("COMPLETE REASONING GENERATED BY THE MODEL\n")
            f.write("=" * 80 + "\n")
            f.write(generated_text.strip() + "\n")
            f.write("=" * 80 + "\n\n")

        f.write(
            f"{'Idx':<5}{'Step':<60}\n"
        )

        for i in range(1, n_reasoning + 1):
            text = all_chunks[i].strip().replace("\n", " ")
            f.write(
                f"{i - 1:<5}{text}\n"
            )

    print(f"Complete Reasoning saved in : {filepath}")