from transformers import StoppingCriteria
import re

def find_boxed(text):
    marker = r"\boxed{"
    idx = text.find(marker)
    if idx == -1:
        return None
    i = idx + len(marker)
    depth = 1
    start_content = i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None 
    return idx, i, text[start_content:i - 1]

def extract_answer_tail(answer_tail):
    """
    Extracts a single option letter (A, B, C, D, ...) from the model's
    stage-2 continuation
    """
    text = answer_tail.strip()

    # 1) Optional \boxed{...} wrapper, if the dataset is in a math domain 
    boxed = find_boxed(text)
    if boxed:
        _, _, content = boxed
        content = content.strip()
        m = re.match(r"^[A-Za-z]", content)
        if m:
            return m.group(0).upper()

    # A single letter possibly followed by ')' or '.' e.g. "A)", "A.", "A"
    m = re.search(r"\b([A-D])\b[\)\.]?", text)
    if m:
        return m.group(1).upper()

    line = text.split("\n", 1)[0].strip()
    line = re.sub(r"[*_`]", "", line)  
    m = re.match(r"^\s*([A-Za-z])", line)
    if m:
        return m.group(1).upper()

    return None

class StopOnAnswer(StoppingCriteria):

    def __init__(self, tokenizer, prompt_len,
                 pattern=r"Answer:\s*-?\d+(?:\.\d+)?"):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def __call__(self, input_ids, scores, **kwargs):
        generated = self.tokenizer.decode(
            input_ids[0][self.prompt_len:], skip_special_tokens=True
        )
        last_token_is_eos = input_ids[0, -1].item() == self.tokenizer.eos_token_id

        match = self.pattern.search(generated)
        if match:
            has_trailing_char = match.end() < len(generated)
            return has_trailing_char or last_token_is_eos

        boxed = find_boxed(generated)
        if boxed:
            _, end, _ = boxed
            has_trailing_char = end < len(generated)
            return has_trailing_char or last_token_is_eos

        return False

def get_sentence_token_spans(tokenizer, sentences, device):
    full_text = "".join(sentences)
    enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
 
    spans = []
    char_pos = 0
    for sent in sentences:
        char_start = char_pos
        char_end = char_pos + len(sent)
        tok_start = next(i for i, (s, e) in enumerate(offsets) if e > char_start)
        tok_end = next(
            (i for i, (s, e) in enumerate(offsets) if s >= char_end),
            len(offsets),
        )
        spans.append((tok_start, tok_end))
        char_pos = char_end

    input_ids = enc["input_ids"].to(device)
 
    return input_ids, spans

