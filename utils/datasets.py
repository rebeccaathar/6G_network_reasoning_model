from datasets import load_dataset 
import json
import glob

def load_math500(num_samples=1, start=3):
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return dataset.select(range(start, start + num_samples))

def load_omni_math(num_samples=1, start=3):
    dataset = load_dataset("KbsdJames/Omni-MATH", split="test")
    # dataset = dataset.filter(lambda x: x["difficulty"] == 9)

    return dataset.select(range(start, start + num_samples))

def load_AIME_2024(num_samples=1, start=0):
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return dataset.select(range(start, start + num_samples))

def load_sla_violation(num_samples, start):
    with open("sla_violation_prediction_subset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if start + num_samples > len(data):
        raise ValueError(f"Requested {num_samples} samples from index {start}, but only {len(data)} items available.")
    return data[start:start + num_samples]
        
def extract_sla_violation(example):
    question = example["question"]
    options = example["options"]
    answer = example["correct"]

    return question, options, answer

def extract_qa_omni_math(example):
    question = example["problem"]
    answer = example["answer"]

    return question, answer

def extract_qa_math500(example):
    question = example["problem"]
    answer = example["answer"]

    return question, answer

def extract_qa_aime24(example):
    question = example["Problem"]
    answer = example["Answer"]

    return question, answer


def filter_by_task_name(data, task_name="SLA Violation Prediction"):
    """Return only the questions matching task_name, keeping episode_id context."""
    matches = []
    for q in data.get("questions", []):
        if q.get("task_name") == task_name:
            matches.append({
                "episode_id": data.get("episode_id"),
                **q
            })
    return matches


def create_dataset(file_path = "../6GBench_3k_Validated/mcq_questions_only/*.mcq.json"): 
    all_matches = []
    for filepath in glob.glob(file_path, recursive=True):
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue  
        all_matches.extend(filter_by_task_name(data))

    print(f"Found {len(all_matches)} 'SLA Violation Prediction' questions")

    with open("sla_violation_prediction_subset.json", "w") as f:
        json.dump(all_matches, f, indent=2)

