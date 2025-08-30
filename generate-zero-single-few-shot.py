from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
import pandas as pd
import re
import random
import yaml
from tqdm import tqdm
import os
import warnings
import csv
os.environ["PYTORCH_ENABLE_TRITON"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Fully disables torch.compile()
warnings.filterwarnings("ignore")

def load_config(config_path="zero-single-few-shots-config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

model_id = "google/gemma-3-1b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.manual_seed(42)

model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map=device
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

try:
    full_df = pd.read_csv("./finetune/cleaned.csv")
    if len(full_df) == 0:
        raise ValueError("Dataset is empty")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Limit actual questions if configured
df = full_df.copy()
if config['dataset'].get('num_questions', 0) > 0:
    df = df.head(int(config['dataset']['num_questions']))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_examples(strategy, full_data, num_examples=3):
    if strategy == "zero-shot":
        return []

    records = full_data.to_dict('records')
    if len(records) == 0:
        return []

    if strategy == "one-shot":
        ex = random.choice(records)
        # for fix example
        #ex = full_data[full_data["questionID"] == 715].to_dict("records")[0]
        return [
            {"role": "user", "content": [{"type": "text", "text": f"Example 1:\nClient: {clean_text(ex['questionText'])}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"Therapist: {clean_text(ex['answerText'])}"}]}
        ]

    if strategy == "few-shot":
        examples = random.sample(records, min(num_examples, len(records)))
        # speicific example one
        #ids = [715, 913, 486]
        #examples = full_data[full_data["questionID"].isin(ids)].to_dict("records")
        messages = []
        for idx, ex in enumerate(examples):
            messages.append({"role": "user", "content": [{"type": "text", "text": f"Example {idx+1}:\nClient: {clean_text(ex['questionText'])}"}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"Therapist: {clean_text(ex['answerText'])}"}]})
        return messages

    return []

def generate_response(question: str, strategy: str, full_data, question_index: int) -> str:
    try:
        examples = get_examples(strategy, full_data, config['dataset'].get('num_few_shot_examples', 3))
        question_cleaned = clean_text(question)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": config['prompting']['system_prompt']}]},
            *examples,
            {"role": "user", "content": [{"type": "text", "text": "---\nNow respond to the following:\nClient: " + question_cleaned}]}
        ]

        print(f"\n➡️ Question {question_index + 1} | Strategy: {strategy}")

        input_tensor = tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        outputs = model.generate(
            input_tensor,
            max_new_tokens=config['prompting']['max_new_tokens'],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        generated = outputs[0][input_tensor.shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        return clean_text(response)

    except Exception as e:
        print(f"❌ Error in {strategy}: {str(e)}")
        return "ERROR"

csv_path = config['output']['csv_path']
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

fieldnames = ["question", "answer"] + [
    f"gemma_{s.replace('-', '_')}" for s in (
        ["zero-shot", "one-shot", "few-shot"]
        if config['prompting']['strategy'] == "all"
        else [config['prompting']['strategy']]
    )
]

# Load already processed questions if results.csv exists
processed_questions = set()
if os.path.exists(csv_path):
    try:
        existing_df = pd.read_csv(csv_path)
        processed_questions = set(existing_df["question"].apply(clean_text))
    except Exception as e:
        print(f"⚠️ Couldn't load existing results.csv: {e}")

write_header = not os.path.exists(csv_path)
csv_file = open(csv_path, "a", newline='', encoding='utf-8')
csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, escapechar="\\")

if write_header:
    csv_writer.writeheader()

# Run generation
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    question = clean_text(row["questionText"])

    if question in processed_questions:
        print(f"⏭️ Skipping already processed question {idx + 1}")
        continue
    #skip logic
    #if row["questionID"] in [715, 913, 486]:
        #print(f"⏭️ Skipping fixed example questionID {row['questionID']}")
        #continue
    answer = clean_text(row["answerText"])
    row_data = {
        "question": question,
        "answer": answer,
    }

    for strategy in fieldnames[2:]:
        s = strategy.replace('gemma_', '').replace('_', '-')
        row_data[strategy] = generate_response(question, s, full_data=full_df, question_index=idx)

    csv_writer.writerow(row_data)
    csv_file.flush()

csv_file.close()
print(f"\n✅ Streamed results saved to {csv_path}")

# Preview
print("\n📋 Sample output:")
print(pd.read_csv(csv_path).head(2).to_markdown(index=False))
