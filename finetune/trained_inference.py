import pandas as pd
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
import random
import os
import re
import numpy as np

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

model_id = "google/gemma-3-1b-it"
peft_model_id = "/home/ge53wex/PSLM/finetune/output/lora-16/v0-20250705-215100/checkpoint-43/"
config = PeftConfig.from_pretrained(peft_model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
base_model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map=device
    ).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = PeftModel.from_pretrained(base_model, peft_model_id).eval()



# Set seeds for reproducibility
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)  # if using CUDA
random.seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def generate_answer_question(best_answer_file):
    # Read the input CSV file
    data = pd.read_csv(best_answer_file)

    # Create lists to store results
    questions = []
    predictions = []
    answers = []
    system_prompt = ''
    with open('system_prompt.txt', 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    print(f"System prompt loaded:\n {system_prompt}")

    # Process each question from the CSV
    #data = data.iloc[501:800]
    for idx, (_, row) in enumerate(data.iterrows(), 1):
        input_message = row['questionText']
        print("inference question:\n", input_message)
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt},]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": input_message},]
                },
            ],
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True)

        outputs = tokenizer.batch_decode(outputs)[0]
        postCleaned = ""
        # Clean output: extract model response after `<start_of_turn>model\n`
        match = re.search(r"<start_of_turn>model\n(.+?)<end_of_turn>", outputs, re.DOTALL)
        if match:
            postCleaned = match.group(1).strip()
        else:
            # Fallback: remove special tokens manually
            postCleaned = outputs.replace("<bos>", "").split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
       
         # Store results
        print("inference answer:\n", postCleaned)
        questions.append(input_message)
        predictions.append(postCleaned)
        answers.append(row['answerText'])
        print(f"Inference progress: {idx}/{len(data)}")

    # Create output DataFrame and save to CSV
    output_df = pd.DataFrame({
        'question': questions,
        'answer': answers,
        'llm_prediction': predictions
        
    })
    
    # Save to CSV with fixed name
    output_file = 'lora-16.csv'
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    best_answer_file = "/home/ge53wex/PSLM/finetune/test_split.csv"
    generate_answer_question(best_answer_file)
