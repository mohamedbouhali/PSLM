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
peft_model_id = "/home/ge53wex/PSLM/finetune/output/lora-4/v0-20250630-172414/checkpoint-86/"
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
    system_prompt = """
    
    You are a licensed therapist with expertise in trauma, depression, and anxiety.

    Your role is to provide compassionate emotional support to individuals seeking guidance. Your responses should:
    - Validate the user’s feelings and lived experiences.
    - Offer safe, practical, and realistic suggestions.
    - Maintain therapeutic boundaries at all times.
    - Avoid clinical diagnoses, medical advice, or treatment prescriptions.

    Please follow these instructions:
    1. Carefully read the client’s question or concern.
    2. If examples are included, match their tone and style when crafting your response.
    3. Respond with empathy, clarity, and warmth—like you would in a real-life therapy conversation.
    4. Keep the response focused, supportive, and relevant to the user’s situation.

    Always aim to be grounding, nonjudgmental, and helpful.
    
    """

    # Process each question from the CSV
    #data = data.iloc[501:800]
    for _, row in data.iterrows():
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
        ).to(device).to(torch.float16)


        with torch.inference_mode():
            print(inputs)
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            print(outputs)
        outputs = tokenizer.batch_decode(outputs)[0]
        print("Outputs before cleaning: ", outputs)
        postCleaned = ""
        # Clean output: extract model response after `<start_of_turn>model\n`
        match = re.search(r"<start_of_turn>model\n(.+?)<end_of_turn>", outputs, re.DOTALL)
        if match:
            postCleaned = match.group(1).strip()
        else:
            # Fallback: remove special tokens manually
            postCleaned = outputs.replace("<bos>", "").split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()

        # ["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company<end_of_turn>\n<start_of_turn>model\nOkay, here's a poem about Hugging Face, aiming to capture its essence and impact:\n\n---\n\nThe algorithm's a gentle hand,\nAcross the data, a shifting sand.\nHugging Face, a vibrant hue,\nOf models born, for me and you.\n\nFrom transformers, a"]
        #["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company, in one line<end_of_turn>\n<start_of_turn>model\nHere's a poem about Hugging Face:\n\nHugging Face: Building the neural network's heart.<end_of_turn>"]
        print("Outputs after cleaning: ", postCleaned)
        # return postCleaned

        # with torch.inference_mode():
        #     outputs = model.generate(**inputs, max_new_tokens=524)
        #     full_prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
        #     # More robust way to extract the model's response
        #     try:
        #         # Try to find the last occurrence of the user's message
        #         last_user_msg_index = full_prediction.rfind(input_message)
        #         if last_user_msg_index != -1:
        #             prediction = full_prediction[last_user_msg_index + len(input_message):].strip()
        #         else:
        #             # If we can't find the user's message, take the last part of the response
        #             prediction = full_prediction.split("assistant")[-1].strip()
        #             print("inference answer:\n", prediction)
        #     except Exception as e:
        #         print(f"Warning: Could not properly extract prediction for message: {input_message}")
        #         prediction = full_prediction
            
            # Store results
        print("inference answer:\n", postCleaned)
        questions.append(input_message)
        predictions.append(postCleaned)
        answers.append(row['answerText'])

    # Create output DataFrame and save to CSV
    output_df = pd.DataFrame({
        'question': questions,
        'answer': answers,
        'llm_prediction': predictions
        
    })
    
    # Save to CSV with fixed name
    output_file = 'lora-4.csv'
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    best_answer_file = "/home/ge53wex/PSLM/finetune/test_split.csv"
    generate_answer_question(best_answer_file)
