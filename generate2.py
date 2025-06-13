from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
import pandas as pd
import re

df = pd.read_csv("./finetune/cleaned.csv")

model_id = "google/gemma-3-1b-it"

device = "mps" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)  # Ensures reproducibility
model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

def generate_response(question: str, system_prompt= "You are a helpful assistant, give one line answer.") -> str:
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt},] # system prompt same
            },
            {
                "role": "user", # user prompt, the question or the task, this will be the question or the task that the user wants to solve
                "content": [{"type": "text", "text": question},]
            },
        ],
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device).to(torch.bfloat16)


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
    return postCleaned




# Apply to DataFrame (progress-aware if many rows)
from tqdm import tqdm
tqdm.pandas()

prompt = """
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

df["gemma_answer"] = df["questionText"].progress_apply(generate_response, system_prompt=prompt)

# Save the new DataFrame
df[["questionID", "questionText", "answerText", "gemma_answer", "upvotes", "views"]].to_csv("cleaned_with_gemma_answers.csv", index=False)
print(df[["questionID", "questionText", "answerText", "gemma_answer"]].head())