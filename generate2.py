from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch

model_id = "google/gemma-3-1b-it"

device = "mps" if torch.cuda.is_available() else "cpu"
model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},] # system prompt same
        },
        {
            "role": "user", # user prompt, the question or the task, this will be the question or the task that the user wants to solve
            "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company, in one line"},]
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
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(outputs)
outputs = tokenizer.batch_decode(outputs)
print(outputs)

# ["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company<end_of_turn>\n<start_of_turn>model\nOkay, here's a poem about Hugging Face, aiming to capture its essence and impact:\n\n---\n\nThe algorithm's a gentle hand,\nAcross the data, a shifting sand.\nHugging Face, a vibrant hue,\nOf models born, for me and you.\n\nFrom transformers, a"]


#["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company, in one line<end_of_turn>\n<start_of_turn>model\nHere's a poem about Hugging Face:\n\nHugging Face: Building the neural network's heart.<end_of_turn>"]