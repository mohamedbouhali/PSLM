from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
from datasets import load_dataset




ds = load_dataset("nbertagnolli/counsel-chat")
print("\nColumn names:")
print(ds['train'].column_names)

# Print the value of 'questionText' column of the first row
#print("\nValue of 'questionText' column of the first row:")
#print(ds['train'][0]['questionText'])




model_id = "google/gemma-3-1b-it"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

print(f"Model is running on device: {model.device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_message = ds['train'][0]['questionText']
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant. Reply with a single line."},] # system prompt same
        },
        {
            "role": "user", # user prompt, the question or the task, this will be the question or the task that the user wants to solve
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
).to(model.device).to(torch.bfloat16)



with torch.inference_mode():
    print("This is the input message: \n", input_message)
    outputs = model.generate(**inputs, max_new_tokens=512)
    #print(outputs)
outputs = tokenizer.batch_decode(outputs,skip_special_tokens=True)
#outputs_model=outputs[0].split('<start_of_turn>model\n')[1].replace('<end_of_turn>', '').strip()
#print("This is the ouput message: \n", outputs_model)
print("This is the ouput message:")
print(outputs[0])







# ["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company<end_of_turn>\n<start_of_turn>model\nOkay, here's a poem about Hugging Face, aiming to capture its essence and impact:\n\n---\n\nThe algorithm's a gentle hand,\nAcross the data, a shifting sand.\nHugging Face, a vibrant hue,\nOf models born, for me and you.\n\nFrom transformers, a"]
#["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company, in one line<end_of_turn>\n<start_of_turn>model\nHere's a poem about Hugging Face:\n\nHugging Face: Building the neural network's heart.<end_of_turn>"]