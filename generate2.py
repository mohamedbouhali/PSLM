import torch
import yaml
from transformers import AutoTokenizer, Gemma3ForCausalLM
from datasets import load_dataset
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(text1, text2, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_prompt_strategy(dataset, strategy, num_examples=3):
    if strategy == "zero-shot":
        return []
    elif strategy == "one-shot":
        # Get all unique question IDs
        question_ids = set(row["questionID"] for row in dataset)
        # Randomly select one question ID
        selected_qid = random.choice(list(question_ids))
        # Get the best answer for this question
        best_answer = get_best_answer_by_question_id(dataset, selected_qid)
        return [{
            "role": "user",
            "content": [{"type": "text", "text": "Here is an example of a client's question and my response:\n\nClient: " + best_answer["questionText"]}]
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": "Therapist: " + best_answer["answerText"]}]
        }]
    elif strategy == "few-shot":
        # Get all unique question IDs
        question_ids = set(row["questionID"] for row in dataset)
        # Randomly select num_examples question IDs
        selected_qids = random.sample(list(question_ids), num_examples)
        few_shot_examples = []
        for qid in selected_qids:
            # Get the best answer for each selected question
            best_answer = get_best_answer_by_question_id(dataset, qid)
            few_shot_examples.extend([
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Example {len(few_shot_examples)//2 + 1}:\n\nClient: {best_answer['questionText']}"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"Therapist: {best_answer['answerText']}"}]
                }
            ])
        return few_shot_examples
    return []

def get_best_answer_by_question_id(dataset, question_id):
    answers = [row for row in dataset if row["questionID"] == question_id]
    if not answers:
        raise ValueError(f"No answers found for questionID: {question_id}")

    answers.sort(key=lambda x: (x.get("views", 0), x.get("upvotes", 0)), reverse=True)
    return answers[0]

def main():
    config = load_config()

    dataset = load_dataset(
        "nbertagnolli/counsel-chat",
        split=config["dataset"]["split"],
        trust_remote_code=True
    )

    question_id = config.get("dataset", {}).get("target_question_id", None)
    if not question_id:
        raise ValueError("Missing 'target_question_id' in config.yaml under 'dataset'")

    best_entry = get_best_answer_by_question_id(dataset, question_id)
    question = best_entry["questionText"]
    dataset_answer = best_entry["answerText"]

    print("\n--- Question from Dataset ---")
    print(f"Question ID: {question_id}")
    print(f"Client: {question}")
    print("\n--- Best Answer from Dataset ---")
    print(f"Therapist: {dataset_answer}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gemma3ForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=getattr(torch, config["model"]["dtype"]),
        device_map=device
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    few_shot_examples = get_prompt_strategy(
        dataset,
        config["prompting"]["strategy"],
        config["dataset"]["num_few_shot_examples"]
    )

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": config["prompting"]["system_prompt"]}]
            },
            *few_shot_examples,
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Client: {question}"}]
            }
        ]
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device).to(getattr(torch, config["model"]["dtype"]))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["prompting"]["max_new_tokens"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
        )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    model_answer = response[0]
    
    # Calculate similarity between model's answer and dataset answer
    similarity_score = calculate_similarity(model_answer, dataset_answer)
    
    print("\n--- Model's Response ---")
    print(model_answer)
    print("\n--- Similarity Analysis ---")
    print(f"Cosine Similarity Score: {similarity_score:.4f}")
    print("(Score ranges from -1 to 1, where 1 indicates perfect similarity)")

if __name__ == "__main__":
    main()

# ["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company<end_of_turn>\n<start_of_turn>model\nOkay, here's a poem about Hugging Face, aiming to capture its essence and impact:\n\n---\n\nThe algorithm's a gentle hand,\nAcross the data, a shifting sand.\nHugging Face, a vibrant hue,\nOf models born, for me and you.\n\nFrom transformers, a"]


#["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company, in one line<end_of_turn>\n<start_of_turn>model\nHere's a poem about Hugging Face:\n\nHugging Face: Building the neural network's heart.<end_of_turn>"]