import torch
import yaml
from transformers import AutoTokenizer, Gemma3ForCausalLM
from datasets import load_dataset
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
torch.manual_seed(123)

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

def generate_response(model, tokenizer, messages, config, device):
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
    return response[0]

def generate_dataset_csv(config_path="config.yaml", output_path="dataset_questions.csv"):
    config = load_config(config_path)
    
    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"],
        trust_remote_code=True
    )
    
    # Get all unique question IDs
    question_ids = set(row["questionID"] for row in dataset)
    print(f"Found {len(question_ids)} unique questions in the dataset")

    results = []
    
    for question_id in question_ids:
        try:
            best_entry = get_best_answer_by_question_id(dataset, question_id)
            results.append({
                "question_id": question_id,
                "question": best_entry["questionText"],
                "answer": best_entry["answerText"],
                "views": best_entry.get("views", 0),
                "upvotes": best_entry.get("upvotes", 0)
            })
        except Exception as e:
            print(f"Error processing question ID {question_id}: {str(e)}")
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nDataset questions and answers saved to {output_path}")
    print(f"Total questions processed: {len(results)}")

def process_single_question(question_id, dataset_df, model, tokenizer, config, device):
    print(f"\n{'='*20} Processing Question ID: {question_id} {'='*20}")
    
    # Get the question and answer from the dataset
    question_row = dataset_df[dataset_df['question_id'] == question_id].iloc[0]
    question = question_row['question']
    dataset_answer = question_row['answer']

    print("\n=== Question and Best Answer from Dataset ===")
    print(f"Question ID: {question_id}")
    print(f"Client: {question}")
    print(f"Therapist: {dataset_answer}")
    print("=" * 50)

    # Initialize results dictionary for this question
    results = {
        "question_id": question_id,
        "question": question,
        "answer": dataset_answer,
        "llm_zero_shot": "",
        "llm_one_shot": "",
        "llm_few_shots": ""
    }

    strategies = ["zero-shot", "one-shot", "few-shot"] if config["prompting"]["strategy"] == "all" else [config["prompting"]["strategy"]]

    for strategy in strategies:
        print(f"\n=== Running {strategy.upper()} Strategy ===")
        
        few_shot_examples = get_prompt_strategy(
            dataset_df.to_dict('records'),
            strategy,
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

        model_answer = generate_response(model, tokenizer, messages, config, device)
        
        # Calculate similarity between model's answer and dataset answer
        similarity_score = calculate_similarity(model_answer, dataset_answer)
        
        print(f"\nModel's Response ({strategy}):")
        print(model_answer)
        print(f"\nSimilarity Score: {similarity_score:.4f}")
        print("=" * 50)

        # Store the result
        results[f"llm_{strategy.replace('-', '_')}"] = model_answer

    return results

def main():
    config = load_config()
    
    dataset_df = pd.read_csv("best-answers-counsel-chat.csv")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gemma3ForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=getattr(torch, config["model"]["dtype"]),
        device_map=device
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    question_ids = config.get("dataset", {}).get("question_ids", [])
    if not question_ids:
        question_ids = dataset_df['question_id'].unique().tolist()

    all_results = []

    for question_id in question_ids:
        try:
            results = process_single_question(question_id, dataset_df, model, tokenizer, config, device)
            all_results.append(results)
        except Exception as e:
            print(f"Error processing question ID {question_id}: {str(e)}")
            continue

    df = pd.DataFrame(all_results)
    csv_path = config.get("output", {}).get("csv_path", "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate responses using different strategies')
    parser.add_argument('--question_id', type=str, help='Specific question ID to process')
    parser.add_argument('--num_questions', type=int, help='Number of questions to process starting from the beginning')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--output', default='results.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    if args.question_id:
        # Load config and modify it to only process the specified question
        config = load_config(args.config)
        config["dataset"]["question_ids"] = [args.question_id]
        config["output"]["csv_path"] = args.output
        main()
    elif args.num_questions:
        config = load_config(args.config)
        dataset_df = pd.read_csv("best-answers-counsel-chat.csv")
        question_ids = dataset_df['question_id'].unique()[:args.num_questions].tolist()
        config["dataset"]["question_ids"] = question_ids
        config["output"]["csv_path"] = args.output
        main()
    else:
        main()

# ["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company<end_of_turn>\n<start_of_turn>model\nOkay, here's a poem about Hugging Face, aiming to capture its essence and impact:\n\n---\n\nThe algorithm's a gentle hand,\nAcross the data, a shifting sand.\nHugging Face, a vibrant hue,\nOf models born, for me and you.\n\nFrom transformers, a"]


#["<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite a poem on Hugging Face, the company, in one line<end_of_turn>\n<start_of_turn>model\nHere's a poem about Hugging Face:\n\nHugging Face: Building the neural network's heart.<end_of_turn>"]