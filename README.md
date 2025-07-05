# CounselChatDataPrep

## Overview
This repository contains the data preparation process for fine-tuning a language model on counseling-related data. The data cleaning and saving scripts are located in the `finetune` folder, implemented in the `CounselChatDataPrep.ipynb` Jupyter notebook. The notebook processes raw data, cleans it, and generates two JSONL files for training and classification tasks.

## Data Preparation
- **Location**: All data cleaning and saving logic is housed in `finetune/CounselChatDataPrep.ipynb`.
- **Input Data**: A Pandas DataFrame with columns: `questionID`, `questionTitle`, `questionText`, `questionLink`, `topic`, `therapistInfo`, `therapistURL`, `answerText`, `upvotes`, `views`.
- **Cleaning**: Rows with missing `questionText`, `answerText`, or `topic` are dropped.
- **Filtering**: The notebook selects the highest-scoring rows (based on `upvotes` + `views`) for each unique `questionText`, with the option to adjust the number of top entries (`n`).
- **Output Files**:
  - `finetune/qa_train.jsonl`: Q&A instruction-tuning dataset with system role, user questions, and assistant answers.
  - `finetune/classification_train.jsonl`: Classification dataset with system role, user questions, and topic labels.

## Training with the Data
To fine-tune a model using the generated datasets, use the `swift sft` command from the `ms-swift` framework. Below is a sample command to get started:

```
swift sft --model LLM-Research/gemma-3-1b-it --dataset ./finetune/qa_train.jsonl ./finetune/classification_train.jsonl --train_type lora --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lora_rank 8 --lora_alpha 32 --target_modules all-linear --gradient_accumulation_steps 16 --eval_steps 50 --save_steps 50 --save_total_limit 2 --logging_steps 5 --max_length 2048 --output_dir output
```
## Notes

- Check CounselChatDataPrep.ipynb for the n parameter to control the number of top-scoring rows per question.

- Monitor GPU usage with `sudo powermetrics --samplers gpu_power -i 1000 -n 10` (for MPS) during training.

Happy fine-tuning!