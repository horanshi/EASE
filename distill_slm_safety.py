import os
import torch
import random
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import argparse
from trl import DataCollatorForCompletionOnlyLM


def parse_args():
    parser = argparse.ArgumentParser(description='Full finetuning for Qwen2.5-1.5B or Llama3.2-1B models')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B',
                        help='Model name, options: Qwen/Qwen2.5-1.5B or meta-llama/Llama-3-1.1B')
    parser.add_argument('--output_dir', type=str, default='./finetuned-model',
                        help='Output directory')
    parser.add_argument("--dataset_name", type=str, default="thu-ml/STAIR-Prompts", help="Dataset name")
    parser.add_argument('--num_train_epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2,
                        help='Batch size per device for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=8192,
                        help='Maximum sequence length')
    parser.add_argument('--samples_per_dataset', type=int, default=20000,
                        help='Number of samples to randomly select from datasets (default: 20000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='stair',
                        help='Dataset options: stair (all data) or stair_ultrafeedback (only UltraFeedback source)')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help='Overwrite the output directory to avoid consuming extra disk space')
    return parser.parse_args()


def format_dataset(example, tokenizer, max_length):
    """
    Format general dataset samples (prompt, answer format)
    """
    eos_token = tokenizer.eos_token

    prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['original_question']}<|im_end|>\n<|im_start|>assistant\n<think>{example['response']}<|im_end|>"
    #prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#Cutting Knowledge Date: December 2023
#Today Date: 23 July 2024
#You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
#{example['original_question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#{example['response']}<|eot_id|>"""

    encoded = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None
    )

    return encoded


def randomly_sample_dataset(dataset, num_samples, seed):
    """
    Randomly sample a fixed number of examples from a dataset
    """
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Get the total size of the dataset
    dataset_size = len(dataset)

    # If requested samples are more than available, take all samples
    if num_samples >= dataset_size:
        print(
            f"Warning: Requested {num_samples} samples but dataset only has {dataset_size} examples. Using all available examples.")
        return dataset

    # Generate random indices without replacement
    indices = random.sample(range(dataset_size), num_samples)

    # Select only those random examples
    sampled_dataset = dataset.select(indices)

    print(f"Randomly sampled {len(sampled_dataset)} examples from a total of {dataset_size}")
    return sampled_dataset


# Custom optimizer creation function
def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer


def main():
    args = parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load pretrained model and tokenizer - WITHOUT device_map for data parallel
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        # No device_map specified - important for data parallel training
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    print(f"Added custom padding token: {tokenizer.pad_token}")
    special_tokens_dict = {'additional_special_tokens': ['<think>', '</think>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Added special tokens: {special_tokens_dict['additional_special_tokens']}")

    # Update tokenizer with new max length if possible
    if hasattr(tokenizer, 'model_max_length'):
        tokenizer.model_max_length = args.max_length

    # Load general dataset
    print("Loading dataset...")
    dataset_safety = load_dataset('json', data_files=args.dataset_name)
    print("General dataset structure:", dataset_safety)

    # Process general dataset
    if 'train' in dataset_safety:
        print("\nFirst sample in general training set:")
        print(dataset_safety['train'][0])
        safety_train_data = dataset_safety["train"]
    else:
        print("\nFirst sample in general dataset:")
        print(dataset_safety[list(dataset_safety.keys())[0]][0])
        safety_train_data = dataset_safety[list(dataset_safety.keys())[0]]

    # Sample from general dataset
    if args.samples_per_dataset > 0 and args.samples_per_dataset < len(safety_train_data):
        print(f"Limiting general dataset to {args.samples_per_dataset} samples")
        sampled_safety_data = randomly_sample_dataset(safety_train_data, args.samples_per_dataset, args.seed)
    else:
        sampled_safety_data = safety_train_data
        print(f"Using all {len(sampled_safety_data)} samples from general dataset")

    # Set model-specific dialog format templates for DataCollatorForCompletionOnlyLM
    if "qwen" in args.model_name.lower():
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant"
    else:  # Llama
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>"

    # Process general dataset
    print("Processing general dataset...")
    processed_safety_data = sampled_safety_data.map(
        lambda x: format_dataset(x, tokenizer, args.max_length),
        remove_columns=sampled_safety_data.column_names,
        batched=False,
        desc="Processing general dataset examples"
    )

    # Use trl library's DataCollatorForCompletionOnlyLM to train only on assistant responses
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        response_template=response_template,
        mlm=False
    )

    # Shuffle the combined dataset
    training_dataset =processed_safety_data.shuffle(seed=args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.03,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none", 
        dataloader_drop_last=True,  
        lr_scheduler_type="cosine",  
        max_grad_norm=1.0,  
        seed=args.seed,  
        save_total_limit=1,  
        overwrite_output_dir=args.overwrite_output_dir,  
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(get_optimizer(model, training_args), None),  
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    trainer.save_state()

    print("Training completed!")


if __name__ == "__main__":
    main()
