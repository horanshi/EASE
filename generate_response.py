import os
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Use VLLM for inference on Qwen models with STAIR-Prompts dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="Qwen model name to use")
    parser.add_argument("--dataset_name", type=str, default="thu-ml/STAIR-Prompts", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name containing prompts")
    parser.add_argument("--data_source", type=str, default="all",help="Data source selection, can be 'all' or specific sources like 'PKU-SafeRLHF', multiple sources separated by commas")
    parser.add_argument("--max_samples", type=int, default=None,help="Maximum number of samples to process, default is None meaning process all samples")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p parameter")
    parser.add_argument("--output_file", type=str, default="qwen_responses.jsonl", help="Output file path")
    parser.add_argument("--dataset_type", type=str, default="huggingface", choices=["huggingface", "local"],
                        help="Type of dataset: 'huggingface' for HF datasets or 'local' for local files")

    # Parallelism parameters
    parser.add_argument("--tensor_parallel_size", type=int, default=1,help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,help="GPU memory utilization ratio (0.0 to 1.0)")
    # System prompt parameters
    parser.add_argument("--system_prompt", type=str, default="", help="System prompt to prepend to all user prompts")
    parser.add_argument("--template_type", type=str, default="qwen", choices=["qwen", "llama", "chatml"],help="Chat template type to use")
    return parser.parse_args()


def apply_chat_template(prompt, system_prompt, template_type):
    """Apply chat template to format the prompt with system message"""
    if template_type == "qwen":
      # Qwen/Qwen2 format
        if system_prompt:
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    return formatted_prompt


def main():
    args = parse_args()

    # Load dataset
    print(f"Loading dataset {args.dataset_name}...")
    if args.dataset_type == "huggingface":
        dataset = load_dataset(args.dataset_name, split=args.split)
    else:
        dataset = load_dataset("json", data_files=args.dataset_name)["train"]

    # Filter dataset by data source
    if args.data_source != "all":
        data_sources = [source.strip() for source in args.data_source.split(",")]
        print(f"Filtering dataset, keeping only these data sources: {data_sources}")

        # Make sure the dataset has a source column
        if "source" in dataset.column_names:
            dataset = dataset.filter(lambda example: example["source"] in data_sources)
            print(f"Dataset size after filtering: {len(dataset)}")
        else:
            print("Warning: Dataset does not have a 'source' column, cannot filter by data source")

    # Limit sample count
    if args.max_samples is not None and args.max_samples < len(dataset):
        print(f"Limiting sample count to: {args.max_samples}")
        dataset = dataset.select(range(args.max_samples))

    # Initialize model with tensor parallelism
    print(f"Initializing model {args.model_name} with tensor_parallel_size={args.tensor_parallel_size}...")
    model = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # Get raw prompts
    raw_prompts = dataset[args.prompt_column]

    args.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    print(f"Applying chat template with system prompt")
    prompts = [apply_chat_template(p, args.system_prompt, args.template_type) for p in raw_prompts]

    # VLLM will automatically handle the batching internally in an optimal way
    outputs = model.generate(prompts, sampling_params)

    # Collect results
    all_responses = []
    for output in outputs:
        all_responses.append(output.outputs[0].text.strip())

    # Create new dataset
    result_dataset = Dataset.from_dict({
        "original_question": raw_prompts,
        "response": all_responses
    })

    # Save results
    print(f"Saving results to {args.output_file}...")
    result_dataset.to_json(args.output_file)

    # Only save local dataset, do not push to HuggingFace Hub
    print(f"Dataset saved to local file: {args.output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
