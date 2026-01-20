import json
import argparse
from tqdm import tqdm
from datasets import Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Use Llama Guard to filter AI content for safety")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-Guard-3-8B",
                        help="Llama Guard model name")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSONL file containing prompt-answer pairs")
    parser.add_argument("--output_file", type=str, default="filtered_data.jsonl",
                        help="Output file path for safe content")
    parser.add_argument("--rejected_file", type=str, default="rejected_data.jsonl",
                        help="Output file path for unsafe content")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism")
    return parser.parse_args()


def create_llama_guard_prompt(tokenizer, prompt, answer):
    """Create a prompt in Llama Guard format"""
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]
    
    # Format the conversation using the tokenizer
    formatted_prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    
    return formatted_prompt


def load_conversation_data(input_file, max_samples=None):
    """Load conversation data from a JSONL file"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue

    if max_samples and max_samples < len(data):
        data = data[:max_samples]

    return Dataset.from_list(data)


def main():
    global args
    args = parse_args()

    # Load data
    dataset = load_conversation_data(args.input_file, args.max_samples)
    print(f"Loaded {len(dataset)} conversation samples")

    # Load tokenizer for prompt formatting
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model
    print(f"Loading model: {args.model_name}")
    model = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20
    )

    # Prepare prompts
    print("Creating prompts...")
    prompts = []
    for item in dataset:
        formatted_prompt = create_llama_guard_prompt(tokenizer, item["question"], item["response"])
        prompts.append(formatted_prompt)

    # Test the first prompt
    if len(prompts) > 0:
        print("\n" + "="*50)
        print("Testing first conversation:")
        print(f"Original question: {dataset[0]['question'][:200]}...")
        print(f"Response: {dataset[0]['response'][:200]}...")
        print(f"\nFormatted prompt:\n{prompts[0]}")
        print("="*50 + "\n")

    # Batch generation
    print("Running safety evaluation...")
    outputs = model.generate(prompts, sampling_params)

    # Process results
    safe_data = []
    unsafe_data = []

    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        
        # Print detailed information for the first sample
        if i == 0:
            print(f"First sample Llama Guard response: {response}")
        
        # Simple check: responses starting with "safe" are considered safe
        is_safe = response.lower().startswith("safe")
        
        result = {
            "question": dataset[i]["question"],
            "response": dataset[i]["response"]
        }
        
        if is_safe:
            safe_data.append(result)
        else:
            unsafe_data.append(result)

    # Save results
    safe_dataset = Dataset.from_list(safe_data)
    unsafe_dataset = Dataset.from_list(unsafe_data)

    print(f"Saving {len(safe_dataset)} safe conversations to {args.output_file}")
    safe_dataset.to_json(args.output_file)

    print(f"Saving {len(unsafe_dataset)} unsafe conversations to {args.rejected_file}")
    unsafe_dataset.to_json(args.rejected_file)

    print(f"Safety evaluation complete! Safe rate: {len(safe_data) / len(dataset) * 100:.2f}%")


if __name__ == "__main__":
    main()
