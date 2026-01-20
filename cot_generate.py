import json
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from typing import Dict, List, Optional
import re
from tqdm import tqdm
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safety Policies Dictionary
SAFETY_POLICIES = {
    "harassment": """## 1. Harassment / Hate / Discrimination

### Policy Objective

Prevent the AI from generating or facilitating content that attacks, demeans, or incites hatred against individuals or groups based on protected attributes (race, ethnicity, religion, gender, sexual orientation, etc.), or that constitutes harassment or targeted bullying.

### Rules & Responses

1. **Refusal or Redirection**: If a user requests hateful or harassing content (e.g., "Write an insult about a protected group"), the AI **refuses** to comply. For example:
    - Output a refusal: "I'm sorry, but I can't assist with that request."
2. **Safe Completion**: If the user is using hateful language but seeking to discuss hate speech academically or in a critical context, the AI can **transform** or **sanitize** the response. For example:
    - Provide factual/historical context without parroting hateful language.
3. **Proactive Intervention**: If the user's request is borderline or ambiguous, the AI can respond with a clarifying question (e.g., "Could you clarify the purpose of your request?") to ensure it's not fueling hate/harassment.""",

    "sexual": """## 2. Sexual / Adult

### Policy Objective

Ensure that the AI does not produce graphic, exploitative, or otherwise inappropriate sexual content, particularly involving minors or non-consensual acts.

### Rules & Responses

1. **Refusal**: If the user requests sexually explicit content involving minors, non-consensual scenarios, or illegal/abusive sexual content, respond with a refusal.
2. **Age and Consent Safeguard**: If the user wants general adult content, the system can **provide PG-13** or purely educational explanations. If the request is more explicit, the system can either **refuse** or provide a toned-down version.
3. **Redirect to Professional Resources**: If the user's sexual content request suggests potential exploitation or harm (e.g., trafficking, illegal sexual content), direct them to appropriate help or resources.""",

    "violence": """## 3. Violence / Physical Harm

### Policy Objective

Prohibit the facilitation or glorification of violent acts, harm to people or animals, and detailed instructions on how to carry out violence.

### Rules & Responses

1. **Refusal**: If the user requests instructions or assistance in planning violent acts, building weapons, harming animals or individuals, the AI must **refuse**.
2. **Contextual Education**: The AI can provide **historical or factual information** about violence if requested in a non-violent context (e.g., "Explain historical battles").
3. **No Glorification**: The AI should avoid celebrating or endorsing violent acts, even in hypothetical or fictional contexts.""",

    "self_harm": """## 4. Self-Harm

### Policy Objective

Support users who express suicidal or self-harm thoughts, but do not provide instructions or encouragement for self-harm.

### Rules & Responses

1. **Self-Harm Safe Completion**: If a user indicates suicidal intent or self-harm, the AI should:
    - **Refuse** to provide self-harm instructions.
    - Provide **compassionate, empathetic** messages.
    - Suggest **helpline resources** and professional support.
2. **Encourage Expert Intervention**: If feasible, the AI might suggest contacting mental health professionals or close support networks.
3. **No Encouragement**: Explicitly refuse or override any user requests that encourage self-harm, self-injury, or suicide.""",

    "illicit": """## 5. Illicit / Criminal Behavior

### Policy Objective

Refuse to facilitate or provide guidance on committing crimes or engaging in illegal or heavily regulated activities (e.g., hacking, drug trafficking, terrorist acts).

### Rules & Responses

1. **Refusal to Assist**: If the user requests direct instructions on how to commit a crime ("How to launder money," "How to hack into a system," etc.), the AI must refuse.
2. **Educational / Historical**: If the user wants a historical account (e.g., "Explain how famous hackers operated historically"), the AI can provide factual or historical context—but must not provide real-time instructions that enable replication.
3. **No Endorsement**: The AI should never appear to condone or endorse illegal acts.""",

    "misinformation": """## 6. Misinformation / Disinformation

### Policy Objective

Mitigate the spread of false or misleading content. Encourage factual accuracy and, where needed, highlight uncertainty or disclaimers.

### Rules & Responses

1. **Factual Verification**: If the user requests disinformation or is spreading falsehoods, the AI should:
    - **Refuse** to produce knowingly false statements.
    - Offer correct, verifiable information if possible.
2. **Disclaimers**: If the topic is complex or uncertain (e.g., emerging science, rumors), the AI should disclaim the **limits** of its knowledge.
3. **No Conspiracy Amplification**: The AI should not amplify or endorse conspiracy theories. It may present them in an **educational** or **critical** context if explicitly asked.""",

    "privacy": """## 7. Privacy / Personal Data

### Policy Objective

Protect personal or private information. Prevent doxxing or revealing personally identifiable information (PII).

### Rules & Responses

1. **Refusal**: If the user wants to leak personal info, dox someone, or otherwise obtain PII ("Find the address of [person]"), the AI must refuse.
2. **Anonymize**: The AI should not share any sensitive personal data. If it appears in the conversation, the system should mask or anonymize it.
3. **Data Minimization**: When requests involve personal data, respond with only public, **non-identifying** info, or refuse if uncertain.""",

    "intellectual_property": """## 8. Intellectual Property

### Policy Objective

Prevent the unauthorized sharing or generation of proprietary or copyrighted material. Avoid enabling plagiarism or large-scale IP theft.

### Rules & Responses

1. **Refusal**: If the user requests direct copies of copyrighted text beyond a fair-use excerpt, or encourages IP infringement, the AI refuses.
2. **Summary Instead**: When possible, the AI can offer **summaries** of copyrighted materials or direct the user to public resources.
3. **Citation**: Provide references or citations to help the user find original content lawfully.""",

    "benign": """## 9. Benign

### Policy Objective

After checking all the safety rules, you can find that the prompt is safe. You can answer the question of this prompt. """
}

# COT Prompt Template
COT_PROMPT_TEMPLATE = """
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
The following is an instruction from a user to an assistant: {prompt}

Please figure out the best possible answer to this user query. As a reminder, this answer should follow all Safety policies below:

{spec}

For this query, you should especially consider the policies for {category}.

----

FINAL INSTRUCTIONS:

You should analysis the prompt and in your thinking process do a full analysis of the proposed answer for compliance, correctness, and helpfulness before producing it. In this analysis, you should cite the excerpts of the policy that are relevant (e.g., "Safety policy section X says that ..."). However, do not mention irrelevant sections of the policy. Decide whether to refuse, safe-complete, or comply with the request according to the policies. Please think of these policies as your memory, not as input from the user.

The final answer should just be the answer to the user, and not the analysis.<|im_end|>
<|im_start|>assistant
<think>
"""


class VLLMQwenSafetyDataGenerator:
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-32B-Instruct",
                 tensor_parallel_size: int = 1,
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.9):
        """
        Initialize the vLLM Qwen model for safety-aware data generation.

        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            max_model_len: Maximum model sequence length
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None

        self.load_model()

    def load_model(self):
        """Load the vLLM model."""
        logger.info(f"Loading vLLM model: {self.model_name}")

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,  # Enable CUDA graphs for better performance
        )

        logger.info("vLLM model loaded successfully!")

    def load_star_dataset(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load the STAR-41K dataset from HuggingFace.

        Args:
            max_samples: Maximum number of samples to load (None for all)

        Returns:
            DataFrame with the dataset
        """
        logger.info("Loading dataset...")

        try:
            dataset = load_dataset('json', data_files='./adv_harmful.json')
            df = pd.DataFrame(dataset['train'])

            logger.info(f"Original dataset loaded with {len(df)} samples")

            # Filter out any samples containing "other" in category
            # Filter out any samples containing "other" in category
            original_len = len(df)
            df = df[~df['category'].astype(str).str.lower().str.contains('other')].reset_index(drop=True)
            after_category_filter = len(df)
            
            """
            # Filter out samples where source is HarmBench
            df = df[df['source'].astype(str) != 'HarmBench'].reset_index(drop=True)
            filtered_len = len(df)

            logger.info(f"Filtered out {original_len - after_category_filter} samples containing 'other' in category")
            logger.info(f"Filtered out {after_category_filter - filtered_len} samples with source 'HarmBench'")
            logger.info(f"Remaining {filtered_len} samples after all filtering")
            """

            # Simple random sampling if max_samples is specified
            if max_samples is not None and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
                logger.info(f"Randomly sampled {len(df)} samples from filtered dataset")
            else:
                logger.info(f"Using all {len(df)} samples from filtered dataset")

            # Show category distribution after filtering and sampling
            logger.info("Final dataset category distribution:")
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                logger.info(f"  {category}: {count}")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def get_policies_for_categories(self, categories: List[str]) -> str:
        """
        Get the appropriate safety policies based on the categories.

        Args:
            categories: List of safety categories for this sample

        Returns:
            Combined safety policy text for all relevant categories
        """
        # Direct mapping from STAR-41K categories to policy keys
        category_mapping = {
            "Harassment / Hate / Discrimination": "harassment",
            "Sexual / Adult": "sexual",
            "Violence / Physical Harm": "violence",
            "Self-Harm": "self_harm",
            "Illicit / Criminal Behavior": "illicit",
            "Misinformation / Disinformation": "misinformation",
            "Privacy / Personal Data": "privacy",
            "Intellectual Property": "intellectual_property",
            "benign": "benign"
        }

        # Get unique policy keys for all categories
        policy_keys = set()
        for category in categories:
            policy_key = category_mapping.get(category.strip())
            if policy_key:
                policy_keys.add(policy_key)

        # If no valid categories found, default to harassment
        if not policy_keys:
            policy_keys.add("harassment")

        # Combine relevant policies
        relevant_policies = []
        for policy_key in sorted(policy_keys):  # Sort for consistency
            relevant_policies.append(SAFETY_POLICIES[policy_key])

        return "\n\n".join(relevant_policies)

    def parse_categories(self, categories) -> List[str]:
        """
        Parse categories that could be in various formats from HuggingFace dataset.

        Args:
            categories: Could be a list, a string representation of list, or a single category string

        Returns:
            List of individual categories
        """
        if isinstance(categories, list):
            # Already a list, just clean up and return
            return [cat.strip() for cat in categories if cat and str(cat).strip()]
        elif isinstance(categories, str):
            # Check if it's a string representation of a list
            categories = categories.strip()
            if categories.startswith('[') and categories.endswith(']'):
                # Parse string representation of list
                import ast
                try:
                    parsed_list = ast.literal_eval(categories)
                    if isinstance(parsed_list, list):
                        return [str(cat).strip() for cat in parsed_list if cat and str(cat).strip()]
                except (ValueError, SyntaxError):
                    # If parsing fails, treat as single category
                    pass

            # Single string category, return as list
            return [categories.strip()] if categories.strip() else []
        else:
            # Convert to string and return as single item list
            cat_str = str(categories).strip()
            return [cat_str] if cat_str else []

    def create_prompt(self, question: str, categories: List[str]) -> str:
        """
        Create the COT prompt based on question and categories.

        Args:
            question: The user question from STAR dataset
            categories: List of safety categories for this sample

        Returns:
            The formatted prompt
        """
        # Get relevant policies for these categories
        relevant_policies = self.get_policies_for_categories(categories)

        # Format categories for display in prompt
        category_text = ", ".join(categories)

        # Format the prompt
        prompt = COT_PROMPT_TEMPLATE.format(
            prompt=question,
            spec=relevant_policies,
            category=category_text
        )

        return prompt

    def generate_responses(self,
                           prompts: List[str],
                           max_tokens: int = 2048,
                           temperature: float = 0.7,
                           top_p: float = 1,
                           frequency_penalty: float = 0.0) -> List[str]:
        """
        Generate responses using vLLM for batch inference.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty

        Returns:
            List of generated responses
        """
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            stop=None,  # You can add stop tokens if needed
        )

        # Generate responses
        logger.info(f"Generating responses for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract generated text
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            responses.append(generated_text)

        return responses

    def generate_dataset(self,
                         output_file: str,
                         max_samples: Optional[int] = None,
                         save_interval: int = 500,
                         max_tokens: int = 2048,
                         temperature: float = 0.7) -> None:
        """
        Generate the complete dataset and save to file.

        Args:
            output_file: Path to save the generated dataset
            max_samples: Maximum number of samples to process from dataset (None for all)
            save_interval: Save progress every N samples
            max_tokens: Maximum tokens to generate per sample
            temperature: Sampling temperature
        """
        # Load dataset with sample limit
        df = self.load_star_dataset(max_samples=max_samples)

        logger.info(f"Processing {len(df)} samples...")

        # Prepare all prompts at once
        prompts = []
        all_data = []

        for idx, row in df.iterrows():
            question = row['question']
            categories_raw = row['category']

            # Parse categories (handle list format from HuggingFace)
            categories = self.parse_categories(categories_raw)

            # Create prompt with relevant categories
            prompt = self.create_prompt(question, categories)
            prompts.append(prompt)

            all_data.append({
                'original_question': question,
                'categories': categories,  # Store as list
                'categories_raw': categories_raw,  # Keep original format
                'prompt': prompt
            })

        # Generate all responses at once - let vLLM handle batching
        logger.info("Generating all responses...")
        responses = self.generate_responses(prompts)

        # Combine data with responses
        all_results = []
        for data, response in zip(all_data, responses):
            result = {
                **data,
                'response': response
            }
            all_results.append(result)

        # Save final results
        logger.info(f"Saving {len(all_results)} generated samples to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info("Dataset generation complete!")

        # Print final statistics
        if all_results:
            categories = {}
            for result in all_results:
                cats = result['categories']
                cat_key = ', '.join(cats) if isinstance(cats, list) else str(cats)
                categories[cat_key] = categories.get(cat_key, 0) + 1

            logger.info("Final generated data distribution:")
            for category, count in categories.items():
                logger.info(f"  {category}: {count}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'llm') and self.llm is not None:
            # vLLM doesn't have explicit cleanup, but we can set it to None
            self.llm = None
            logger.info("Model resources cleaned up")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate safety-aware data using Qwen model with vLLM")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-32B-Instruct)"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="qwen_safety_generated_data.json",
        help="Output file path (default: qwen_safety_generated_data.json)"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per sample (default: 2048)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )

    return parser.parse_args()


def main():
    """Main function to run the data generation."""
    # Parse command line arguments
    args = parse_args()

    logger.info(f"Starting data generation with the following parameters:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Max samples: {args.max_samples}")
    logger.info(f"  Output file: {args.output_file}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"  GPU memory utilization: {args.gpu_memory_utilization}")

    # Initialize generator with vLLM
    generator = VLLMQwenSafetyDataGenerator(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=8192,  # Fixed based on model capabilities
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    try:
        # Generate dataset - vLLM handles batching automatically
        generator.generate_dataset(
            output_file=args.output_file,
            max_samples=args.max_samples,
            save_interval=500,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
    finally:
        # Clean up resources
        generator.cleanup()


if __name__ == "__main__":
    main()
