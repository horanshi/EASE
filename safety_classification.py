import json
import argparse
import openai
from typing import List, Dict
import time
import random

# Set OpenAI API
def setup_openai_client(api_key: str):
    """Set up OpenAI client"""
    return openai.OpenAI(api_key=api_key)

def categorize_prompt(client, prompt: str, max_retries: int = 3) -> List[str]:
    """
    Use GPT-4o to classify the prompt, with retry mechanism
    """
    categories = [
        "Harassment / Hate / Discrimination",
        "Sexual / Adult", 
        "Violence / Physical Harm",
        "Self-Harm",
        "Illicit / Criminal Behavior",
        "Misinformation / Disinformation",
        "Privacy / Personal Data",
        "Intellectual Property"
    ]
    
    system_prompt = f"""You are a content classifier. Given a prompt, classify it into one or more of the following categories:

{', '.join(categories)}

Instructions:
1. Return ONLY the category names that apply, separated by commas
2. If multiple categories apply, list all relevant ones
3. If none apply perfectly, choose the closest match
4. Be concise - return only category names
5. You MUST return at least one category - do not return empty responses

Example output: "Violence / Physical Harm, Illicit / Criminal Behavior"
"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify this prompt: {prompt}"}
                ],
                temperature=0,
                max_tokens=100
            )
            
            result = response.choices[0].message.content
            if result is None or result.strip() == "":
                print(f"Attempt {attempt + 1}: GPT returned empty result, retrying...")
                time.sleep(1)  # Short delay before retry
                continue
            
            result = result.strip()
            # Clean and split categories
            categories_found = [cat.strip() for cat in result.split(',') if cat.strip()]
            
            if categories_found:  # Ensure there is a classification result
                return categories_found
            else:
                print(f"Attempt {attempt + 1}: Parsed empty categories, retrying...")
                time.sleep(1)
                continue
                
        except Exception as e:
            print(f"Attempt {attempt + 1}: API call failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait longer on error
                continue
    
    print(f"Failed to obtain valid classification after {max_retries} attempts, skipping this item")
    return None  # Final failure returns None

def load_dataset(file_path: str) -> List[Dict]:
    """Load JSONL dataset"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_categorized_data(data: List[Dict], output_file: str):
    """Save categorized data"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Categorize dataset')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file path')
    parser.add_argument('--output', '-o', default='categorized_data.jsonl', help='Output file name')
    parser.add_argument('--api_key', '-k', required=True, help='OpenAI API key')
    parser.add_argument('--test_count', '-t', type=int, default=None, help='Number of test samples (optional, for testing)')
    parser.add_argument('--delay', '-d', type=float, default=1.0, help='API call interval (seconds)')
    
    args = parser.parse_args()
    
    # Set up OpenAI client
    client = setup_openai_client(args.api_key)
    
    # Load dataset
    print(f"Loading dataset: {args.input}")
    dataset = load_dataset(args.input)
    print(f"Total loaded: {len(dataset)} items")
    
    # If test count specified, randomly sample
    if args.test_count and args.test_count < len(dataset):
        print(f"Randomly sampling {args.test_count} items for testing")
        dataset = random.sample(dataset, args.test_count)
    
    # Separate harmful and benign data
    harmful_data = [item for item in dataset if item['data_type'] == 'Adversarial Harmful']
    #harmful_data = [item for item in dataset]
    benign_data = [item for item in dataset if item['data_type'] == 'Adversarial Benign']
    
    #print(f"Harmful data to classify: {len(harmful_data)} items")
    #print(f"Benign data: {len(benign_data)} items")
    
    # Process data
    categorized_data = []
    
    # Process benign data (directly label as benign)
    for item in benign_data:
        categorized_data.append({
            "question": item['prompt'],
            "category": "benign"
        })
    
    # Process harmful data (requires API classification)
    print("Starting classification of harmful data...")
    successful_count = 0
    failed_count = 0
    
    for i, item in enumerate(harmful_data):
        print(f"Processing item {i+1}/{len(harmful_data)}")
        
        #categories = categorize_prompt(client, item['prompt'])
        categories = categorize_prompt(client, item['prompt'])

        if categories is not None:  # Successfully obtained classification
            category_str = ', '.join(categories)
            
            categorized_data.append({
                "question": item['prompt'],
                "category": category_str
            })
            
            print(f"Classification result: {category_str}")
            successful_count += 1
        else:  # Failed, skip this item
            print(f"Skipping item {i+1}")
            failed_count += 1
        
        # Add delay to avoid API rate limits
        if i < len(harmful_data) - 1:  # No delay for last item
            time.sleep(args.delay)
    
    print(f"\nProcessing statistics:")
    print(f"  Successfully classified: {successful_count} items")
    print(f"  Failed and skipped: {failed_count} items")
    #print(f"  Benign data: {len(benign_data)} items")
    
    # Save results
    print(f"Saving results to: {args.output}")
    save_categorized_data(categorized_data, args.output)
    
    # Statistics
    print("\nClassification summary:")
    category_counts = {}
    for item in categorized_data:
        category = item['category']
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} items")
    
    print(f"\nDone! Total processed: {len(categorized_data)} items")

if __name__ == "__main__":
    main()
