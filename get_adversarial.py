import json
import random
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description='get adversarial harmful data from WildJailbreak')
    parser.add_argument('--output', '-o', default='adversarial_data.jsonl')
    parser.add_argument('--harmful', type=int, default=1000)
    parser.add_argument('--benign', type=int, default=1000)
    parser.add_argument('--seed', '-s', type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("load dataset...")
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    train_data = dataset['train']
  
    print(f"dataset size: {len(train_data)}")
    print(f"fist data: {train_data[0]}")
    
    harmful_data = [item for item in train_data if item['data_type'] == 'adversarial_harmful']
    benign_data = [item for item in train_data if item['data_type'] == 'adversarial_benign']
    
    print(f"adversarial harmful data: {len(harmful_data)}")
    print(f"adversarial benign data: {len(benign_data)}")
    
    empty_adv_harmful = sum(1 for item in harmful_data if not item['adversarial'])
    empty_adv_benign = sum(1 for item in benign_data if not item['adversarial'])
    
    sampled_harmful = random.sample(harmful_data, min(args.harmful, len(harmful_data)))
    sampled_benign = random.sample(benign_data, min(args.benign, len(benign_data)))
    
    output_data = []
    
    for item in sampled_harmful:
        output_data.append({
            "prompt": item['adversarial'],
            "data_type": "Adversarial Harmful"
        })
    
    for item in sampled_benign:
        output_data.append({
            "prompt": item['adversarial'],
            "data_type": "Adversarial Benign"
        })
    
    random.shuffle(output_data)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
   
    print(f"saved: {args.output}")
    print(f"total: {len(output_data)} samples")

if __name__ == "__main__":
    main()
