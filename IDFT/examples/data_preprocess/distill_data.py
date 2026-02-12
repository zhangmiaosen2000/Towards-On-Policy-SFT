import json
import pandas as pd
from datasets import Dataset
import argparse
import os


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left) :]
        else:
            return None

    left = "\\boxed{"

    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    else:
        return None


def last_boxed_only_string(string):
    if string is None:
        return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

def extract_solution(solution_str):
    try:
        boxed_string = last_boxed_only_string(solution_str)
        if boxed_string is None:
            return None
        return remove_boxed(boxed_string)
    except Exception:
        return None


def convert_messages_to_parquet(input_file, output_dir, output_name, data_source, start_idx, end_idx, fix_original_json=False):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {input_file}...", flush=True)
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"Loaded {len(json_data)} samples from {input_file}", flush=True)
    
    if end_idx == 0:
        end_idx = len(json_data)
    
    selected_data = json_data[start_idx:end_idx]
    
    processed_samples = []
    incomplete_samples = []
    fixed_samples = []
    fixed_indices = []
    
    for idx, item in enumerate(selected_data):
        messages = item.get('messages', [])
        
        if len(messages) >= 2:
            if messages[0].get('role') == 'user' and messages[1].get('role') == 'user':
                original_messages = [m.copy() for m in messages]
                messages[1]['role'] = 'assistant'
                
                actual_idx = start_idx + idx
                fixed_indices.append(actual_idx)
                
                if fix_original_json:
                    json_data[actual_idx]['messages'][1]['role'] = 'assistant'
                
                fixed_samples.append({
                    'index': actual_idx,
                    'original': original_messages,
                    'fixed': messages
                })
        
        user_message = None
        assistant_message = None
        
        for msg in messages:
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
            elif msg.get('role') == 'assistant':
                assistant_message = msg.get('content', '')
        
        if not user_message or not assistant_message:
            incomplete_samples.append({
                'index': start_idx + idx,
                'item': item,
                'user_message': user_message,
                'assistant_message': assistant_message
            })
            continue
        
        solution = extract_solution(assistant_message)
        
        prompt_list = [
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        sample_length = len(user_message) + len(assistant_message)
        
        processed_samples.append({
            "data_source": data_source,
            "prompt": prompt_list,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": output_name,
                "index": start_idx + idx,
                "answer": assistant_message,
                "question": user_message,
                "sample_length": sample_length,
            },
        })
    
    if fixed_samples:
        print(f"\n{'='*80}", flush=True)
        print(f"Auto-fixed {len(fixed_samples)} samples (user->user changed to user->assistant)", flush=True)
        print(f"Showing first {min(3, len(fixed_samples))} examples for verification:", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        for i, sample in enumerate(fixed_samples[:3]):
            print(f"\n--- Fixed Sample #{i+1} (Index: {sample['index']}) ---", flush=True)
            print(f"BEFORE (original):", flush=True)
            for j, msg in enumerate(sample['original'][:2]):
                print(f"  Message {j}: role='{msg.get('role')}', content_length={len(msg.get('content', ''))}", flush=True)
            print(f"\nAFTER (fixed):", flush=True)
            for j, msg in enumerate(sample['fixed'][:2]):
                print(f"  Message {j}: role='{msg.get('role')}', content_length={len(msg.get('content', ''))}", flush=True)
            print(f"\nFull fixed structure:", flush=True)
            print(json.dumps(sample['fixed'], indent=2, ensure_ascii=False)[:500] + "...", flush=True)
            print(f"{'='*80}\n", flush=True)
    
    if incomplete_samples:
        print(f"\n{'='*80}", flush=True)
        print(f"Found {len(incomplete_samples)} incomplete conversations (after auto-fix attempt)", flush=True)
        print(f"Showing first {min(5, len(incomplete_samples))} examples for debugging:", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        for i, sample in enumerate(incomplete_samples[:5]):
            print(f"\n--- Incomplete Sample #{i+1} (Index: {sample['index']}) ---", flush=True)
            print(f"Full item structure:", flush=True)
            print(json.dumps(sample['item'], indent=2, ensure_ascii=False), flush=True)
            print(f"\nExtracted user_message: {repr(sample['user_message'])}", flush=True)
            print(f"Extracted assistant_message: {repr(sample['assistant_message'])}", flush=True)
            print(f"{'='*80}\n", flush=True)
    
    if not processed_samples:
        print("ERROR: No valid samples to convert!", flush=True)
        return
    
    max_length = max(sample['extra_info']['sample_length'] for sample in processed_samples)
    max_length_idx = max(range(len(processed_samples)), 
                         key=lambda i: processed_samples[i]['extra_info']['sample_length'])
    max_length_sample = processed_samples[max_length_idx]
    
    dataset = Dataset.from_list(processed_samples)
    output_path = os.path.join(output_dir, f'{output_name}.parquet')
    dataset.to_parquet(output_path)
    
    if fix_original_json and fixed_indices:
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        fixed_json_path = os.path.join(output_dir, f'{name_without_ext}_fixed.json')
        
        print(f"\n{'='*80}", flush=True)
        print(f"Saving fixed JSON file...", flush=True)
        with open(fixed_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"  Fixed {len(fixed_indices)} samples in original JSON", flush=True)
        print(f"  Saved to: {fixed_json_path}", flush=True)
        print(f"{'='*80}", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"Conversion Summary:", flush=True)
    print(f"  Total input samples: {len(selected_data)}", flush=True)
    print(f"  Auto-fixed samples: {len(fixed_samples)}", flush=True)
    print(f"  Successfully converted: {len(processed_samples)}", flush=True)
    print(f"  Skipped (incomplete): {len(incomplete_samples)}", flush=True)
    print(f"  Max sample length: {max_length} chars (index: {max_length_sample['extra_info']['index']})", flush=True)
    print(f"    - Question length: {len(max_length_sample['extra_info']['question'])} chars", flush=True)
    print(f"    - Answer length: {len(max_length_sample['extra_info']['answer'])} chars", flush=True)
    print(f"  Output file: {output_path}", flush=True)
    if fix_original_json and fixed_indices:
        print(f"  Fixed JSON file: {fixed_json_path}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"\nExample of successfully converted data:", flush=True)
    print(dataset[0], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_dir', default='data/messages_sft', help='Directory to save the output parquet file.')
    parser.add_argument('--output_name', default='train', help='Name of the output parquet file (e.g., train, test).')
    parser.add_argument('--data_source', default='messages_json', help='Name of the data source.')
    parser.add_argument('--start', type=int, default=0, help='Start index for slicing the dataset.')
    parser.add_argument('--end', type=int, default=0, help='End index for slicing the dataset (0 for full dataset).')
    parser.add_argument('--fix_json', action='store_true', help='Fix the original JSON file by correcting user->user to user->assistant.')
    
    args = parser.parse_args()
    
    convert_messages_to_parquet(
        args.input_file,
        args.output_dir,
        args.output_name,
        args.data_source,
        args.start,
        args.end,
        args.fix_json
    )
