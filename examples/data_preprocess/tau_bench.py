import argparse
import os
import sys
import json
from typing import List, Dict, Any
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = script_dir
for _ in range(5):
    parent_dir = os.path.dirname(current_dir)
    tau_bench_module = os.path.join(parent_dir, "tau_bench")
    if os.path.isdir(tau_bench_module) and os.path.exists(os.path.join(tau_bench_module, "__init__.py")):
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        break
    current_dir = parent_dir
else:
    raise RuntimeError(f"Cannot find tau_bench module. Searched from {script_dir}")


def convert_tools_to_openai_format(tools) -> List[Dict[str, Any]]:
    tool_schemas = []
    for tool in tools:
        tool_info = tool.get_info()
        tool_schemas.append(tool_info)
    return tool_schemas


def action_to_assistant_message(action) -> str:
    tool_call = {
        "name": action.name,
        "arguments": action.kwargs
    }
    message = json.dumps(tool_call, ensure_ascii=False, indent=2)
    return message


def convert_task_to_multiturn_format(
    task, 
    wiki_content: str, 
    rules_content: List[str], 
    tools
) -> Dict[str, Any]:
    messages = []
    
    rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(rules_content)])
    system_prompt = f"""You are a customer service representative for an online retail company.

# Company Knowledge Base
{wiki_content}

# Important Rules
{rules_text}

Please help the customer by calling the appropriate tools to complete their request."""
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    messages.append({
        "role": "user",
        "content": task.instruction
    })
    
    for i, action in enumerate(task.actions):
        action_json = json.dumps({
            "name": action.name,
            "arguments": action.kwargs
        }, ensure_ascii=False, indent=2)
        
        tool_call_content = f"I will call the {action.name} function with the following parameters:\n```json\n{action_json}\n```"
        
        messages.append({
            "role": "assistant",
            "content": tool_call_content
        })
    
    return {
        "messages": messages,
        "tools": convert_tools_to_openai_format(tools),
        "user_id": task.user_id,
        "annotator": getattr(task, "annotator", "unknown")
    }


def convert_simple_format(task, tools=None) -> Dict[str, Any]:
    response_parts = []
    for action in task.actions:
        action_text = f"<tool_call>\n{{\n  \"name\": \"{action.name}\",\n  \"arguments\": {json.dumps(action.kwargs, ensure_ascii=False)}\n}}\n</tool_call>"
        response_parts.append(action_text)
    
    response = "\n\n".join(response_parts)
    
    return {
        "messages": [
            {"role": "user", "content": task.instruction},
            {"role": "assistant", "content": response}
        ],
        "user_id": task.user_id,
        "annotator": getattr(task, "annotator", "unknown")
    }


def convert_structured_format(task, tools=None) -> Dict[str, Any]:
    messages = []
    
    messages.append({
        "role": "user",
        "content": task.instruction
    })
    
    for i, action in enumerate(task.actions):
        action_json = json.dumps({
            "name": action.name,
            "arguments": action.kwargs
        }, ensure_ascii=False, indent=2)
        
        assistant_msg = f"I will call the {action.name} function:\n```json\n{action_json}\n```"
        
        messages.append({
            "role": "assistant",
            "content": assistant_msg
        })
        
        if i < len(task.actions) - 1:
            messages.append({
                "role": "user",
                "content": "Please continue."
            })
    
    return {
        "messages": messages,
        "user_id": task.user_id,
        "annotator": getattr(task, "annotator", "unknown")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert tau-bench training data to verl SFT format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/tau_bench_sft",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["multiturn", "simple", "structured"],
        default="simple",
        help="Output format"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["retail", "airline"],
        default="retail",
        help="Environment to convert"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "dev"],
        default="train",
        help="Data split to convert"
    )
    
    args = parser.parse_args()
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.env == "retail":
        if args.split == "train":
            from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
        elif args.split == "test":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
        elif args.split == "dev":
            from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks
        
        from tau_bench.envs.retail.tools import ALL_TOOLS as tools
        from tau_bench.envs.retail.wiki import WIKI
        from tau_bench.envs.retail.rules import RULES
    else:
        if args.split == "train":
            from tau_bench.envs.airline.tasks import tasks
        elif args.split == "test":
            from tau_bench.envs.airline.tasks_test import TASKS_TEST as tasks
        else:
            raise ValueError(f"Airline environment doesn't have {args.split} split")
        
        from tau_bench.envs.airline.tools import ALL_TOOLS as tools
        from tau_bench.envs.airline.wiki import WIKI
        from tau_bench.envs.airline.rules import RULES
    
    print(f"Converting {len(tasks)} tasks from {args.env} environment ({args.split} split)...")
    print(f"Format: {args.format}")
    
    converted_data = []
    for i, task in enumerate(tasks):
        try:
            if args.format == "multiturn":
                data = convert_task_to_multiturn_format(task, WIKI, RULES, tools)
            elif args.format == "structured":
                data = convert_structured_format(task, tools)
            else:
                data = convert_simple_format(task, tools)
            
            data["idx"] = i
            converted_data.append(data)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(tasks)} tasks...")
                
        except Exception as e:
            print(f"Warning: Failed to convert task {i}: {e}")
            continue
    
    df = pd.DataFrame(converted_data)
    output_file = os.path.join(output_dir, f"{args.env}_{args.split}_{args.format}.parquet")
    df.to_parquet(output_file)
    
    print(f"\nSuccessfully converted {len(converted_data)} samples")
    print(f"Saved to: {output_file}")
    print(f"\nDataset Statistics:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Columns: {df.columns.tolist()}")
    
    if len(df) > 0:
        sample = df.iloc[0].to_dict()
        print(f"\nSample data (first example):")
        print(f"   User ID: {sample.get('user_id', 'N/A')}")
        print(f"   Annotator: {sample.get('annotator', 'N/A')}")
        print(f"   Messages: {len(sample['messages'])} turns")
        print(f"\n   First message:")
        print(f"   {json.dumps(sample['messages'][0], ensure_ascii=False, indent=2)}")
        if len(sample['messages']) > 1:
            print(f"\n   Second message:")
            print(f"   {json.dumps(sample['messages'][1], ensure_ascii=False, indent=2)[:500]}...")
    else:
        print("\nWarning: No samples were converted successfully!")


if __name__ == "__main__":
    main()
