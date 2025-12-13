import json
import re
import os


def load_json(filename):
    if not os.path.exists(filename):
        print(f"Input file '{filename}' not found.")
        return []
    
    with open(filename, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            print(f"Loaded {len(data)} items from '{filename}'")
            return data
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from '{filename}'")
            return []


def parse_llm_output(llm_output):
    if not llm_output:
        return []
    
    pattern = r"\[\s*([^\]]+?)\s*\]\s*<\s*([^>]+?)\s*>"
    matches = re.findall(pattern, llm_output)
    
    return matches


def generalize_action_sequence(llm_output, task_info, semantic_lists):
    raw_actions = parse_llm_output(llm_output) 
    
    template = []
    extracted_stats = {"tool": None, "location": None}
    
    input_task = task_info['task']
    target_obj = task_info['target']
    
    for action, arg in raw_actions:
        arg_class = arg.split('_')[0] 
        
        new_arg = arg
        
        if arg == target_obj:
            new_arg = "<target>"
            
        elif arg_class in semantic_lists['RECEPTACLE']:
            new_arg = "<location>"
            extracted_stats["location"] = arg_class
            
        elif arg_class in semantic_lists['TOOL']:
            new_arg = "<tool>"
            extracted_stats["tool"] = arg_class
            
        if action == input_task:
            action = "task"

        template.append(f"[{action}] {new_arg}")
        
    return template, extracted_stats
    

def save_memory_json(task_info, template, current_stats):
    target_class = task_info['target'].split('_')[0]
    
    constraints = {
        "target": target_class
    }
    
    template_str = str(template)
    if "<location>" in template_str:
        constraints["location"] = "RECEPTACLE" 
    if "<tool>" in template_str:
        constraints["tool"] = "TOOL"

    stats_data = {
        "location_frequency": {},
        "tool_frequency": {}
    }
    
    if current_stats['location']:
        stats_data['location_frequency'][current_stats['location']] = 1
        
    if current_stats['tool']:
        stats_data['tool_frequency'][current_stats['tool']] = 1

    memory_entry = {
        "task_id": task_info['task'],
        "target_class": target_class,
        "template": template,
        "constraints": constraints,
        "stats": stats_data
    }
    
    return memory_entry


def load_memory_db(filename="memory.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def update_memory_db(memory_db, new_entry):
    task_id = new_entry['task_id']
    target_class = new_entry['target_class']
    
    if task_id not in memory_db:
        memory_db[task_id] = {}

    if target_class in memory_db[task_id]:
        existing_data = memory_db[task_id][target_class]
        
        new_locs = new_entry['stats']['location_frequency']
        for loc, count in new_locs.items():
            prev_count = existing_data['stats']['location_frequency'].get(loc, 0)
            existing_data['stats']['location_frequency'][loc] = prev_count + count
            
        new_tools = new_entry['stats']['tool_frequency']
        for tool, count in new_tools.items():
            prev_count = existing_data['stats']['tool_frequency'].get(tool, 0)
            existing_data['stats']['tool_frequency'][tool] = prev_count + count
            
    else:
        memory_db[task_id][target_class] = new_entry
        
    return memory_db

def save_memory_file(memory_db, filename="memory.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(memory_db, f, indent=4)
    print(f"Memory database saved to '{filename}'")
    
    
if __name__ == "__main__":
    
    semantic_lists = {
        "RECEPTACLE": [
            "table", "countertop", "sink", "cabinet", "sofa", "bed", 
            "bookshelf", "desk", "wardrobe", "toilet", "shelf", "garbage-bin"
        ], 
        "TOOL": [
            "rag", "wet-wipe", "tissue", "paper-napkin", "paper-wipe",
            "mop", "vacuum", "towel", "sponge"
        ]
    }

    INPUT_FILE = 'llm_action_seq.json'
    MEMORY_FILE = 'memory.json'

    input_data = load_json(INPUT_FILE)
    memory_db = load_memory_db(MEMORY_FILE)
    
    if not input_data:
        print("No input data to process. Exiting.")
        exit()

    print(f"--- Start Processing... ---")

    processed_count = 0
    for data in input_data:
        task_info = data.get('task_info')
        llm_output = data.get('llm_output')
        
        if not task_info or not llm_output:
            continue
            
        template, stats = generalize_action_sequence(llm_output, task_info, semantic_lists)
        entry = save_memory_json(task_info, template, stats)
        update_memory_db(memory_db, entry)
        
        processed_count += 1

    print(f"--- Processed {processed_count} items ---")
    save_memory_file(memory_db, MEMORY_FILE)