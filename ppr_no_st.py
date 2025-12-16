import networkx as nx
import re
import time
import math
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util


LLM_KNOWLEDGE_BASE = {
    "Tableware": ["spoon", "fork", "plate", "cup"],
    "Food": ["chips", "bread", "liquid", "fluid"],
    "Cleaning_Tool": ["rag", "wet-wipe", "sponge", "paper-towel", "paper-wipe", "mop", "vacuum"],
    "Stationery": ["paper", "book", "newspaper"],
    "Clothing": ["shoes", "pants", "stovepipe-jeans"],
    "Furniture": ["table", "chair", "sofa", "cabinet", "cupboard", "bed", "wardrobe", "desk", "shelf", "console"],
    "Appliance": ["oven", "stove", "range", "desk-lamp"],
    "Waste": ["debris", "crumbs", "dust", "particle", "mud", "sand"],
    "Container": ["trash-can", "garbage-bin", "trash-bin"]
}

origin_sg = """Environment: In <livingroom_0>, liquid_1 on table_1, chair_0 on floor, chips_1 on sofa_0, plate_2 on table_1, paper_0 on table_1, blanket_1 on sofa_0, bread_0 on table_1, cabinet_0 on floor, cabinet_2 on floor, chips_0 next to garbage-bin_0, chips_0 on floor, cup_0 on coffee-table_1, cupboard_0 on floor, debris_0 next to garbage-bin_0, debris_0 on floor, debris_3 next to paper-wipe_0, debris_3 on table_1, desk-lamp_1 on table_1, garbage-bin_0 on floor, liquid_3 next to paper-towel_1, liquid_3 on sofa_0, paper-towel_1 on floor, paper-wipe_0 next to cabinet_0, paper-wipe_0 on table_1, sand_0 next to garbage-bin_0, sand_0 on floor, shoes_0 on floor, sofa_0 on floor, table_1 on floor, vacuum_0 on floor. In <library_0>, ball_0 next to console_2, ball_0 on floor, cabinet_1 on floor, console_2 on floor, crumbs_0 next to trash-bin_1, crumbs_0 on floor, debris_1 next to wet-wipe_4, debris_1 on console_2, desk-lamp_0 on console_2, newspaper_0 next to console_2, newspaper_0 next to trash-bin_1, newspaper_0 on floor, shelf_0 on floor, trash-bin_1 on floor, wet-wipe_4 on console_2. In <kitchen_0>, book_2 next to liquid_0, book_2 next to trash-can_0, book_2 on floor, cabinet_3 on floor, debris_2 on floor, dust_2 next to trash-can_0, dust_2 on floor, fluid_4 next to rag_0, fluid_4 on countertop_0, fork_0 next to sink_0, fork_0 on countertop_0, liquid_0 next to mop_0, liquid_0 next to paper-napkin_2, liquid_0 on floor, mop_0 on coutertop_0, oven_0 on floor, paper-napkin_2 on floor, particle_1 next to trash-can_0, particle_1 on floor, plate_0 next to sink_0, plate_0 on cabinet_3, plate_1 next to sink_0, plate_1 on countertop_0, pot_0 on floor, rag_0 next to countertop_0, rag_0 next to trash-can_0, rag_0 on floor, rag_1 next to debris_2, rag_1 next to trash-can_0, rag_1 on floor, range_0 on countertop_0, sink_0 on floor, spoon_0 next to sink_0, spoon_0 on countertop_0, stove_0 on floor, trash-can_0 on floor. In <bedroom_0>, bed_0 on floor, blanket_0 on bed_0, closet_1 on floor, ledge_1 on floor, liquid_2 next to tissue_3, liquid_2 on floor, pants_0 next to wardrobe_1, pants_0 on floor, pillow_0 next to bed_0, pillow_0 on floor, stovepipe-jeans_1 next to wardrobe_1, stovepipe-jeans_1 on floor, tissue_3 on floor, wardrobe_1 on floor. In <bathroom_0>, mud_1 next to paper-towel_0, mud_1 on toilet_0, paper-towel_0 next to toilet_0, paper-towel_0 on floor, sink_1 on floor, tissue_0 on countertop_3, toilet_0 on floor, towel_0 on floor, countertop_3 next to sink_1."""
# origin_sg = """Environment: In <livingroom_0>, liquid_1 on table_1, chair_0 on floor, chips_1 on sofa_0, plate_2 on table_1, paper_0 on table_1, blanket_1 on sofa_0, bread_0 on table_1, cabinet_0 on floor, cabinet_2 on floor, chips_0 next to garbage-bin_0, chips_0 on floor, cup_0 on coffee-table_1, cupboard_0 on floor, debris_0 next to garbage-bin_0, debris_0 on floor, debris_3 next to paper-wipe_0, debris_3 on table_1, desk-lamp_1 on table_1, garbage-bin_0 on floor, liquid_3 next to paper-towel_1, liquid_3 on sofa_0, paper-towel_1 on floor, paper-wipe_0 next to cabinet_0, paper-wipe_0 on table_1, sand_0 next to garbage-bin_0, sand_0 on floor, shoes_0 on floor, sofa_0 on floor, table_1 on floor, vacuum_0 on floor. In <library_0>, ball_0 next to console_2, ball_0 on floor, cabinet_1 on floor, console_2 on floor, crumbs_0 next to trash-bin_1, crumbs_0 on floor, debris_1 next to wet-wipe_4, debris_1 on console_2, desk-lamp_0 on console_2, newspaper_0 next to console_2, newspaper_0 next to trash-bin_1, newspaper_0 on floor, shelf_0 on floor, trash-bin_1 on floor, wet-wipe_4 on console_2. In <kitchen_0>, book_2 next to liquid_0, book_2 next to trash-can_0, book_2 on floor, cabinet_3 on floor, debris_2 on floor, dust_2 next to trash-can_0, dust_2 on floor, fluid_4 next to rag_0, fluid_4 on countertop_0, fork_0 next to sink_0, fork_0 on countertop_0, liquid_0 next to mop_0, liquid_0 next to paper-napkin_2, liquid_0 on floor, mop_0 on coutertop_0, oven_0 on floor, paper-napkin_2 on floor, particle_1 next to trash-can_0, particle_1 on floor, plate_0 next to sink_0, plate_0 on cabinet_3, plate_1 next to sink_0, plate_1 on countertop_0, pot_0 on floor, rag_0 next to countertop_0, rag_0 next to trash-can_0, rag_0 on floor, rag_1 next to debris_2, rag_1 next to trash-can_0, rag_1 on floor, range_0 on countertop_0, sink_0 on floor, spoon_0 on countertop_0, stove_0 on floor, trash-can_0 on floor. In <bedroom_0>, bed_0 on floor, blanket_0 on bed_0, closet_1 on floor, ledge_1 on floor, liquid_2 next to tissue_3, liquid_2 on floor, pants_0 next to wardrobe_1, pants_0 on floor, pillow_0 next to bed_0, pillow_0 on floor, stovepipe-jeans_1 next to wardrobe_1, stovepipe-jeans_1 on floor, tissue_3 on floor, wardrobe_1 on floor. In <bathroom_0>, mud_1 next to paper-towel_0, mud_1 on toilet_0, paper-towel_0 next to toilet_0, paper-towel_0 on floor, sink_1 on floor, tissue_0 on countertop_3, toilet_0 on floor, towel_0 on floor, countertop_3 next to sink_1."""

def build_scene_graph(text, knowledge_base, category_edge=False):
    G = nx.Graph()
    home_node = 'Home'
    G.add_node(home_node, type='home')
    
    room_split = re.split(r'In <([^>]+)>', text)
    
    for i in range(1, len(room_split), 2):
        room_name = room_split[i]
        content = room_split[i+1]
        
        G.add_node(room_name, type='room')
        G.add_edge(home_node, room_name, weight=2.0, relation='contains')
        
        items = [x.strip() for x in content.split(',') if x.strip()]
        
        for item in items:
            if "floor" in item: item = item.replace("floor", f"{room_name}_floor")
            match = re.match(r'(.+?)\s+(on|next to)\s+(.+)', item)
            
            if match:
                obj1, rel, obj2 = match.groups()
                G.add_node(obj1, type='object'); G.add_node(obj2, type='object')
                
                # 기본 가중치 설정
                w = 1.0 if rel == 'on' else 0.5
                
                G.add_edge(obj1, obj2, weight=w, relation=rel)
                G.add_edge(room_name, obj1, weight=1.0, relation='in')
                G.add_edge(room_name, obj2, weight=1.0, relation='in')
                
                if category_edge:
                    add_category_edge(G, obj1, knowledge_base)
                    add_category_edge(G, obj2, knowledge_base)
            else:
                obj1 = item.replace('.', '')
                if obj1:
                    G.add_node(obj1, type='object')
                    G.add_edge(room_name, obj1, weight=1.0, relation='in')
                    add_category_edge(G, obj1, knowledge_base)
    return G

def add_category_edge(G, obj_name, knowledge_base):
    obj_class = obj_name.split('_')[0]
    for category, items in knowledge_base.items():
        if obj_class in items:
            cat_node = f"{category}"
            if not G.has_node(cat_node): G.add_node(cat_node, type='category')
            if not G.has_edge(obj_name, cat_node):
                G.add_edge(obj_name, cat_node, weight=1.0, relation='is_a')
            break

def get_candidates_text_matching(graph, target_list):
    scene_objects = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'object']
    candidates = []
    for obj in scene_objects:
        obj_class = obj.split('_')[0]
        if obj_class in target_list: candidates.append(obj)
    return candidates

def ppr_location(G, target_object, candidate_list, exclude_current=True, single=False):
    
    if single:
        seed_nodes = {target_object: 1.0}
    else:
        seed_nodes = {target_object: 1.0}    
        target_class = target_object.split('_')[0]
        
        target_category_items = []
        for cat, items in LLM_KNOWLEDGE_BASE.items():
            if target_class in items:
                target_category_items = items
                print(f">> Found Category Group: {cat} -> {items}") 
                break
                
        if target_category_items:
            for node in G.nodes():
                if node == target_object:
                    continue
                
                if G.nodes[node].get('type') != 'object':
                    continue
                    
                node_class = node.split('_')[0]
                
                # 그래프에 있는 노드가 타겟과 같은 카테고리 리스트에 있다면 Seed에 추가
                if node_class in target_category_items:
                    seed_nodes[node] = 0.5
                
    print(f">> Computed Seed Nodes: {seed_nodes}")
    
    ppr_score = nx.pagerank(G, personalization=seed_nodes, weight='weight', alpha=0.85)
    
    if candidate_list:
        receptacles = get_candidates_text_matching(G, candidate_list)
    
    # [NEW] 현재 위치(on 관계) 찾아서 제외하기
    current_holders = []
    if exclude_current:
        if target_object in G:
            for neighbor in G.neighbors(target_object):
                edge_data = G.get_edge_data(target_object, neighbor)
                # 관계가 'on'인 경우 리스트에 추가 (현재 놓여있는 곳)
                if edge_data.get('relation') == 'on':
                    current_holders.append(neighbor)
                    
        if current_holders:
            print(f"⚠️ Excluding current locations (on): {current_holders}")

    print(f"\n[PPR Location Result for '{target_object}']")
    
    sorted_locs = []
    if candidate_list:
        for loc in receptacles:
            # 현재 위치인 경우 Skip
            if exclude_current and (loc in current_holders):
                continue
            
        score = ppr_score.get(loc, 0.0)
    
    for loc in ppr_score:
        score = ppr_score.get(loc, 0.0)
        
        sorted_locs.append((loc, score))
            
    sorted_locs.sort(key=lambda x: x[1], reverse=True)
    
    # 3. 결과 분석 및 Ratio Check
    best_loc = sorted_locs[0][0] if sorted_locs else None
    
    if len(sorted_locs) >= 2:
        top1_score = sorted_locs[0][1]
        top2_score = sorted_locs[1][1]
        
        ratio = top1_score / top2_score if top2_score > 0 else 999.0
        # print(f"Top 1/2 Ratio: {ratio:.2f}")
    
    # entrophy
    top_k_scores = [score for _, score in sorted_locs[:20]] 
    entropy = calculate_entropy(top_k_scores)
    # print(f"Entropy : {entropy:.4f} (Lower is Better)") 

    print("\nTop Recommended Locations:")
    for loc, score in sorted_locs[:20]: 
        print(f"   - {loc}: {score:.6f}")  
    
    print(f"\nFinal Decision: Move '{target_object}' to -> '{best_loc}'")
    
    
    return best_loc

def ppr_tool(G, target_object, tool_list):
    
    # 1. Seed 설정 (도구를 쓸 대상)
    seed_nodes = {target_object: 1.0}
    
    # 2. PPR 실행
    ppr_score = nx.pagerank(G, personalization=seed_nodes, weight='weight', alpha=0.85)
    
    # 3. 후보군 필터링 (Tool List)
    candidates = get_candidates_text_matching(G, tool_list)
    
    # print(f"\n[PPR Tool Result for target '{target_object}']")
    
    sorted_tools = []
    for tool in candidates:
        score = ppr_score.get(tool, 0.0)
        sorted_tools.append((tool, score))
        
    sorted_tools.sort(key=lambda x: x[1], reverse=True)
    
    best_tool = sorted_tools[0][0] if sorted_tools else None
    
    if len(sorted_tools) >= 2:
        ratio = sorted_tools[0][1] / sorted_tools[1][1] if sorted_tools[1][1] > 0 else 999.0
        # print(f"Top 1/2 Ratio: {ratio:.2f}")

    print("\nTop Recommended Tools:")
    for tool, score in sorted_tools[:5]:
        print(f"   - {tool}: {score:.6f}")
        
    print(f"\nFinal Decision: Use '{best_tool}' for '{target_object}'")
    return best_tool


def get_similar_node(graph, list, model, threshold):
    scene_objects = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'object']
    
    clean_objects = [obj.split('_')[0] for obj in scene_objects]
    
    target_emb = model.encode(list, convert_to_tensor=True)
    obj_embs = model.encode(clean_objects, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(target_emb, obj_embs)[0]
    
    similar_node = []
    for idx, score in enumerate(cosine_scores):
        obj_name = scene_objects[idx]

        if score >= threshold:
            similar_node.append(obj_name)
            
    return similar_node


def visualize_sg(G, category_edge=False):
    plt.figure(figsize=(12, 10)) 
    pos = nx.spring_layout(G, k=0.5, iterations=50) # 간격 넓힘
    
    home_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'home']
    room_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'room']
    obj_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'object']
    
    nx.draw_networkx_nodes(G, pos, nodelist=home_nodes, node_color='red', node_size=1500, label='Home')
    nx.draw_networkx_nodes(G, pos, nodelist=room_nodes, node_color='skyblue', node_size=1000, label='Rooms')
    nx.draw_networkx_nodes(G, pos, nodelist=obj_nodes, node_color='lightgreen', node_size=300, label='Objects')
    
    if category_edge:
        cat_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'category'] # [NEW]
        nx.draw_networkx_nodes(G, pos, nodelist=cat_nodes, node_color='orange', node_size=1200, label='Categories') # [NEW]
    
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Scene Graph with Categories", fontsize=15)
    plt.legend()
    plt.axis('off')
    plt.show()
    

def calculate_entropy(scores):
    
    # 1. 점수 합계가 0이면 불확실성 최대
    total_score = sum(scores)
    if total_score == 0:
        return 999.0
        
    # 2. 확률 분포로 정규화 (Normalization)
    # (후보군 내에서의 확률 합을 1로 맞춰줌)
    probs = [s / total_score for s in scores]
    
    # 3. 엔트로피 계산 (-sum(p * log(p)))
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
            
    return entropy


if __name__ == "__main__":
    
    # if torch.cuda.is_available():
    #     device = 'cuda'
    #     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
    # model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    TOOL = [
        "rag", "wet-wipe", "tissue", "paper-napkin", "paper-wipe",
        "mop", "vacuum", "towel", "sponge"
    ]
    RECEPTACLE = [
        "table", "countertop", "sink", "cabinet", "sofa", "bed", 
        "bookshelf", "desk", "wardrobe", "toilet", "shelf", "coffee-table", "console"
    ]
    DISPOSE_RECEP = [
        "trash-bin", "garbage-bin", "trash-can", "garbage-can"
    ]
    
    category_edge = False
    exclude_current = False
    
    print("--- Building Graph ---")
    total_start_time = time.perf_counter()
    G = build_scene_graph(origin_sg, LLM_KNOWLEDGE_BASE, category_edge)
    

    TASK = 'relocate'    # relocate, washing-up, fold, dispose, mop, wipe, vacuum
    TARGET_OBJECT = 'spoon_0'

    
    print(f"\n>> Current Task: {TASK}")
    print(f">> Target Object: {TARGET_OBJECT}")

    
    if TASK in ['relocate', 'washing-up', 'fold', 'dispose']:
        
        if TASK == 'dispose':
            candidate_list = DISPOSE_RECEP
        else:
            candidate_list = False
            
        # if G.has_edge('spoon_0', 'sink_0'):
        #      G['spoon_0']['sink_0']['weight'] = 5.0
        #      print(">> [Simulated Memory] 'spoon_0' -> 'sink_0' weight boosted to 5.0")
             
        ppr_location(G, TARGET_OBJECT, candidate_list, exclude_current, single=False)
        
    elif TASK in ['mop', 'vacuum', 'wipe']:
        ppr_tool(G, TARGET_OBJECT, TOOL)
        
    else:
        print("Unknown Task")

    total_end_time = time.perf_counter()
    print(f"\nTotal Time: {total_end_time - total_start_time:.4f} sec")
    
    # visualize_sg(G, category_edge)