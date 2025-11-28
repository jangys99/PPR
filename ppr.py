import networkx as nx
import re
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util


origin_sg = """Environment: In <livingroom_0>, liquid_1 on table_1, chair_0 on floor, chips_1 on sofa_0, plate_2 on table_1, paper_0 on table_1, blanket_1 on sofa_0, bread_0 on table_1, cabinet_0 on floor, cabinet_2 on floor, chips_0 next to garbage-bin_0, chips_0 on floor, cup_0 on coffee-table_1, cupboard_0 on floor, debris_0 next to garbage-bin_0, debris_0 on floor, debris_3 next to paper-wipe_0, debris_3 on table_1, desk-lamp_1 on table_1, garbage-bin_0 on floor, liquid_3 next to paper-towel_1, liquid_3 on sofa_0, paper-towel_1 on floor, paper-wipe_0 next to cabinet_0, paper-wipe_0 on table_1, sand_0 next to garbage-bin_0, sand_0 on floor, shoes_0 on floor, sofa_0 on floor, table_1 on floor, vacuum_0 on floor. In <library_0>, ball_0 next to console_2, ball_0 on floor, cabinet_1 on floor, console_2 on floor, crumbs_0 next to trash-bin_1, crumbs_0 on floor, debris_1 next to wet-wipe_4, debris_1 on console_2, desk-lamp_0 on console_2, newspaper_0 next to console_2, newspaper_0 next to trash-bin_1, newspaper_0 on floor, shelf_0 on floor, trash-bin_1 on floor, wet-wipe_4 on console_2. In <kitchen_0>, book_2 next to liquid_0, book_2 next to trash-can_0, book_2 on floor, cabinet_3 on floor, debris_2 on floor, dust_2 next to trash-can_0, dust_2 on floor, fluid_4 next to rag_0, fluid_4 on countertop_0, fork_0 next to sink_0, fork_0 on countertop_0, liquid_0 next to mop_0, liquid_0 next to paper-napkin_2, liquid_0 on floor, mop_0 on coutertop_0, oven_0 on floor, paper-napkin_2 on floor, particle_1 next to trash-can_0, particle_1 on floor, plate_0 next to sink_0, plate_0 on cabinet_3, plate_1 next to sink_0, plate_1 on countertop_0, pot_0 on floor, rag_0 next to countertop_0, rag_0 next to trash-can_0, rag_0 on floor, rag_1 next to debris_2, rag_1 next to trash-can_0, rag_1 on floor, range_0 on countertop_0, sink_0 on floor, spoon_0 next to sink_0, spoon_0 on countertop_0, stove_0 on floor, trash-can_0 on floor. In <bedroom_0>, bed_0 on floor, blanket_0 on bed_0, closet_1 on floor, ledge_1 on floor, liquid_2 next to tissue_3, liquid_2 on floor, pants_0 next to wardrobe_1, pants_0 on floor, pillow_0 next to bed_0, pillow_0 on floor, stovepipe-jeans_1 next to wardrobe_1, stovepipe-jeans_1 on floor, tissue_3 on floor, wardrobe_1 on floor. In <bathroom_0>, mud_1 next to paper-towel_0, mud_1 on toilet_0, paper-towel_0 next to toilet_0, paper-towel_0 on floor, sink_1 on floor, tissue_0 on countertop_3, toilet_0 on floor, towel_0 on floor, countertop_3 next to sink_1."""


def build_scene_graph(text):
    G = nx.Graph()
    
    home_node = 'Home'
    G.add_node(home_node, type='home')
    
    room_split = re.split(r'In <([^>]+)>', text)
    
    current_room = None
    
    for i in range(1, len(room_split), 2):
        room_name = room_split[i]
        content = room_split[i+1]
        
        G.add_node(room_name, type='room')
        G.add_edge(home_node, room_name, weight=2.0, relation='contains')
        
        items = [x.strip() for x in content.split(',') if x.strip()]
        
        for item in items:
            if "floor" in item:
                item = item.replace("floor", f"{room_name}_floor")
                
            match = re.match(r'(.+?)\s+(on|next to)\s+(.+)', item)
            
            if match:
                obj1, rel, obj2 = match.groups()
                
                G.add_node(obj1, type='object')
                G.add_node(obj2, type='object')
                
                if rel == 'on':
                    w = 1.0
                else:
                    w = 0.5
                
                G.add_edge(obj1, obj2, weight=w, relation=rel)
                
                G.add_edge(room_name, obj1, weight=1.0, relation='in')
                G.add_edge(room_name, obj2, weight=1.0, relation='in')
            
            else:
                obj1 = item.replace('.', '')
                
                if obj1:
                    G.add_node(obj1, type='object')
                    G.add_edge(room_name, obj1, weight=1.0, relation='in')
                    
    return G


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


def get_candidates_batch(graph, tool_list, model, threshold):
    scene_objects = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'object']
    
    if not scene_objects:
        return []

    clean_objects = [obj.split('_')[0] for obj in scene_objects]

    tool_embeddings = model.encode(tool_list, convert_to_tensor=True)
    obj_embeddings = model.encode(clean_objects, convert_to_tensor=True)

    cosine_scores = util.cos_sim(tool_embeddings, obj_embeddings)

    max_scores, max_indices = torch.max(cosine_scores, dim=0)

    candidates = []
    
    for idx, score in enumerate(max_scores):
        if score >= threshold:
            obj_name = scene_objects[idx]      
            matched_tool = tool_list[max_indices[idx]] 
            
            # print(f"   [Found] {obj_name} matches '{matched_tool}' (Score: {score:.4f})")
            candidates.append(obj_name)
            
    return candidates
    

def ppr_location(G, model, target_object, list, memory_stats=None):
    similar_node = get_similar_node(G, target_object, model, threshold=0.5)
    print(f"Similar node: {similar_node}")
    
    seed_weight = 1.0 / len(similar_node)
    seed_nodes = {node: seed_weight for node in similar_node}
    
    ppr_score = nx.pagerank(G, personalization=seed_nodes, weight='weight', alpha=0.85)
    
    receptacles = get_candidates_batch(G, list, model, threshold=0.7)
    
    target_class = target_object.split('_')[0]
    obj_stats = {}
    if memory_stats and target_class in memory_stats:
        obj_stats = memory_stats[target_class]
        
    ranked_stats = sorted(obj_stats.items(), key=lambda item: item[1], reverse=True)
    
    rank_list = {}
    for rank, (loc_class, count) in enumerate(ranked_stats):
        if rank == 0:  
            rank_list[loc_class] = 10.0 
        elif rank == 1: 
            rank_list[loc_class] = 4.0
        else:        
            rank_list[loc_class] = 0.5
    
    print("\nTop Recommended Locations:")
    best_loc = None
    max_score = -1
    
    sorted_locs = []
    for loc in receptacles:
        raw_score = ppr_score.get(loc, 0.0)
        
        loc_class = loc.split('_')[0]
        mem_alpha = rank_list.get(loc_class, 0.0)
        count = obj_stats.get(loc_class, 0)
                
        final_score = raw_score * (1 + mem_alpha)
        
        sorted_locs.append((loc, final_score, raw_score, count))
        
        if final_score > max_score:
            max_score = final_score
            best_loc = loc
            
    sorted_locs.sort(key=lambda x: x[1], reverse=True)
    
    for loc, f_score, r_score, cnt in sorted_locs[:5]: 
        print(f"   - {loc}: {f_score:.6f} (PPR: {r_score:.4f} | Mem: {cnt}회)")  
                   
    print(f"\nFinal Decision: Move '{target_object}' to -> '{best_loc}'")
    
    return best_loc, similar_node


def ppr_tool(G, model, target_object, list):
    candidates = get_candidates_batch(G, list, model, threshold=0.9)

    seed_node = {f'{target_object}': 1.0}
    ppr_score = nx.pagerank(G, personalization=seed_node, weight='weight', alpha=0.85)
    
    best_tool = None
    max_score = -1
    
    for cand in candidates:
        score = ppr_score.get(cand, 0.0)
        print(f"-{cand}: {score:.6f}")
        
        if score > max_score:
            max_score = score
            best_tool = cand
    
    print(f'\n Final tool: {best_tool}')
    
    return best_tool, seed_node

        
def visualize_sg(G, highlights=None):
    plt.figure(figsize=(15, 12)) 
    
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    home_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'home']
    room_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'room']
    obj_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'object']
    
    
    nx.draw_networkx_nodes(G, pos, nodelist=home_nodes, node_color='red', node_size=2000, alpha=0.8, label='Home')
    nx.draw_networkx_nodes(G, pos, nodelist=room_nodes, node_color='skyblue', node_size=1000, alpha=0.8, label='Rooms')
    nx.draw_networkx_nodes(G, pos, nodelist=obj_nodes, node_color='lightgreen', node_size=300, alpha=0.6, label='Objects')
    
    if highlights:
        for node, color in highlights.items():
            if node in G.nodes():
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_size=300, alpha=0.6)
                
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3)
    
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.title("Scene Graph Visualization", fontsize=20)
    plt.legend()
    plt.axis('off')
    plt.show()
    

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    TOOL = [
        "rag", "wet-wipe", "tissue", "paper-napkin", "paper-wipe",
        "mop", "vacuum", "towel", "sponge"
    ]
    
    RECEPTACLE = [
        "table", "countertop", "sink", "cabinet", "sofa", "bed", 
        "bookshelf", "desk", "wardrobe", "toilet", "shelf"
    ]
    
    DISPOSE_RECEP = [
        "trash-bin", "garbage-bin", "trash-can", "garbage-can"
    ]
    
    TASK = 'fold'     # leave, relocate, reorient, washing-up, fold, mop, wipe, vacuum, dispose, empty, turn-off, close
    target_object = 'blanket_1'

    memory_stats = {
        "spoon": {
            "sink": 15,
            "countertop": 14
        },
        "fork": {
            "sink": 8,
            "countertop": 4
        },
        "plate": {
            "sink": 20,
            "table": 4,
            "countertop": 2
        }
    }
    
    G = build_scene_graph(origin_sg)
    highlights = {}
    
    if TASK == 'relocate' or 'dispose' or 'washing-up' or 'fold':
        if TASK == 'dispose':
            location, seeds = ppr_location(G, model, target_object, DISPOSE_RECEP, memory_stats=memory_stats)
        else:
            location, seeds = ppr_location(G, model, target_object, RECEPTACLE, memory_stats=memory_stats)
        
        highlights[target_object] = 'orange'
        for s in seeds:
            highlights[s] = 'hotpink'
            
        if location:
            highlights[location] = 'gold'
    
    elif TASK == 'mop' or 'vacuum' or 'wipe':
        best_tool, seed_node = ppr_tool(G, model, target_object, TOOL)
        
        for node in seed_node.keys():
            highlights[node] = 'orange'
            
        if best_tool:
            highlights[best_tool] = 'gold'
    

    visualize_sg(G, highlights=highlights)

    
    # print(f"Final Candidates: {candidates}")
    
    # print(f"- 노드 개수: {len(G.nodes())}")
    # print(f"- 엣지 개수: {len(G.edges())}")
    # print(f"- Home과 연결된 방: {[n for n in G.neighbors('Home')]}")

    # path = nx.shortest_path(G, "liquid_1", "sink_0")
    # print(f"- 경로: {path}")