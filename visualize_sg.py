import math
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
    

def calculate_hierarchical_pos_3d(G):
    """
    Home -> Room -> Object 순서의 수직 계층 구조와
    Category를 측면에 배치하는 좌표를 계산합니다.
    """
    pos = {}
    
    # 1. 노드 분류
    home_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'home']
    room_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'room']
    cat_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'category']
    obj_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'object']

    # 2. 좌표 설정 파라미터
    Z_HOME = 3.0
    Z_ROOM = 2.0
    Z_CAT  = 1.5  # 룸과 오브젝트 사이 높이
    Z_OBJ  = 0.0
    
    ROOM_RADIUS = 6.0    # Home을 중심으로 Room들이 퍼지는 반경
    OBJ_RADIUS = 1.5     # Room 바로 아래에서 Object들이 퍼지는 반경
    CAT_X_OFFSET = 12.0  # 카테고리가 배치될 X 좌표 (측면으로 뺌)
    
    # 3. Home 배치 (중앙 최상단)
    for node in home_nodes:
        pos[node] = (0, 0, Z_HOME)
        
    # 4. Room 배치 (Home 주변 원형 배치)
    num_rooms = len(room_nodes)
    room_coords = {} # 나중에 Object 배치할 때 참조하기 위해 저장
    
    for i, node in enumerate(room_nodes):
        angle = 2 * math.pi * i / num_rooms
        x = ROOM_RADIUS * math.cos(angle)
        y = ROOM_RADIUS * math.sin(angle)
        pos[node] = (x, y, Z_ROOM)
        room_coords[node] = (x, y) # (x, y) 저장

    # 5. Object 배치 (부모 Room 아래에 군집화)
    # 각 Object가 어떤 Room에 있는지 찾기 (그래프 엣지 이용)
    # 만약 연결된 Room이 없으면 중앙 바닥에 배치
    
    # 5-1. Room별 Object 리스트 생성
    room_to_objs = {room: [] for room in room_nodes}
    orphaned_objs = []
    
    for obj in obj_nodes:
        assigned = False
        neighbors = list(G.neighbors(obj))
        for nb in neighbors:
            # 엣지 관계가 'in'이고 상대방이 Room인 경우
            edge_data = G.get_edge_data(nb, obj) or G.get_edge_data(obj, nb)
            if G.nodes[nb].get('type') == 'room': # 관계 체크 없이 Room 타입이면 소속으로 인정 (단순화)
                room_to_objs[nb].append(obj)
                assigned = True
                break
        if not assigned:
            orphaned_objs.append(obj)
            
    # 5-2. 좌표 할당
    for room, objs in room_to_objs.items():
        if not objs: continue
        cx, cy = room_coords[room]
        num_objs = len(objs)
        
        for i, obj in enumerate(objs):
            # Room 좌표 (cx, cy)를 중심으로 작은 원 그리기
            # Object가 많으면 나선형으로 배치하여 겹침 방지
            radius_variation = OBJ_RADIUS + (0.1 * (i % 3)) # 약간의 변화
            angle = 2 * math.pi * i / num_objs
            
            ox = cx + radius_variation * math.cos(angle)
            oy = cy + radius_variation * math.sin(angle)
            pos[obj] = (ox, oy, Z_OBJ)

    # 연결 안 된 고아 객체 처리 (중앙 바닥)
    for i, obj in enumerate(orphaned_objs):
        angle = 2 * math.pi * i / (len(orphaned_objs) or 1)
        pos[obj] = (2 * math.cos(angle), 2 * math.sin(angle), Z_OBJ)

    # 6. Category 배치 (측면에 벽처럼 정렬)
    # Y축을 따라 일렬로 배치하거나 격자로 배치
    num_cats = len(cat_nodes)
    cols = 2 # 2열로 배치
    rows = math.ceil(num_cats / cols)
    
    for i, node in enumerate(cat_nodes):
        r = i // cols
        c = i % cols
        # Y축 중심으로 퍼지게
        y = (r - rows/2) * 1.5 
        # X축은 고정 (측면), Z축은 약간의 변화를 주거나 고정
        x = CAT_X_OFFSET + (c * 1.5)
        pos[node] = (x, y, Z_CAT)
        
    return pos


def visualize_sg_3d_plotly_hierarchical(G, target_object=None):
    print("Generating Hierarchical 3D Visualization...")
    
    # [핵심] 계층형 좌표 계산 함수 호출
    pos_3d = calculate_hierarchical_pos_3d(G)

    # =========================================
    # [Edge Traces] 엣지 그리기
    # =========================================
    edge_traces = []
    
    relation_styles = {
        'on': {'color': 'blue', 'width': 3, 'opacity': 0.6},
        'next to': {'color': 'green', 'width': 2, 'opacity': 0.5},
        'is_a': {'color': 'orange', 'width': 1, 'opacity': 0.3}, # 카테고리 연결선은 연하게
        'in': {'color': 'gray', 'width': 1, 'opacity': 0.2},     # 룸-오브젝트 연결선은 아주 연하게 (수직선 느낌)
        'contains': {'color': 'red', 'width': 2, 'opacity': 0.2} # 홈-룸
    }

    for u, v, data in G.edges(data=True):
        if u not in pos_3d or v not in pos_3d: continue
        
        rel = data.get('relation', 'other')
        style = relation_styles.get(rel, {'color': 'gray', 'width': 1, 'opacity': 0.2})
        
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        
        edge_trace = go.Scatter3d(
            x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None],
            mode='lines',
            line=dict(width=style['width'], color=style['color']),
            opacity=style['opacity'],
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)

    # =========================================
    # [Node Traces] 노드 그리기
    # =========================================
    node_styles = {
        'home': {'color': 'red', 'size': 12, 'symbol': 'circle'},
        'room': {'color': 'skyblue', 'size': 10, 'symbol': 'circle'},
        'category': {'color': 'orange', 'size': 8, 'symbol': 'circle'},
        'object': {'color': 'lightgreen', 'size': 5, 'symbol': 'circle'}
    }
    target_style = {'color': 'magenta', 'size': 5, 'symbol': 'circle'}

    node_x, node_y, node_z = [], [], []
    node_colors, node_sizes, node_symbols = [], [], []
    node_texts = []

    for node, coord in pos_3d.items():
        x, y, z = coord
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        data = G.nodes[node]
        node_type = data.get('type', 'object')
        
        if node == target_object:
            style = target_style
            hover_txt = f"<b>TARGET: {node}</b><br>Type: {node_type}"
        else:
            style = node_styles.get(node_type, node_styles['object'])
            hover_txt = f"{node}<br>({node_type})"
            
        node_colors.append(style['color'])
        node_sizes.append(style['size'])
        node_symbols.append(style.get('symbol', 'circle'))
        node_texts.append(hover_txt)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text', # 텍스트(라벨) 함께 표시 (복잡하면 'markers'로 변경)
        marker=dict(
            symbol=node_symbols,
            size=node_sizes,
            color=node_colors,
            line=dict(color='black', width=0.5),
            opacity=0.9
        ),
        text=[n for n in pos_3d.keys()], # 노드 이름 표시
        textposition="top center",
        textfont=dict(size=8, color='white'), # 텍스트 스타일
        hovertext=node_texts,
        hoverinfo='text'
    )

    # =========================================
    # [Layout] Scene 설정
    # =========================================
    layout = go.Layout(
        title='Hierarchical 3D Scene Graph (Home -> Room -> Object)',
        width=1200, height=1200,
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, title='', visible=False),
            yaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, title='', visible=False),
            zaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, title='Height Level', visible=False),
            bgcolor='rgb(20, 20, 20)',
            # 카메라 시점 초기화 (약간 위에서 내려다보도록)
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
    fig.show()