import copy
import networkx as nx
from collections import deque
from itertools import combinations

from .bin_split import graph_split


def orient_bfs(graph, source, targets, init_orient, max_depth=5):
    """
    带 orient 约束的单源 BFS，从指定方向出发。
    约束：相邻两条边，前一条的 to_orient = 后一条的 from_orient。
    首步仅选择 from_orient == init_orient 的出边。

    Args:
        graph: MultiDiGraph（边属性含 from_orient, to_orient）
        source: 起始节点
        targets: 目标节点集合（set）
        init_orient: 起始方向（'+' 或 '-'），首步只走该方向的出边
        max_depth: 最大路径边数

    Returns:
        到达第一个目标节点的最短合法路径 [path_nodes]，未找到则返回 None
    """
    visited = set()
    queue = deque()

    queue.append((source, None, [source], 0))
    visited.add((source, None))

    while queue:
        curr, last_to, path, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for _, nxt, edge_data in graph.out_edges(curr, data=True):
            f_orient = edge_data.get('from_orient')
            t_orient = edge_data.get('to_orient')

            # 首步：只走 init_orient 方向的出边
            if last_to is None:
                if f_orient != init_orient:
                    continue
            else:
                # 后续步：要求 last_to_orient == from_orient
                if last_to != f_orient:
                    continue

            state = (nxt, t_orient)
            if state in visited:
                continue
            visited.add(state)

            new_path = path + [nxt]

            # 到达目标节点，立即返回（BFS 保证最短）
            if nxt in targets:
                return new_path

            queue.append((nxt, t_orient, new_path, depth + 1))

    return None


def graph_refine(init_g, bins_egs, egs_bins, tolerance=0.3):
    """
    基于装配图连通子图的路径搜索，将共享 edge 按深度匹配分配给对应的 bin。

    Args:
        init_g: 完整的装配图 (MultiDiGraph)
        bins_egs: {bin_id: set(edge_ids)}
        egs_bins: {edge_id: set(bin_ids)}
        tolerance: 深度匹配容差

    Returns:
        refined_bins: {bin_id: {seq_name: sequence}}
    """

    # 1. 属性注入：初始化图节点
    for node in init_g.nodes():
        init_g.nodes[node]['bin_ids'] = set()
        init_g.nodes[node]['path_bins'] = set()

    # 2. 将节点和 bin id 绑定
    for edge_id, bin_id_set in egs_bins.items():
        if init_g.has_node(edge_id):
            init_g.nodes[edge_id]['bin_ids'].update(bin_id_set)

    # 3. 提取节点数 >= 3 的连通子图
    valid_subgraphs = []
    components = nx.connected_components(init_g.to_undirected()) if init_g.is_directed() else nx.connected_components(init_g)
    for comp in components:
        if len(comp) >= 3:
            valid_subgraphs.append(init_g.subgraph(comp).copy())

    print(f"提取到 {len(valid_subgraphs)} 个节点数 >= 3 的连通图。")

    updated_bins_egs = copy.deepcopy(bins_egs)

    splited_bins = {}

    for idx, sub_g in enumerate(valid_subgraphs):
        # ========== 按 bin_id 归类连通子图内的节点 ==========
        bin_node_groups = {}
        for node, data in sub_g.nodes(data=True):
            b_ids = data.get('bin_ids', set())
            for b_id in b_ids:
                if b_id not in bin_node_groups:
                    bin_node_groups[b_id] = []
                bin_node_groups[b_id].append(node)
        # ========== 先尝试 graph_split（所有 edge 视为一个 bin）==========
        all_comp_nodes = list(sub_g.nodes())
        if len(all_comp_nodes) >= 4:
            split_result = graph_split(init_g, all_comp_nodes)
            if len(split_result) > 1:
                comp_node_set = set(all_comp_nodes)
                group_sets = [set(g) for g in split_result]

                # 创建新 bin
                gs_names = []
                for i, sub_edges in enumerate(split_result):
                    gs_name = f"comp{idx}_gs{i}"
                    updated_bins_egs[gs_name] = set(sub_edges)
                    gs_names.append(gs_name)

                # 贪心一对一分配：每个 comp_gs 最多被一个原始 bin 合并
                bin_ids = list(bin_node_groups.keys())
                used_gs = set()
                used_bids = set()

                # 计算所有 (bin, group) 的重叠
                overlap_pairs = []
                for b_id in bin_ids:
                    bin_edges_in_comp = set(bin_node_groups[b_id])
                    for gi in range(len(group_sets)):
                        score_edge = bin_edges_in_comp & group_sets[gi]
                        score = sum([init_g.nodes[_eg]['length'] for _eg in score_edge])
                        if score > sum([init_g.nodes[_eg]['length'] for _eg in group_sets[gi]])*0.2:
                            overlap_pairs.append((score, b_id, gi))
                overlap_pairs.sort(reverse=True)

                # 贪心匹配
                updated_bin_name = {}
                for score, b_id, gi in overlap_pairs:
                    if b_id in used_bids or gi in used_gs:
                        continue

                    # 将原始 bin 的非组件 edge 合并到对应 comp_gs
                    remaining = updated_bins_egs[b_id] - comp_node_set
                    if remaining:  
                        updated_bins_egs[gs_names[gi]] |= remaining  
                    used_gs.add(gi)
                    used_bids.add(b_id)
                    updated_bin_name[gs_names[gi]] = f"{b_id}_{gs_names[gi]}"

                for gi in range(len(group_sets)):
                    if gi not in used_gs:
                        used_gs.add(gi)
                        updated_bin_name[gs_names[gi]] = f"N_{gs_names[gi]}"
                        

                # 匹配的 bin：删除（非连通图 edge 已合并到 comp_gs）
                for b_id in used_bids:
                    for _n in updated_bins_egs[b_id]:
                        init_g.nodes[_n]['bin_ids'].discard(b_id)

                    del updated_bins_egs[b_id]
                
                for gsi in gs_names:
                    nb_name = gsi
                    if gsi in updated_bin_name:
                        nb_name = updated_bin_name[gsi]
                        updated_bins_egs[nb_name] = updated_bins_egs.pop(gsi)

                    for _n in updated_bins_egs[nb_name]:
                        init_g.nodes[_n]['bin_ids'].add(nb_name)


                # 未匹配的 bin：移除组件内 edge（避免与 comp_gs 重复） 暂时不考虑删除
                # for b_id in bin_ids:
                #     if b_id not in used_bids:
                #         updated_bins_egs[b_id] = updated_bins_egs[b_id] - comp_node_set
                continue  # 跳过本连通分量的 refine

            for b_id, nodes in list(bin_node_groups.items()):
                split_result_eval = graph_split(init_g, nodes)
                if len(split_result_eval) > 1:
                    split_result = graph_split(init_g, list(updated_bins_egs[b_id]))
                    # 清理该 bin 所有节点上的 bin_id 引用（避免跨子图残留引发 KeyError）
                    for _n in updated_bins_egs[b_id]:
                        if init_g.has_node(_n):
                            init_g.nodes[_n]['bin_ids'].discard(b_id)
                    del updated_bins_egs[b_id]
                    del bin_node_groups[b_id]

                    for i, sub_edges in enumerate(split_result):
                        gs_name = f"{b_id}_comp{idx}_gs{i}"
                        updated_bins_egs[gs_name] = set(sub_edges)
                    # 将新 sub-bin 名称写回节点，保持 bin_ids 与 updated_bins_egs 一致
                    for _n in sub_edges:
                        if init_g.has_node(_n):
                            init_g.nodes[_n]['bin_ids'].add(gs_name)

        # ========== 为每个 bin 生成内部连通路径图 ==========
        bin_all_paths = {}
        for b_id, nodes in bin_node_groups.items():
            if len(nodes) < 2:
                continue

            node_set = set(nodes)
            paths = []
            for src in nodes:
                targets = node_set - {src}
                for orient in ['+', '-']:
                    path = orient_bfs(sub_g, src, targets, init_orient=orient, max_depth=5)
                    if path is not None:
                        paths.append(path)
            bin_all_paths[b_id] = paths

        # ========== 构建路径图节点集 ==========
        bin_path_nodes = {}
        for b_id, paths in bin_all_paths.items():
            nodes = set()
            for p in paths:
                nodes.update(p)
            bin_path_nodes[b_id] = nodes

        # 更新图节点的 path_bins 属性
        for b_id, nodes in bin_path_nodes.items():
            for node in nodes:
                if sub_g.has_node(node):
                    sub_g.nodes[node]['path_bins'].add(b_id)

        # ========== 公共节点深度分配 ==========
        # 计算加权 avg_dp
        bin_unique_nodes = {}
        for node, data in sub_g.nodes(data=True):
            if len(data['path_bins']) == 1:
                b_id = next(iter(data['path_bins']))
                dp = data.get('dp', 0)
                length = data.get('length', 0)
                bin_unique_nodes.setdefault(b_id, []).append((dp, length))

        avg_dp = {}
        for b_id, items in bin_unique_nodes.items():
            total_len = sum(l for _, l in items)
            if total_len > 0:
                avg_dp[b_id] = sum(dp * l for dp, l in items) / total_len

        # 公共节点判定与分配
        for node, data in sub_g.nodes(data=True):
            path_bins = data['path_bins']
            bin_ids = data['bin_ids']
            c = len(path_bins)
            if c < 2 or c > 5:
                continue
            if len(bin_ids) > 1:
                continue

            dp_v = data.get('dp', 0)
            involved = list(path_bins)


            if len(bin_ids) == 1 and next(iter(bin_ids)) in involved:
                # 情况 A：节点已属于一个 bin
                b1 = next(iter(bin_ids))
                if c == 2:
                    b2 = [b for b in involved if b != b1][0]
                    if (b1 in avg_dp and b2 in avg_dp
                        and dp_v > (avg_dp[b1] + avg_dp[b2]) * 0.75
                        and dp_v > max(avg_dp[b1], avg_dp[b2])):
                        updated_bins_egs[b2].add(node)
                else:
                    others = [b for b in involved if b != b1]
                    best_err, best_S = float('inf'), None
                    for r in range(1, len(others) + 1):
                        for combo in combinations(others, r):
                            S = [b1] + list(combo)
                            if not all(b in avg_dp for b in S):
                                continue
                            sum_dp = sum(avg_dp[b] for b in S)
                            err = abs(sum_dp - dp_v) / dp_v if dp_v > 0 else float('inf')
                            if err < tolerance and err < best_err:
                                best_err, best_S = err, S
                    if best_S:
                        for b in best_S:
                            if b != b1:
                                updated_bins_egs[b].add(node)

            else:
                # 情况 B：Unbinned 或不在当前 involved 列表中
                if c == 2:
                    b1, b2 = involved
                    if (b1 in avg_dp and b2 in avg_dp
                        and dp_v > (avg_dp[b1] + avg_dp[b2]) * 0.75
                        and dp_v > max(avg_dp[b1], avg_dp[b2])):
                        updated_bins_egs[b1].add(node)
                        updated_bins_egs[b2].add(node)
                else:
                    best_err, best_S = float('inf'), None
                    for r in range(1, len(involved) + 1):
                        for combo in combinations(involved, r):
                            S = list(combo)
                            if not all(b in avg_dp for b in S):
                                continue
                            sum_dp = sum(avg_dp[b] for b in S)
                            err = abs(sum_dp - dp_v) / dp_v if dp_v > 0 else float('inf')
                            if err < tolerance and err < best_err:
                                best_err, best_S = err, S
                    if best_S:
                        for b in best_S:
                            updated_bins_egs[b].add(node)

    updated_bins_egs_flag = {}
    for k in bins_egs.keys():
        if k not in updated_bins_egs:
            continue  # 已被 graph_split 删除

        if len(updated_bins_egs[k]) > len(bins_egs[k]):
            updated_bins_egs_flag[k+'_gr'] = updated_bins_egs[k]
        else:
            updated_bins_egs_flag[k] = bins_egs[k]
    for k in updated_bins_egs.keys():
        if k not in bins_egs:
            updated_bins_egs_flag[k] = updated_bins_egs[k]

    return updated_bins_egs_flag
