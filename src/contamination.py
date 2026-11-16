import numpy as np
from collections import defaultdict
from sklearn.cluster import dbscan
from sklearn.neighbors import kneighbors_graph


def check_contamination(bins_egs, contig_marker, threshold=0.1):
    """
    检测每个 bin 的 marker gene 污染度，按阈值分为干净/污染两组。

    Args:
        bins_egs: {bin_id: set(edge_ids)}
        contig_marker: {contig_id: [marker_list]}
        threshold: 污染度阈值，默认 0.1

    Returns:
        clean: {bin_id: set(edge_ids)} — 污染度 < 阈值的 bin
        contaminated: {bin_id: set(edge_ids)} — 污染度 >= 阈值的 bin
    """
    clean = {}
    contaminated = {}

    for bin_name, egs in bins_egs.items():
        marker_list = []
        for eg in egs:
            if eg in contig_marker:
                marker_list.extend(contig_marker[eg])

        if len(marker_list) == 0:
            clean[bin_name] = egs
            continue

        contamination = (len(marker_list) - len(set(marker_list))) / len(marker_list)
        if contamination > threshold:
            contaminated[bin_name] = egs
        else:
            clean[bin_name] = egs

    print(f"污染检测: {len(clean)} 个干净 bin, {len(contaminated)} 个污染 bin (阈值={threshold})")
    return clean, contaminated


def _get_best_bin(dbscan_results, contig_marker, namelist, processed_ctg, contig_len, minfasta, beta=1):
    """从多组 DBSCAN 结果中贪心选择 F1 最高的 bin。"""
    for max_contamination in [0.1, 0.2, 0.3, 0.4, 0.5, 1]:
        max_F1 = 0
        weight_of_max = 1e9
        max_bin = None

        for res_labels in dbscan_results:
            res = defaultdict(list)
            for label, name in zip(res_labels, namelist):
                if label != -1 and name not in processed_ctg:
                    res[label].append(name)
            for bin_contig in res.values():
                cur_weight = sum(contig_len[contig] for contig in bin_contig)
                if cur_weight < minfasta:
                    continue
                marker_list = []
                for contig in bin_contig:
                    if contig in contig_marker:
                        marker_list.extend(contig_marker[contig])
                if len(marker_list) == 0:
                    continue
                recall = len(set(marker_list)) / 107
                contamination = (len(marker_list) - len(set(marker_list))) / len(marker_list)
                F1 = (1 + beta * beta) * recall * (1 - contamination) / (
                    recall + (1 - contamination) * beta * beta)

                if contamination <= max_contamination:
                    if F1 > max_F1:
                        max_F1 = F1
                        weight_of_max = cur_weight
                        max_bin = bin_contig
                    elif F1 == max_F1 and cur_weight <= weight_of_max:
                        weight_of_max = cur_weight
                        max_bin = bin_contig
        if max_F1 > 0:
            return max_bin
    return None


def _run_dbscan_multi_eps(egs_list, egs_emb, lens_list, eps_list=None, threads=32, min_samples=5):
    """使用 KNN 稀疏距离矩阵运行多 eps DBSCAN，返回所有结果。"""
    if eps_list is None:
        eps_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    N = len(egs_list)
    n_neighbors = min(200, N - 1)

    dist_matrix = kneighbors_graph(
        egs_emb,
        n_neighbors=n_neighbors,
        mode='distance',
        p=2,
        n_jobs=threads)

    dbscan_results = []
    for eps_value in eps_list:
        _, labels = dbscan(dist_matrix,
                           eps=eps_value, min_samples=min_samples, n_jobs=threads, metric='precomputed',
                           sample_weight=lens_list)
        dbscan_results.append(labels.tolist())

    return dbscan_results


def per_bin_redbscan(contaminated_bins, init_g, eg_rev, eg_emb_norm, contig_marker, minfasta=200000, mode='comebin'):
    """
    策略 A：对每个污染 bin 独立执行 DBSCAN 去重聚类。

    Args:
        contaminated_bins: {bin_id: set(edge_ids)}
        init_g: 装配图
        eg_rev: {edge_id: index} embedding 索引映射
        eg_emb_norm: 归一化后的 embedding 矩阵
        contig_marker: {contig_id: [marker_list]}
        minfasta: 最小 bin 大小
        mode: 'comebin' 或 'semibin'，影响 eps_list 和 min_samples

    Returns:
        result_bins: {bin_id: set(edge_ids)}
    """
    if mode == 'semibin':
        eps_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
        min_samples = 5
    else:
        eps_list = None  # 使用默认
        min_samples = 5

    result_bins = {}

    for bin_name, egs in contaminated_bins.items():
        egs_list = [eg for eg in egs if eg in eg_rev]
        if len(egs_list) < 5:
            result_bins[bin_name] = egs
            continue

        egs_idx = [eg_rev[eg] for eg in egs_list]
        egs_emb = eg_emb_norm[egs_idx]
        lens_list = [init_g.nodes[eg].get('length', 0) for eg in egs_list]
        lens_dt = {egs_list[i]: lens_list[i] for i in range(len(egs_list))}

        dbscan_results = _run_dbscan_multi_eps(egs_list, egs_emb, lens_list,
                                                eps_list=eps_list, min_samples=min_samples)

        # 贪心选择最佳 bin
        best_bins = []
        processed = set()
        remaining_len = sum(lens_list)

        while remaining_len >= minfasta:
            max_bin = _get_best_bin(dbscan_results, contig_marker, egs_list, processed, lens_dt, minfasta)
            if not max_bin:
                break
            best_bins.append(max_bin)
            processed.update(max_bin)
            remaining_len -= sum(lens_dt[c] for c in max_bin)

        if len(best_bins) > 1:
            for i, sb in enumerate(best_bins):
                result_bins[f"{bin_name}_pe{i}"] = set(sb)
        else:
            result_bins[bin_name] = egs

    return result_bins


def pooled_redbscan(contaminated_bins, init_g, eg_rev, eg_emb_norm, contig_marker, minfasta=200000, mode='comebin'):
    """
    策略 B：汇总所有污染 bin 的 edge，统一执行 DBSCAN 聚类。

    Args:
        contaminated_bins: {bin_id: set(edge_ids)}
        init_g: 装配图
        eg_rev: {edge_id: index} embedding 索引映射
        eg_emb_norm: 归一化后的 embedding 矩阵
        contig_marker: {contig_id: [marker_list]}
        minfasta: 最小 bin 大小
        mode: 'comebin' 或 'semibin'，影响 eps_list 和 min_samples

    Returns:
        result_bins: {bin_id: set(edge_ids)}
    """
    if mode == 'semibin':
        eps_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
        min_samples = 5
    else:
        eps_list = None
        min_samples = 5

    # 汇总所有污染 bin 的 edge
    pooled_egs = set()
    for egs in contaminated_bins.values():
        pooled_egs.update(egs)

    pooled_list = [eg for eg in pooled_egs if eg in eg_rev]
    if len(pooled_list) < 5:
        return {k: v for k, v in contaminated_bins.items()}

    pooled_idx = [eg_rev[eg] for eg in pooled_list]
    pooled_emb = eg_emb_norm[pooled_idx]
    lens_list = [init_g.nodes[eg].get('length', 0) for eg in pooled_list]
    lens_dt = {pooled_list[i]: lens_list[i] for i in range(len(pooled_list))}

    dbscan_results = _run_dbscan_multi_eps(pooled_list, pooled_emb, lens_list,
                                            eps_list=eps_list, min_samples=min_samples)

    # 贪心选择最佳 bin
    result_bins = {}
    best_bins = []
    processed = set()
    remaining_len = sum(lens_list)

    while remaining_len >= minfasta:
        max_bin = _get_best_bin(dbscan_results, contig_marker, pooled_list, processed, lens_dt, minfasta)
        if not max_bin:
            break
        best_bins.append(max_bin)
        processed.update(max_bin)
        remaining_len -= sum(lens_dt[c] for c in max_bin)

    for i, sb in enumerate(best_bins):
        result_bins[f"pr_{i}"] = set(sb)

    return result_bins
