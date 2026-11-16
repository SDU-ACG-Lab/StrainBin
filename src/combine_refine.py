from itertools import combinations


def find_triplets_with_tolerance(nums_dict, tolerance=0.1):
    """
    在 nums_dict 中寻找满足 val_i + val_j ≈ val_k 的三元组。

    Args:
        nums_dict: {name: value}
        tolerance: 相对误差容差

    Returns:
        [(name_i, name_j, name_k), ...] 其中 val_i + val_j ≈ val_k
    """
    items = sorted(nums_dict.items(), key=lambda x: x[1])
    n = len(items)
    results = []

    for k in range(2, n):
        target_name, target_val = items[k]
        if target_val <= 10:
            continue

        i, j = 0, k - 1

        while i < j:
            name_i, val_i = items[i]
            name_j, val_j = items[j]

            current_sum = val_i + val_j
            diff = abs(current_sum - target_val) / target_val

            if diff <= tolerance:
                results.append((name_i, name_j, target_name))
                i += 1
                j -= 1
            elif current_sum < target_val * (1 - tolerance):
                i += 1
            else:
                j -= 1

    return list(set(results))


def combine_refine(bins_items, item_covs, item_lens, item_marker):
    """
    基于覆盖度三元组匹配 + marker 验证，合并因装配分支导致的碎片 bin。

    识别满足 cov(A) + cov(B) ≈ cov(T) 的三元组，验证 marker 条件后
    将 A∪B∪T 合并为一个新 bin。

    Args:
        bins_items: {bin_id: set(item_ids)}
        item_covs: {item_id: float} — 单值覆盖度
        item_lens: {item_id: int}
        item_marker: {item_id: [marker_list]}

    Returns:
        combined_bin: {new_bin_id: set(item_ids)} — 合并后的 bin
        left_bin: {bin_id: set(item_ids)} — 未参与合并的 bin
    """
    # 计算每个 bin 的加权平均覆盖度
    bins_cov = {}
    for _b, _cs in bins_items.items():
        total_weight = sum(item_covs.get(_c, 0) * item_lens.get(_c, 0) for _c in _cs)
        total_len = sum(item_lens.get(_c, 0) for _c in _cs)
        bins_cov[_b] = int(total_weight / total_len) if total_len > 0 else 0

    trip_list = find_triplets_with_tolerance(bins_cov, 0.1)

    # 计算每个 bin 的 marker gene 集合
    bins_marker = {}
    for _b, _cs in bins_items.items():
        bins_marker[_b] = set()
        for _c in _cs:
            if _c in item_marker:
                bins_marker[_b] |= set(item_marker[_c])

    # 对每个三元组打分
    trip_score = []
    for tri in trip_list:
        m1, m2, mt = bins_marker[tri[0]], bins_marker[tri[1]], bins_marker[tri[2]]
        comp1 = len(m1 | mt)
        comp2 = len(m2 | mt)
        cont1 = len(m1 & mt)
        cont2 = len(m2 & mt)

        balance = comp1 > 0.6 * comp2 and comp2 > 0.6 * comp1
        branch = len(m1) > 20 and len(m2) > 20
        purity = cont1 < 10 and cont2 < 10
        not_comp = len(m1) < 90 and len(m2) < 90

        if balance and branch and purity and not_comp:
            trip_score.append(comp1 + comp2 - (cont1 + cont2))
        else:
            trip_score.append(0)

    # 按得分降序贪心合并
    sorted_indices = sorted(range(len(trip_score)), key=lambda i: trip_score[i], reverse=True)

    used_bins = set()
    # {(bini, binj): [bint1, bint2, ...]} — 记录每对分支及其关联的 trunk
    branch_pairs = {}

    for idc in sorted_indices:
        if trip_score[idc] < 10:
            break

        bini, binj, bint = trip_list[idc]

        # bint 不能被重复使用
        if bint in used_bins:
            continue

        pair = (bini, binj)

        # 链式合并：同一对 (bini, binj) 已存在时，追加新 trunk
        if pair in branch_pairs:
            existing_marker = set()
            for _t in branch_pairs[pair]:
                existing_marker.update(bins_marker[_t])
            if len(existing_marker & bins_marker[bint]) < len(bins_marker[bint]) * 0.3:
                branch_pairs[pair].append(bint)
                used_bins.add(bint)
            continue

        # 正常合并：bini 和 binj 都不能已被使用
        if bini in used_bins or binj in used_bins:
            continue

        branch_pairs[pair] = [bint]
        used_bins.update((bini, binj, bint))

    # 生成合并后的 bin：每对分支 + 共享 trunk → 两个独立 bin
    combined_bin = {}
    for (bini, binj), trunks in branch_pairs.items():
        trunk_items = set()
        for _t in trunks:
            trunk_items |= set(bins_items[_t])
        combined_bin[f"{bini}_cb"] = set(bins_items[bini]) | trunk_items
        combined_bin[f"{binj}_cb"] = set(bins_items[binj]) | trunk_items

    left_bin = {k: v for k, v in bins_items.items() if k not in used_bins}
    return combined_bin, left_bin
