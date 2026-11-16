import numpy as np
import networkx as nx
import leidenalg
from igraph import Graph as iGraph
from typing import List, Optional, Union

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan


def has_approx_sum(nums, er = 0.25):
    # a=b+c
    k = len(nums)
    for i in range(k):
        a = nums[i]
        for j in range(k):
            if j == i:
                continue
            for l in range(k):
                if l == i or l == j:
                    continue
                b = nums[j]
                c = nums[l]
                if abs(a - (b + c))  < a * er:
                    # print(abs(a - (b + c))  /a )
                    return True, (a, b, c), (i,j,l)
    return False, None, None


def in_bin_cluster(dps):
    X = np.array(dps).reshape(-1,1)

    # rep = 10 
    # best_labels = []
    # best_k = 3
    # best_score = -1
    # max_k = 4

    # if len(dps) < 5:
    #     return [0 for _ in dps]

    # if np.mean(dps) < 5:
    #     return [0 for _ in dps]
    
    # for k in range(3, max_k + 1):
    #     for r in range(rep):
    #         labels = KMeans(n_clusters=k).fit_predict(X)

    #         if len(set(labels)) == len(dps):
    #             continue

    #         score = silhouette_score(X, labels)
    #         if score > best_score:
    #             best_k = k
    #             best_score = score
    #             best_labels = labels
    # labels = [int(i) for i in best_labels]

    clusterer = hdbscan.HDBSCAN()
    labels = clusterer.fit_predict(X)
    
    if max(labels) < 0:
        return [0 for _ in dps]

    return labels



def get_inout_edges(sub_g,node, positive=True):
    in_egs = sub_g.in_edges(node,data=True)
    pos_in_egs = []
    neg_in_egs = []
    for u,v,e_attr in in_egs:
        if e_attr['to_orient'] == "+":
            pos_in_egs.append((u,v,e_attr))
        else:
            neg_in_egs.append((u,v,e_attr))

    out_egs = sub_g.out_edges(node,data=True)
    pos_out_egs = []
    neg_out_egs = []
    for u,v,e_attr in out_egs:
        if e_attr['from_orient'] == "+":
            pos_out_egs.append((u,v,e_attr))
        else:
            neg_out_egs.append((u,v,e_attr))
    
    if positive:
        return pos_in_egs,pos_out_egs
    else:
        return neg_in_egs,neg_out_egs


def get_simple_path(_sub_g, node):
    sub_g = _sub_g.copy()
    visited = set([node])
    st_in_egs,st_out_egs = get_inout_edges(sub_g,node)

    pt = []

    if len(st_in_egs) == 1 and len(st_out_egs) == 1:
        if st_in_egs[0][0] == node or st_out_egs[0][1] == node or st_in_egs[0][0] == st_out_egs[0][1]:
            # print('node is a loop')
            return [node]

        nt_pt = [node]
        in_egs,out_egs = st_in_egs.copy(),st_out_egs.copy()
        while True:
            next_node = out_egs[0][1]
            next_node_pos = out_egs[0][2]['to_orient'] == '+'
            in_egs,out_egs  = get_inout_edges(sub_g,next_node,next_node_pos)
            # print('next_node:',next_node,in_egs,out_egs)

            if len(in_egs) > 1:
                break
            if len(out_egs)==0 or len(out_egs) > 1:
                nt_pt.append(next_node)
                break
            if next_node in visited:
                break
            
            nt_pt.append(next_node)
            visited.add(next_node)
            

        ft_pt = []
        in_egs,out_egs = st_in_egs.copy(),st_out_egs.copy()
        # print('ft start',ft_pt,in_egs)
        while True:
            next_node = in_egs[0][0]
            next_node_pos = in_egs[0][2]['from_orient'] == '+'
            in_egs,out_egs  = get_inout_edges(sub_g,next_node,next_node_pos)

            if len(out_egs) > 1:
                break
            if len(in_egs)==0 or len(in_egs) > 1:
                ft_pt.append(next_node)
                break
            if next_node in visited:
                break
            
            ft_pt.append(next_node)
            visited.add(next_node)

        pt = ft_pt[::-1] + nt_pt
    return pt



def MaxWeightMatching(namelist,emblist,contig_markers,bin_n = -1):

    N = len(namelist)

    i_idx, j_idx = np.triu_indices(N, k=1)
    dists = np.linalg.norm(emblist[i_idx] - emblist[j_idx], axis=1)

    edgelist =  [
        (namelist[i], namelist[j], float(w))
        for i, j, w in zip(i_idx, j_idx,dists)
    ]
    wei_g = nx.Graph()
    wei_g.add_weighted_edges_from(edgelist)
    
    sorted_edges = sorted(edgelist,key=lambda x:x[2])
    uf = nx.utils.UnionFind()


    bins_partition_markers_set = {contig:set(contig_markers[contig]) if contig in contig_markers else set() for contig in namelist}
    __ = [uf[contig] for contig in namelist]

    for u,v,w in sorted_edges:
        conflict = False
        if u in contig_markers and v in contig_markers:
            conflict = len(set(contig_markers[u]) & set(contig_markers[v])) > 0
        if (not conflict) and len(bins_partition_markers_set[uf[u]] & bins_partition_markers_set[uf[v]]) == 0:
            marker_uni = bins_partition_markers_set[uf[u]] | bins_partition_markers_set[uf[v]]
            uf.union(u,v)
            uv_root = uf[u]
            bins_partition_markers_set[uv_root] = marker_uni

            bins = [list(s) for s in uf.to_sets()]
            if len(bins) == bin_n:
                ctg_bins = {c: i for i, b in enumerate(bins) for c in b}
                return bins, ctg_bins

    conf_list = [1,3,6,10,15,20]  
    lq_num = 60
    for _c in conf_list: 
        
        bins_dt =  { uf[list(i)[0]]:list(i) for i in list(uf.to_sets()) }
        if bin_n > 0:
            lqbins =  list( bins_dt.keys())
        else :
            lqbins = [i for i in bins_dt.keys() if len(bins_partition_markers_set[i])<lq_num]

        # print(f"low quality bins number :{len(lqbins)}")
        lq_egs = []
        for _i in range(len(lqbins)):
            for _j in range(_i +1 , len(lqbins)):
                wei_sum = 0
                src_l = bins_dt[lqbins[_i]]
                tgt_l = bins_dt[lqbins[_j]]
                for _s in src_l:
                    for _t in tgt_l:
                        wei_sum+= wei_g.get_edge_data(_s, _t, default={}).get('weight', 0)
                wei_sum = wei_sum/ (len(src_l)*len(tgt_l))
                lq_egs.append((lqbins[_i],lqbins[_j],wei_sum))
        
        sorted_edges = sorted(lq_egs,key=lambda x:x[2])
        for u,v,w in sorted_edges:
            conflict = False
            if u in contig_markers and v in contig_markers:
                conflict = len(set(contig_markers[u]) & set(contig_markers[v])) > _c
            if (not conflict) and len(bins_partition_markers_set[uf[u]] & bins_partition_markers_set[uf[v]]) <= _c:
                marker_uni = bins_partition_markers_set[uf[u]] | bins_partition_markers_set[uf[v]]
                uf.union(u,v)
                uv_root = uf[u]
                bins_partition_markers_set[uv_root] = marker_uni

                bins = [list(s) for s in uf.to_sets()]
                if len(bins) == bin_n:
                    ctg_bins = {c: i for i, b in enumerate(bins) for c in b}
                    return bins, ctg_bins
                
    bins = [list(s) for s in uf.to_sets()]
    ctg_bins = {c: i for i, b in enumerate(bins) for c in b}
    return bins, ctg_bins




def run_leiden(namelist: List[str],
               emblist,
               length_weight: List[float], 
               bandwidth: float = 0.1,
               initial_list: Optional[List[Union[int, None]]] = None,
               is_membership_fixed: Optional[bool] = None,
               resolution_parameter: float = 1.0):
    """
    Run Leiden community detection algorithm.

    :param namelist: A list of contig names.
    :param emblist: Array of embeddings.
    :param length_weight: List of length weights.
    :param bandwidth: Bandwidth parameter for distance weighting (default: 0.1).
    :param initial_list: Initial membership list (default: None).
    :param is_membership_fixed: Whether membership is fixed (default: None).
    :param resolution_parameter: Resolution parameter (default: 1.0).

    :return: Tuple of (bins, ctg_bins) where bins is a list of contig groups
             and ctg_bins is a dict mapping contigs to bin indices.
    """

    vcount = len(namelist)

    i_idx, j_idx = np.triu_indices(vcount, k=1)
    edgelist = list(zip(i_idx, j_idx))

    dists = np.linalg.norm(emblist[i_idx] - emblist[j_idx], axis=1)
    wei = np.exp(-dists / bandwidth)


    g = iGraph(vcount, edgelist)

    res = leidenalg.RBERVertexPartition(g,
                                        weights=wei, initial_membership = initial_list,
                                        resolution_parameter = resolution_parameter,node_sizes=length_weight)

    optimiser = leidenalg.Optimiser()
    optimiser.optimise_partition(res, is_membership_fixed=is_membership_fixed,n_iterations=-1)

    part = list(res)
    bins = []
    for ci in range(len(part)):
        bins.append([namelist[id] for id in part[ci]])
    ctg_bins = {c: i for i, b in enumerate(bins) for c in b}

    return bins, ctg_bins
