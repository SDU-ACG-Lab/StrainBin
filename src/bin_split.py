import numpy as np
import hdbscan
from sklearn.cluster import dbscan
from collections import defaultdict
from itertools import combinations


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from .bin_io import networkx_graph_to_gfa
from .bin_utils import has_approx_sum, in_bin_cluster, get_inout_edges, get_simple_path, MaxWeightMatching, run_leiden

def dpsum(l1,l2):
    return [a + b for a, b in zip(l1, l2)]
def dpcompare(l1,l2):
    pass

def graph_split(init_g,egs_list, extra = False ):

    eg_cluster_dt = {}
    ex_add_node = []
    ex_add_length = []
    if extra:
        # 添加额外的节点 测试用, 有的数据及效果好，有的效果不好。
        connetcted = set(egs_list)
        
        for eg in egs_list:
            connetcted.update(init_g.successors(eg))
            connetcted.update(init_g.predecessors(eg))
        for _ex_n in list(set(connetcted) - set(egs_list)):
            pos_in_egs,pos_out_egs = get_inout_edges(init_g,_ex_n)
            suc_neibor = set() 
            pre_neibor = set()
            for u,v,e_attr in pos_in_egs:
                pre_neibor.add(u)
            for u,v,e_attr in pos_out_egs:
                suc_neibor.add(v)

            if len(suc_neibor)>0 and len(pre_neibor)>0 and  len(suc_neibor & set(egs_list)) == len(suc_neibor) and len(pre_neibor & set(egs_list)) == len(pre_neibor):
                ex_add_node.append(_ex_n)
                ex_add_length.append(init_g.nodes[_ex_n]['length'])
                # if init_g.nodes[_ex_n]['length'] <  init_g.nodes[list(pre_neibor)[0]]['length'] or init_g.nodes[_ex_n]['length'] <  init_g.nodes[list(suc_neibor)[0]]['length'] :
                    # 避免因为图上短重复片段引入错误的contig

    egs_list = egs_list + ex_add_node
    # 按覆盖深度从大到小排序 egs_list，高深度的 contig 优先参与聚类
    egs_list.sort(key=lambda x: init_g.nodes[x]['dp'])

    sub_g = init_g.subgraph(egs_list).copy()
    if sub_g.number_of_edges() <=1 :
        return [egs_list]

    dps_list = [init_g.nodes[eg]['dp'] for eg in egs_list]
    dps_dt = {egs_list[i]:dps_list[i] for i in range(len(egs_list))}
    lens_list = [init_g.nodes[eg]['length'] for eg in egs_list]
    lens_dt = {egs_list[i]:lens_list[i] for i in range(len(egs_list))}

    

    
    dead_egs = set() #悬挂节点
    ai,bi,ci = 0,1,2
    for eg in egs_list:
        pos_in_egs,pos_out_egs = get_inout_edges(sub_g,eg)
        # print(f"eg {eg}, in edges: {pos_in_egs}, out edges: {pos_out_egs}")
        if len(pos_in_egs)==2 and len(pos_out_egs)==2:
            u1,u2 = pos_in_egs[0][0],pos_in_egs[1][0]
            v1,v2 = pos_out_egs[0][1],pos_out_egs[1][1]
            if u1 == v1 or u1 ==v2 or u2 == v1 or u2 == v2 or v1==v2 or u1==u2:
                continue

            
            great_in  = dps_dt[eg]> max(dps_dt[u1],dps_dt[u2],(dps_dt[u1]+dps_dt[u2])*0.75 )
            great_out  = dps_dt[eg]> max(dps_dt[v1],dps_dt[v2],(dps_dt[v1]+dps_dt[v2])*0.75 )
            great_mid = (dps_dt[v1]+dps_dt[v2]) > dps_dt[eg] *0.75 and (dps_dt[u1]+dps_dt[u2]) > dps_dt[eg]  *0.75

            # print('resign',eg,dps_dt[eg])
            # print(dps_dt[u1],dps_dt[u2],dps_dt[v1],dps_dt[v2],great_in,great_out,great_mid)
            
            
            if great_in and great_out and great_mid:
                u1_in,u1_out = get_inout_edges(sub_g, u1)
                u1_dead = (len(u1_in) + len(u1_out)) == 1
                u1_simple = (len(u1_in) <=1)  and (len(u1_out) <= 1)

                u2_in,u2_out = get_inout_edges(sub_g, u2)
                u2_dead = (len(u2_in) + len(u2_out)) == 1
                u2_simple = (len(u2_in) <= 1) and (len(u2_out) <= 1)

                v1_in,v1_out = get_inout_edges(sub_g, v1)
                v1_dead = (len(v1_in) + len(v1_out)) == 1
                v1_simple = (len(v1_in) <= 1) and (len(v1_out) <= 1)

                v2_in,v2_out = get_inout_edges(sub_g, v2)
                v2_dead = (len(v2_in) + len(v2_out)) == 1
                v2_simple = (len(v2_in) <= 1) and (len(v2_out) <= 1)

                # print('dead',u1_dead,u2_dead,v1_dead,v2_dead)

                # if u1_simple and u2_simple and v1_simple and v2_simple:

                if (u1_dead!=u2_dead) and (v1_dead!=v2_dead) :
                    eg_cluster_dt[eg] = ai  
                elif u1_dead!=u2_dead:
                    # print('dead eg',v1,v2)
                    eg_cluster_dt[eg] = ai  
                    if dps_dt[v1] > dps_dt[v2]:
                        eg_cluster_dt[v1] = bi 
                        eg_cluster_dt[v2] = ci
                    else:
                        eg_cluster_dt[v1] = ci 
                        eg_cluster_dt[v2] = bi
                    
                    if u1_dead:
                        dead_egs.add(u1)
                    if u2_dead:
                        dead_egs.add(u2)
                    
                    # if u1_dead:
                    #     print('dead eg', u1)
                    #     if dps_dt[u1] > dps_dt[u2]:
                    #         eg_cluster_dt[u1] = bi 
                    #     else:
                    #         eg_cluster_dt[u1] = ci 
                    # if u2_dead:
                    #     print('dead eg', u2)
                    #     if dps_dt[u1] > dps_dt[u2]:
                    #         eg_cluster_dt[u2] = ci 
                    #     else:
                    #         eg_cluster_dt[u2] = bi 

                elif v1_dead!=v2_dead:
                    # print('dead eg',u1,u2)
                    eg_cluster_dt[eg] = ai  
                    if dps_dt[u1] > dps_dt[u2]:
                        eg_cluster_dt[u1] = bi 
                        eg_cluster_dt[u2] = ci 
                    else:
                        eg_cluster_dt[u1] = ci 
                        eg_cluster_dt[u2] = bi

                    if v1_dead:
                        dead_egs.add(v1)
                    if v2_dead:
                        dead_egs.add(v2)
                        
                    # if v1_dead:
                    #     print('dead eg', v1)
                    #     if dps_dt[v1] > dps_dt[v2]:
                    #         eg_cluster_dt[v1] = bi 
                    #     else:
                    #         eg_cluster_dt[v1] = ci 
                    # if v2_dead:
                    #     print('dead eg', v2)
                    #     if dps_dt[v1] > dps_dt[v2]:
                    #         eg_cluster_dt[v2] = ci 
                    #     else:
                    #         eg_cluster_dt[v2] = bi 
                    
                else:
                    eg_cluster_dt[eg] = ai  

                    if (eg_cluster_dt.get(u1) == ai) and  (eg_cluster_dt.get(u2)==ai):
                        pass
                        # eg_cluster_dt[v1] = ai 
                        # eg_cluster_dt[v2] = ai 
                    elif (eg_cluster_dt.get(v1) == ai) and  (eg_cluster_dt.get(v2)==ai):
                        pass
                        # eg_cluster_dt[u1] = ai 
                        # eg_cluster_dt[u2] = ai 

                    else:
                        if dps_dt[u1] > dps_dt[u2]:
                            eg_cluster_dt[u1] = bi 
                            eg_cluster_dt[u2] = ci 
                        else:
                            eg_cluster_dt[u1] = ci 
                            eg_cluster_dt[u2] = bi 
                        if dps_dt[v1] > dps_dt[v2]:
                            eg_cluster_dt[v1] = bi 
                            eg_cluster_dt[v2] = ci
                        else:
                            eg_cluster_dt[v1] = ci 
                            eg_cluster_dt[v2] = bi

            elif great_in and  great_mid:
                u1_in,u1_out = get_inout_edges(sub_g, u1)
                u1_dead = (len(u1_in) + len(u1_out)) == 1
                u1_simple = (len(u1_in) <=1)  and (len(u1_out) <= 1)

                u2_in,u2_out = get_inout_edges(sub_g, u2)
                u2_dead = (len(u2_in) + len(u2_out)) == 1
                u2_simple = (len(u2_in) <= 1) and (len(u2_out) <= 1)

                if u1_dead!=u2_dead:
                    # print('dead eg',v1,v2)
                    eg_cluster_dt[eg] = ai  
                    if u1_dead:
                        dead_egs.add(u1)
                    if u2_dead:
                        dead_egs.add(u2)
                    
                else:
                    eg_cluster_dt[eg] = ai  

                    if (eg_cluster_dt.get(u1) == ai) and  (eg_cluster_dt.get(u2)==ai):
                        pass
                    else:
                        if dps_dt[u1] > dps_dt[u2]:
                            eg_cluster_dt[u1] = bi 
                            eg_cluster_dt[u2] = ci 
                        else:
                            eg_cluster_dt[u1] = ci 
                            eg_cluster_dt[u2] = bi 
                        
            elif great_out and great_mid:

                v1_in,v1_out = get_inout_edges(sub_g, v1)
                v1_dead = (len(v1_in) + len(v1_out)) == 1
                v1_simple = (len(v1_in) <= 1) and (len(v1_out) <= 1)

                v2_in,v2_out = get_inout_edges(sub_g, v2)
                v2_dead = (len(v2_in) + len(v2_out)) == 1
                v2_simple = (len(v2_in) <= 1) and (len(v2_out) <= 1)

                if v1_dead!=v2_dead:
                    eg_cluster_dt[eg] = ai  
                    if v1_dead:
                        dead_egs.add(v1)
                    if v2_dead:
                        dead_egs.add(v2)
                        
                else:
                    eg_cluster_dt[eg] = ai  

                    if (eg_cluster_dt.get(v1) == ai) and  (eg_cluster_dt.get(v2)==ai):
                        pass
                    else:
                        
                        if dps_dt[v1] > dps_dt[v2]:
                            eg_cluster_dt[v1] = bi 
                            eg_cluster_dt[v2] = ci
                        else:
                            eg_cluster_dt[v1] = ci 
                            eg_cluster_dt[v2] = bi
                            

        if len(pos_in_egs)==2 and len(pos_out_egs)<=1:
            u1,u2 = pos_in_egs[0][0],pos_in_egs[1][0]
            if u1 == u2:
                continue
            great_in  = dps_dt[eg]> max(dps_dt[u1],dps_dt[u2],(dps_dt[u1]+dps_dt[u2])*0.75 )
            great_mid = (dps_dt[u1]+dps_dt[u2]) >  dps_dt[eg] * 0.75

            # print('resign21',eg,dps_dt[eg])
            # print(dps_dt[u1],dps_dt[u2],great_in,great_mid)

            if great_in and great_mid:
                

                eg_cluster_dt[eg] = ai  
                if (eg_cluster_dt.get(u1) == ai) and  (eg_cluster_dt.get(u2)==ai):
                    pass
                    # if  len(pos_out_egs)==1:
                    #     v1 = pos_out_egs[0][1]
                    #     eg_cluster_dt[v1] = ai  
                else:
                    if dps_dt[u1] > dps_dt[u2]:
                        eg_cluster_dt[u1] = bi 
                        eg_cluster_dt[u2] = ci 
                    else:
                        eg_cluster_dt[u1] = ci 
                        eg_cluster_dt[u2] = bi 

                if  len(pos_out_egs)==1 :
                    v1 = pos_out_egs[0][1]
                    if dps_dt[v1] > dps_dt[eg] * 0.5:
                        eg_cluster_dt[v1] = ai  

        if len(pos_in_egs)<=1 and len(pos_out_egs)==2:
            v1,v2 = pos_out_egs[0][1],pos_out_egs[1][1]
            if v1 == v2:
                continue
            great_out  = dps_dt[eg]> max(dps_dt[v1],dps_dt[v2],(dps_dt[v1]+dps_dt[v2])*0.75 )
            great_mid = (dps_dt[v1]+dps_dt[v2]) >  dps_dt[eg] * 0.75
            # print('resign12',eg,dps_dt[eg])
            # print(dps_dt[v1],dps_dt[v2],great_out,great_mid)

            if great_out and great_mid:

                eg_cluster_dt[eg] = ai  
                if (eg_cluster_dt.get(v1) == ai) and  (eg_cluster_dt.get(v2)==ai):
                    pass
                    # if  len(pos_in_egs)==1:
                    #     u1 = pos_in_egs[0][0]
                    #     eg_cluster_dt[u1] = ai  
                else:
                    if dps_dt[v1] > dps_dt[v2]:
                        eg_cluster_dt[v1] = bi 
                        eg_cluster_dt[v2] = ci 
                    else:
                        eg_cluster_dt[v1] = ci 
                        eg_cluster_dt[v2] = bi 
                    
                if  len(pos_in_egs)==1:
                    u1 = pos_in_egs[0][0]
                    if dps_dt[u1] > dps_dt[eg] * 0.5:
                        eg_cluster_dt[u1] = ai  


    # print(eg_cluster_dt)
    # print('---')
    
    for eg in egs_list:
        if eg not in eg_cluster_dt:
            pt = get_simple_path(sub_g, eg)
            if len(pt) > 1:
                # temp1,temp2 = get_inout_edges(sub_g,eg)
                # print('flagsimplenode',eg,pt)
                if pt[0] in eg_cluster_dt and pt[-1] in eg_cluster_dt:
                    if eg_cluster_dt[pt[0]] == eg_cluster_dt[pt[-1]]:
                        for _nd in pt:
                            eg_cluster_dt[_nd] = eg_cluster_dt[pt[0]]
                elif pt[0] in eg_cluster_dt:
                    for _nd in pt:
                        eg_cluster_dt[_nd] = eg_cluster_dt[pt[0]]
                elif pt[-1] in eg_cluster_dt:
                    for _nd in pt:
                        eg_cluster_dt[_nd] = eg_cluster_dt[pt[-1]]
    # print(eg_cluster_dt)                    
    if len(eg_cluster_dt.keys()) > len(egs_list) * 0.5 and len(set(eg_cluster_dt.values())) == 3:
        
        cluster_dp_list = [[], [], []]
        cluster_len_list = [[], [], []]
        cluster_dplen_list = [[], [], []]
        cluster_eg_list = [[], [], []]  # 添加：按簇分组的 contig 列表
        for k, v in eg_cluster_dt.items():
            cluster_dp_list[v].append(dps_dt[k])
            cluster_len_list[v].append(lens_dt[k])
            cluster_dplen_list[v].append(dps_dt[k]*lens_dt[k])
            cluster_eg_list[v].append(k)
        # ave_dp = [float(sum(cluster_dplen_list[i]))/sum(cluster_len_list[i]) if len(cluster_dplen_list[i])>0 else 0 for i in range(len(cluster_dplen_list))]
        ave_dp = [float(sum(cluster_dp_list[i]))/len(cluster_len_list[i]) if len(cluster_dp_list[i])>0 else 0 for i in range(len(cluster_dp_list))]

        # print('cluster_dp_list:',cluster_dp_list)
        # print('cluster_eg_list:',cluster_eg_list)
        # print('ave_dp:',ave_dp)

        # 噪声过滤：两轮筛选，先去除粗噪声得到更可靠的均值，再基于该均值重新过滤
        # 第一轮：用原始均值 a 去除偏差 > 0.5*a 的元素
        # 第二轮：对剩余元素计算均值 b，再用初始全集重新判断偏差 > 0.5*b 的元素
        noise_removed = []
        dp_deviation_threshold = 0.5  # 允许的深度偏差比例
        for i in range(3):
            if len(cluster_eg_list[i]) == 0:
                continue
            # 第一轮：用 ave_dp (均值 a) 过滤
            first_keep = []
            for j, eg in enumerate(cluster_eg_list[i]):
                dp = cluster_dp_list[i][j]
                if ave_dp[i] > 0 and abs(dp - ave_dp[i]) / ave_dp[i] > dp_deviation_threshold:
                    continue  # 第一轮仅标记待排除，但先不操作原始列表
                first_keep.append(j)
            # 如果没有元素通过第一轮，全部视为噪声
            if len(first_keep) == 0:
                noise_removed.extend(cluster_eg_list[i])
                cluster_eg_list[i] = []
                cluster_dp_list[i] = []
                cluster_len_list[i] = []
                cluster_dplen_list[i] = []
                continue
            # 计算均值 b：基于第一轮保留的元素
            first_dp = [cluster_dp_list[i][j] for j in first_keep]
            mean_b = float(sum(first_dp)) / len(first_dp)
            # 第二轮：用均值 b 从初始全集重新判断
            keep_idx = []
            for j, eg in enumerate(cluster_eg_list[i]):
                dp = cluster_dp_list[i][j]
                if mean_b > 0 and abs(dp - mean_b) / mean_b > dp_deviation_threshold:
                    noise_removed.append(eg)
                    continue
                keep_idx.append(j)
            cluster_eg_list[i] = [cluster_eg_list[i][j] for j in keep_idx]
            cluster_dp_list[i] = [cluster_dp_list[i][j] for j in keep_idx]
            cluster_len_list[i] = [cluster_len_list[i][j] for j in keep_idx]
            cluster_dplen_list[i] = [cluster_dplen_list[i][j] for j in keep_idx]

        # 从 eg_cluster_dt 中移除噪声点，使它们进入未分配池进行后续重新分配
        for eg in noise_removed:
            eg_cluster_dt.pop(eg, None)

        # 重新计算 ave_dp
        ave_dp = [float(sum(cluster_dplen_list[i]))/sum(cluster_len_list[i]) if len(cluster_dplen_list[i])>0 else 0 for i in range(len(cluster_dplen_list))]

        # print('cluster_dp_list:',cluster_dp_list)
        # print('cluster_eg_list:',cluster_eg_list)
        # print('ave_dp:',ave_dp)

        for _eg in set(egs_list) - set(eg_cluster_dt.keys()):
            if _eg in dead_egs:
                continue
            min_cls = min((ai,bi,ci),key=lambda x: abs(dps_dt[_eg] - ave_dp[x]))
            if len(cluster_dp_list[min_cls]) > 0:
                if dps_dt[_eg] > min(cluster_dp_list[min_cls]) and dps_dt[_eg] < max(cluster_dp_list[min_cls]):
                    eg_cluster_dt[_eg] = min_cls
                    cluster_eg_list[min_cls].append(_eg)

        len_list = [sum(init_g.nodes[eg]['length'] for eg in _cls) for _cls in cluster_eg_list]
        # 补入节点后，按最终成员重新计算长度加权深度，用于最终深度判定。
        ave_dp = [
            sum(dps_dt[eg] * lens_dt[eg] for eg in cluster) / total_len
            if total_len > 0 else 0
            for cluster, total_len in zip(cluster_eg_list, len_list)
        ]

        len_a = len_list[ai]
        len_b = len_list[bi]
        len_c = len_list[ci]

        have_abc,abc,abci = has_approx_sum(ave_dp, 0.25)
        if not have_abc:
            # print('flag 0')
            return [egs_list]

        if len_a + len_b + len_c < 0.7 * sum(len_list):
            # print('flag 1')
            return [egs_list]
        # if len (cluster_eg_list[ai] ) +  len (cluster_eg_list[bi]) + len(cluster_eg_list[ci]) < 0.5 * len(egs_list):
        #     print('flag 1')
        #     print(len (cluster_eg_list[ai] ) +  len (cluster_eg_list[bi]) + len(cluster_eg_list[ci]),  len(egs_list))
        #     return [egs_list]

        if len_b < len_a * 0.1 or len_c < len_a * 0.1 or len_b < len_c * 0.5 or len_c < len_b * 0.5:
            # print('flag 2')
            return [egs_list]

        return [cluster_eg_list[ai] + cluster_eg_list[bi], cluster_eg_list[ai] + cluster_eg_list[ci]]
    # print('flag 3')
    return [egs_list]

def cov_split(init_g,egs_list):
    dps_list = [init_g.nodes[eg]['dp'] for eg in egs_list]
    dps_dt = {egs_list[i]:dps_list[i] for i in range(len(egs_list))}
    lens_list = [init_g.nodes[eg]['length'] for eg in egs_list]
    lens_dt = {egs_list[i]:lens_list[i] for i in range(len(egs_list))}

    labels = in_bin_cluster(dps_list)
    eg_cluster_dt = {}
    for i in range(len(labels)):
        if labels[i] < 0:
            continue
        eg_cluster_dt[egs_list[i]] = labels[i]

    # 空字典保护
    if len(eg_cluster_dt) == 0:
        return [egs_list]

    cluster_eg_list = [[] for _ in range(max(eg_cluster_dt.values()) + 1)]
    cluster_dp_list = [[] for _ in range(max(eg_cluster_dt.values()) + 1)]
    cluster_len_list = [[] for _ in range(max(eg_cluster_dt.values()) + 1)]
    cluster_dplen_list = [[] for _ in range(max(eg_cluster_dt.values()) + 1)]
    for k,v in eg_cluster_dt.items():
        cluster_eg_list[v].append(k)
        cluster_dp_list[v].append(dps_dt[k])
        cluster_len_list[v].append(lens_dt[k])
        cluster_dplen_list[v].append(dps_dt[k]*lens_dt[k])
    # ave_dp = [float(sum(cluster_dp_list[i]))/len(cluster_dp_list[i]) if len(cluster_dp_list[i])>0 else 0 for i in range(len(cluster_dp_list))]
    len_list = [sum(lengths) for lengths in cluster_len_list]
    ave_dp = [
        sum(depth_lengths) / total_len if total_len > 0 else 0
        for depth_lengths, total_len in zip(cluster_dplen_list, len_list)
    ]
    # print('ave_dp',ave_dp,)
    # print(cluster_dp_list)

    # 遍历所有共享簇及其两组特异簇，保留通过全部条件且总长度最大的三元组。
    cluster_ids = [i for i, length in enumerate(len_list) if length > 0]
    total_len = sum(len_list)
    best_triplet = None
    best_length = -1
    for ai in cluster_ids:
        for bi, ci in combinations((i for i in cluster_ids if i != ai), 2):
            if not abs(ave_dp[ai] - (ave_dp[bi] + ave_dp[ci])) < 0.25 * ave_dp[ai]:
                continue

            len_a, len_b, len_c = len_list[ai], len_list[bi], len_list[ci]
            combined_len = len_a + len_b + len_c
            if combined_len < 0.8 * total_len:
                continue
            if len_b < len_a * 0.1 or len_c < len_a * 0.1 or len_b < len_c * 0.5 or len_c < len_b * 0.5:
                continue

            if combined_len > best_length:
                best_triplet = (ai, bi, ci)
                best_length = combined_len

    if best_triplet is not None:
        ai, bi, ci = best_triplet
        return [cluster_eg_list[ai]+cluster_eg_list[bi], cluster_eg_list[ai]+cluster_eg_list[ci]]

    return [egs_list]


def get_best_bin(results, contig_to_marker, namelist, processed_ctg, contig_len, minfasta,beta=1):

    for max_contamination in [0.1, 0.2, 0.3, 0.4, 0.5, 1]:
        max_F1 = 0
        weight_of_max = 1e9
        max_bin = None

        for res_labels in results:
            res = defaultdict(list)
            for label, name in zip(res_labels, namelist):
                if label != -1 and name not in processed_ctg:
                    res[label].append(name)
            for bin_contig in res.values():
                cur_weight = sum(contig_len[contig] for contig in bin_contig)
                if cur_weight < minfasta:
                    continue
                marker_list = []
                for contig in bin_contig :
                    if  contig in contig_to_marker:
                        marker_list.extend(contig_to_marker[contig])
                if len(marker_list) == 0:
                    continue
                recall = len(set(marker_list)) / 107
                contamination = (len(marker_list) - len(set(marker_list))) / len(marker_list)
                F1 = (1 + beta * beta) * recall * (1 - contamination) / (
                            recall + (1 - contamination) * beta * beta)

                # precison 计算更新
                # um = len(marker_list)
                # sm = len(set(marker_list))
                # xm = um/sm
                # k=2
                # prec = 1/xm if xm < 1.5 else 1/xm * (1.5/xm)**k
                # F1 = (1 + beta * beta) * recall * prec / (recall + prec * beta * beta)
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
        
def matching_split(init_g, ctgs, ctgs_emb, contig_marker):

    bin_contig = list(ctgs)
    ctg_rev = {c: i for i, c in enumerate(ctgs)}
    # 深拷贝，避免后续添加虚拟 marker 时污染原始 contig_marker
    bin_contig_marker = {c: list(contig_marker[c]) for c in bin_contig if c in contig_marker}

    lens_list = [init_g.nodes[eg].get('length', 0) for eg in bin_contig]
    lens_dt = {bin_contig[i]: lens_list[i] for i in range(len(bin_contig))}
    sum_len = sum(lens_dt.values())

    # 获取覆盖深度
    dps_dt = {c: init_g.nodes[c].get('dp', 0) for c in bin_contig}

    # 计算 contamination
    marker_list = []
    for contig in bin_contig:
        if contig in bin_contig_marker:
            marker_list.extend(bin_contig_marker[contig])
    if len(marker_list) == 0:
        # print(f'marker_list == 0')
        return [bin_contig]
    
    contamination = (len(marker_list) - len(set(marker_list))) / len(marker_list)
    
    if contamination < 0.1:
        # print(f'scg contamination < 0.1')
        return [bin_contig]

    # 构建 bin_marker_contig
    bin_marker_contig = {}
    for c, ms in bin_contig_marker.items():
        for m in ms:
            if m not in bin_marker_contig:
                bin_marker_contig[m] = [c]
            else:
                bin_marker_contig[m].append(c)

    isolated_contigs = []
    for c in bin_contig:
        if c not in bin_contig_marker:
            isolated_contigs.append(c)
        else:
            has_conflict = False
            for m in bin_contig_marker[c]:
                if m in bin_marker_contig and len(bin_marker_contig[m]) > 1:
                    has_conflict = True
                    break
            if not has_conflict:
                isolated_contigs.append(c)
    isolated_contigs_set = set(isolated_contigs)

    # 计算 conf_cc_len 确定 num_bins
    # len_list = [len(bin_marker_contig[m]) for m in bin_marker_contig]
    # if len(len_list) == 0:
    #     return [bin_contig]
    
    # conf_cc = [set() for _ in range(max(len_list) + 1)]
    # for mk in bin_marker_contig:
    #     set_c = set(bin_marker_contig[mk])
    #     conf_cc[len(bin_marker_contig[mk])] |= set_c
    
    # for i in range(len(conf_cc)):
    #     for j in range(i + 1, len(conf_cc)):
    #         conf_cc[i] -= conf_cc[j]
    
    # conf_cc_len = [sum([lens_dt.get(ci, 0) for ci in conf_cc[i]]) for i in range(len(conf_cc))]

    # num_bins = max(
    #     (i for i, v in enumerate(conf_cc_len) if v > sum_len * 0.2),
    #     default=1
    # )
    
    conf_num_marker = [0]*max([len(cl) for cl in bin_marker_contig.values()])
    for k,v in bin_marker_contig.items():
        conf_num_marker[len(v)-1] += 1
    num_bins = max(
            (i+1 for i, v in enumerate(conf_num_marker) if v > len(bin_marker_contig.keys()) * 0.25),
            default=1
        )
    # 取两者最大值
    # num_bins = max(max_len_idx, threshold_idx)

    if num_bins == 1:
        # print(f'num_bins=1, 无需分组')
        return [bin_contig]

    # print(f'dedup_bin_matching1: contamination={contamination:.2%}, num_bins={num_bins}')

    # ========== 根据 num_bins 决定分组策略 ==========
    if num_bins == 2:
        # 只有 num_bins == 2 时，才对 isolated contigs 做特殊处理
        # 先对非孤立 contig 进行分组，再对孤立 contig 按 coverage 策略处理
        non_isolated_contigs = [c for c in bin_contig if c not in isolated_contigs_set]
        
        if len(non_isolated_contigs) == 0:
            # 所有 contig 都是孤立的，无法分组
            return [bin_contig]
        
        # 构建非孤立 contig 的索引映射
        non_iso_indices = [ctg_rev[c] for c in non_isolated_contigs]
        non_iso_emb = ctgs_emb[non_iso_indices]
        non_iso_marker = {c: bin_contig_marker[c] for c in non_isolated_contigs if c in bin_contig_marker}
        
        all_bins, all_ctg_bins = MaxWeightMatching(non_isolated_contigs, non_iso_emb, non_iso_marker, num_bins)
        
        # 过滤空的 bin
        final_bins = [b for b in all_bins if len(b) > 0]
        
        if len(final_bins) == 0:
            return [bin_contig]
        
        # ========== 处理孤立 contig（仅 num_bins == 2）==========
        if len(final_bins) == 2 and len(isolated_contigs) > 0:
            # 计算两个 bin 内根据长度的加权平均覆盖深度 a, b
            def calc_weighted_avg_dp(bin_contigs):
                total_len = sum([lens_dt.get(c, 0) for c in bin_contigs])
                if total_len == 0:
                    return 0
                weighted_dp = sum([dps_dt.get(c, 0) * lens_dt.get(c, 0) for c in bin_contigs])
                return weighted_dp / total_len
            
            a = calc_weighted_avg_dp(final_bins[0])
            b = calc_weighted_avg_dp(final_bins[1])
            dp_threshold = (a + b) * 0.75
            
            # print(f'孤立 contig 处理: bin0 加权平均深度={a:.2f}, bin1 加权平均深度={b:.2f}, 阈值={(a+b)*0.75:.2f}')
            
            for iso_ctg in isolated_contigs:
                iso_dp = dps_dt.get(iso_ctg, 0)
                
                if iso_dp >= dp_threshold:
                    # 覆盖深度超过阈值，添加到两个 bin 中
                    final_bins[0].append(iso_ctg)
                    final_bins[1].append(iso_ctg)
                    # print(f'  {iso_ctg}: dp={iso_dp:.2f} >= {dp_threshold:.2f}, 添加到两个 bin')
                else:
                    # 只添加到距离近的一个 bin
                    iso_idx = ctg_rev[iso_ctg]
                    
                    # 计算与每个 bin 的平均距离（dist_mat 是距离矩阵）
                    bin0_indices = [ctg_rev[c] for c in final_bins[0] if c in ctg_rev and c != iso_ctg]
                    bin1_indices = [ctg_rev[c] for c in final_bins[1] if c in ctg_rev and c != iso_ctg]
                    
                    avg_dist_0 = np.linalg.norm(np.mean(ctgs_emb[bin0_indices], axis=0) - ctgs_emb[iso_idx]) if len(bin0_indices) > 0 else 1e9
                    avg_dist_1 =np.linalg.norm(np.mean(ctgs_emb[bin1_indices] ,axis=0) - ctgs_emb[iso_idx])  if len(bin1_indices) > 0 else 1e9
                    
                    # 选择距离更小的 bin
                    if avg_dist_0 <= avg_dist_1:
                        final_bins[0].append(iso_ctg)
                        # print(f'  {iso_ctg}: dp={iso_dp:.2f} < {dp_threshold:.2f}, 添加到 bin0 (dist={avg_dist_0:.4f})')
                    else:
                        final_bins[1].append(iso_ctg)
                        # print(f'  {iso_ctg}: dp={iso_dp:.2f} < {dp_threshold:.2f}, 添加到 bin1 (dist={avg_dist_1:.4f})')
    else:
        # num_bins != 2，使用 DBSCAN 聚类方法选择最佳分 bin 结果
        # print(f'使用 DBSCAN 聚类方法，期望分成 {num_bins} 个 bin')
        

        # pre_bins, all_ctg_bins = MaxWeightMatching(bin_contig, ctgs_emb, contig_marker)
        # initial_list = []
        # is_membership_fixed = []
        # index = len(pre_bins) + 1
        # for contig in bin_contig:
        #     if contig in all_ctg_bins:
        #         initial_list.append(all_ctg_bins[contig])
        #         is_membership_fixed.append(True)
        #     else:
        #         initial_list.append(index)
        #         is_membership_fixed.append(False)
        #         index += 1


        # length_weight = lens_list
        # bandwidth = 0.1 # [0.05, 0.1,0.15, 0.2,0.3]
        # resolution_parameter = 1 
        
        # final_bins,__ = run_leiden(ctgs, ctgs_emb, length_weight, bandwidth, initial_list,is_membership_fixed, resolution_parameter )
        
        # if len(final_bins) == 0:
        #     # print('DBSCAN 未能找到任何有效的 bin，回退到不分组')
        #     return [bin_contig]
        dbscan_results = []
        processed_ctg = set()
        eps_list = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

        # bin_contig, ctgs_emb, contig_marker lens_dt
        N = len(bin_contig)

        i_idx, j_idx = np.triu_indices(N, k=1)
        dist_values = np.linalg.norm(ctgs_emb[i_idx] - ctgs_emb[j_idx], axis=1)
        # 构建完整的 N×N 距离矩阵（DBSCAN precomputed 模式需要）
        dist_matrix = np.zeros((N, N))
        dist_matrix[i_idx, j_idx] = dist_values
        dist_matrix[j_idx, i_idx] = dist_values  # 对称
        for eps_value in eps_list:
            _, labels = dbscan(dist_matrix,
                    eps=eps_value, min_samples=5, n_jobs=32, metric='precomputed',
                    sample_weight=lens_list)
            dbscan_results.append(labels.tolist())

        best_bins = []
        processed_ctg = set()
        minfasta = 200000

        remaining_len = sum(lens_dt[c] for c in bin_contig)

        while remaining_len >= minfasta:
            
            max_bin = get_best_bin(dbscan_results,
                                    bin_contig_marker,
                                    bin_contig,
                                    processed_ctg,
                                    lens_dt,
                                    minfasta,
                                    1)
            if not max_bin:
                break

            best_bins.append(max_bin)
            processed_ctg.update(max_bin)
            remaining_len -= sum([lens_dt[c] for c in max_bin])

        final_bins = best_bins
        
        if len(final_bins) == 0:
            # print('DBSCAN 未能找到任何有效的 bin，回退到不分组')
            return [bin_contig]

    return final_bins


def bin_split(init_g,ctgs,ctgs_emb,contig_marker, extra=False):
    
    egs_list = list(ctgs)

    if len(egs_list) < 4:
        return [egs_list], None
    
    # graph_res = graph_split(init_g,egs_list,extra)
    # if len(graph_res)>1:
    #     return graph_res, 'sg'
    
    cov_res  = cov_split(init_g,egs_list)
    if len(cov_res)>1:
        return cov_res, 'sc'
    
    # matching_split 已移至阶段 4 (contamination_check + re-DBSCAN)
    # 此处只保留图结构和覆盖度两级分裂策略
    
    # 所有分裂策略都未成功，返回原始列表
    return [egs_list], None
