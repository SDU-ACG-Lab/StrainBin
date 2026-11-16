import gfapy
import networkx as nx
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO   
import pandas as pd

def gfa_to_networkx_graph(gfa_path):
    # 1. 解析GFA
    gfa = gfapy.Gfa.from_file(gfa_path)
    graph = nx.MultiDiGraph()

    # 2. 加入节点
    for segment in gfa.segments:
        node_id = segment.name
        node_attr = {
            'sequence': segment.sequence, 
            'length' : len(segment.sequence),
            'dp' : segment.dp
        }
        
        
        graph.add_node(node_id, **node_attr)
    # 3. 加入边
    for link in gfa.edges:
        # edge: from_, to, from_orient, to_orient, overlap, ...
        from_id = link.from_segment.name
        to_id = link.to_segment.name
        edge_attr = {
            'from_orient': link.from_orient,
            # 'from_dp' : graph.nodes[from_id].get('dp'),
            'to_orient': link.to_orient,
            # 'to_dp' : graph.nodes[to_id].get('dp'),
            'overlap': str(link.overlap)
        }

        edge_in = False
        if graph.has_edge(from_id, to_id):
            for k,k_attr in graph[from_id][to_id].items():
                if k_attr.get('from_orient') == edge_attr['from_orient'] and k_attr.get('to_orient') == edge_attr['to_orient']:
                    edge_in = True
                    break
        if not edge_in:
            graph.add_edge(from_id, to_id, **edge_attr)
        
        edge_attr_rev = {
            'from_orient': '+' if link.to_orient == '-' else '-',
            # 'from_dp' : graph.nodes[to_id].get('dp'),
            'to_orient':  '+' if link.from_orient == '-' else '-',
            # 'to_dp' : graph.nodes[from_id].get('dp'),
            'overlap': str(link.overlap)
        }

        edge_in = False
        if graph.has_edge(to_id, from_id):
            for k,k_attr in graph[to_id][from_id].items():
                if k_attr.get('from_orient') == edge_attr_rev['from_orient'] and k_attr.get('to_orient') == edge_attr_rev['to_orient']:
                    edge_in = True
                    break
        if not edge_in:
            graph.add_edge(to_id, from_id, **edge_attr_rev)

    egs_ctgs = {} # {'edge_1',set(['contig_1','contig_2'])} edge 所在contig
    ctgs_egs = {} # {'contig_2',set(['edge_1','edge_1'])}  contig 有那些edge
    for pth in gfa.paths:
        ctgs_egs[pth.path_name] = set()
        # print(pth.path_name)
        for seg in pth.segment_names:
            # print( seg.name,seg.orient)
            ctgs_egs[pth.path_name].add(seg.name)

            if seg.name not in egs_ctgs:
                egs_ctgs[seg.name] = set()
            egs_ctgs[seg.name].add(pth.path_name)

    return graph,egs_ctgs,ctgs_egs

def read_contigs(contig_file):
    contig_seqs = {}
    for record in SeqIO.parse(contig_file, "fasta"):
        contig_seqs[record.id] = str(record.seq)
    return contig_seqs



def read_contig_info(contig_info_file):
   df = pd.read_csv(contig_info_file, delim_whitespace=True)
   cov_dict = dict(zip(df["#seq_name"], df["cov."]))
   return cov_dict



def read_bins(bin_dir,ctgs_egs=None):
    bin_list = [os.path.splitext(fn)[0] for fn in os.listdir(bin_dir) if fn.endswith('.fa')]
    # print(bin_list)
    bins_ctgs ={}
    ctgs_bins = {}

    for _b in bin_list:
        bin_file = os.path.join(bin_dir, f"{_b}.fa")
        bins_ctgs[_b] = set()

        for record in SeqIO.parse(bin_file, "fasta"):
            bins_ctgs[_b].add(record.id)
            if record.id not in ctgs_bins:
                ctgs_bins[record.id] = set()
            ctgs_bins[record.id].add(_b)
    if ctgs_egs is not None:
        bins_egs = {}
        egs_bins = {}
        for _b,ctgs in bins_ctgs.items():
            bins_egs[_b] = set()
            edge_set = set()
            for ctg in ctgs:
                if ctg in ctgs_egs:
                    edge_set.update(ctgs_egs[ctg])
            bins_egs[_b] = edge_set
            for e in edge_set:
                if e not in egs_bins:
                    egs_bins[e] = set()
                egs_bins[e].add(_b)
            
    
        return bins_egs,egs_bins
    else:
        return bins_ctgs,ctgs_bins
    

def write_bins(bins_dir, refined_list):
    os.makedirs(bins_dir, exist_ok=True)

    for _b,ctgs in refined_list.items():
        records = []
        # print(refined_bins[i].keys())
        for rn,seq in ctgs.items():
            record = SeqRecord(Seq(seq), id=rn, description="")
            records.append(record)
        bin_path = os.path.join(bins_dir, f"{_b}.fa")
        SeqIO.write(records, bin_path, "fasta-2line")


def  networkx_graph_to_gfa(G, gfa_path):

    gfa_lines = ["H\tVN:Z:1.0\n"]
    # 写入 S（segment）行：节点
    for node, attrs in G.nodes(data=True):
        seq = attrs.get('sequence', '*')  # GFA 要求有序列，默认为 *
        dp = attrs.get('dp',0)
        gfa_lines.append(f"S\t{node}\t{seq}\tdp:i:{dp}\n")

    # 写入 L（link）行：边
    for u, v, attrs in G.edges(data=True):
        # 简化：默认方向为 +，重叠为 0M
        from_orient = attrs.get('from_orient', '+')
        to_orient = attrs.get('to_orient', '+')
        overlap = attrs.get('overlap', '0M')
        gfa_lines.append(f"L\t{u}\t{from_orient}\t{v}\t{to_orient}\t{overlap}\n")
    with open(gfa_path,'w') as gfa:
        gfa.writelines(gfa_lines)


def cover_bin_ctg_egs(bins_ctgs,ctgs_egs):
    bins_egs = {}
    egs_bins = {}
    for _b,ctgs in bins_ctgs.items():
        bins_egs[_b] = set()
        edge_set = set()
        for ctg in ctgs:
            edge_set.update(ctgs_egs[ctg])
        bins_egs[_b] = edge_set
        for e in edge_set:
            if e not in egs_bins:
                egs_bins[e] = set()
            egs_bins[e].add(_b)
    
    return bins_egs,egs_bins


def load_contig_cov(file_path):
    contig_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            # 过滤空行并按空白字符（空格或制表符）分割
            parts = line.strip().split()
            if len(parts) >= 2:
                contig_dict[parts[0]] = [float(cov) for cov in parts[1:]]
    return contig_dict


def load_contig_seq(bin_dir):
    bin_list = [os.path.splitext(fn)[0] for fn in os.listdir(bin_dir) if fn.endswith('.fa')]
    # print(bin_list)
    ctgs_seq = {}

    for _b in bin_list:
        bin_file = os.path.join(bin_dir, f"{_b}.fa")
        for record in SeqIO.parse(bin_file, "fasta"):
            ctgs_seq[record.id] = record.seq
   
    return ctgs_seq



def read_gfa_paths(gfa_path):
    """
    读取GFA文件中的path信息，输出contig的组成（edge和方向）

    参数:
        gfa_path: GFA文件路径

    返回:
        dict: {path_name: [(edge_name, orientation), ...]}
              path_name是contig名称，edge_name是edge名称，orientation是方向(+/-)
    """
    gfa = gfapy.Gfa.from_file(gfa_path)
    paths_info = {}

    for pth in gfa.paths:
        path_name = pth.path_name
        paths_info[path_name] = []

        for seg in pth.segment_names:
            edge_name = seg.name
            orientation = seg.orient
            paths_info[path_name].append((edge_name, orientation))

    return paths_info

