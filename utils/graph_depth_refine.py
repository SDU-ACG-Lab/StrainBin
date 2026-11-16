import gfapy
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def gfa_edge_to_fasta(gfa_path,fasta_path):

    records = []
    gfa = gfapy.Gfa.from_file(gfa_path)
    # 2. 加入节点
    for segment in gfa.segments:
        node_id = segment.name
        seq = segment.sequence
        record = SeqRecord(Seq(seq), id=node_id, description="")
        records.append(record)
    SeqIO.write(records, fasta_path, "fasta")

def update_gfa_depth(gfa_path, depth_path,new_gfa_path):
    meandepth_dict = {}
    with open(depth_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.strip().split('\t')
            edge_id, meandepth = cols[0], cols[6]
            meandepth_dict[edge_id] = int(float(meandepth))

    gfa = gfapy.Gfa.from_file(gfa_path)
    # 2. 加入节点
    for segment in gfa.segments:
        node_id = segment.name
        if segment.dp:
            segment.dp = meandepth_dict.get(node_id, segment.dp)
    gfa.to_file(new_gfa_path)
