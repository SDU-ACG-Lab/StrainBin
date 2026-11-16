import os 
from .diy_cmd import run_command
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio import SeqIO
import csv
import networkx as nx
import copy
import re




# 可以通过sylph快速找出对应的reference，也可以将read/contig 比对到所有的reference上，找出coverage超过某个阈值的reference
# 这里假设找出来的reference放在selected_refs 文件夹下



def merge_intervals(_intervals):
    """
    intervals: list of [start, end]
    Returns merged non-overlapping list of intervals.
    """
    if not _intervals:
        return []
    # sort by start
    intervals = copy.deepcopy(_intervals)
    intervals.sort()
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            # overlap: merge
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged


def minimap_contig(ref_path,contig_path, output_path, threads=2):
    """
    使用 minimap2 将 contig 比对到参考基因组。
    """
    os.makedirs(output_path, exist_ok=True)
    fn = os.path.basename(ref_path)
    fn = os.path.splitext(fn)[0]  # 去掉扩展名
    output_path = os.path.join(output_path, f"{fn}.paf")
    run_command(f"minimap2 -x asm20 -t 2 -m10000  {ref_path} {contig_path} > {output_path}")
    return output_path


def find_fna_files(root_dir):
    matches = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.fna'):
                matches.append(os.path.join(dirpath, filename))
    return matches

def delete_empty_files(target_dir):
    deleted = []
    for dirpath, dirnames, filenames in os.walk(target_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    deleted.append(file_path)
            except Exception as e:
                print(f"[WARN] {file_path}: {e}")

def get_refs_aln(ref_dir, contig_path, aln_dir, threads=48):
    if not os.path.exists(ref_dir):
        raise FileNotFoundError(f"Reference directory {ref_dir} does not exist.")
    os.makedirs(aln_dir, exist_ok=True)

    start = time.perf_counter()
    refs = find_fna_files(ref_dir)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = { pool.submit(minimap_contig, ref_path, contig_path, aln_dir): ref_path for ref_path in refs }

        for fut in as_completed(futures):
            ref = futures[fut]
            try:
                map_file = fut.result()
            except Exception as e:
                print(f"[EXC] {ref}: {e}")
    end = time.perf_counter()
    delete_empty_files(aln_dir)
    print(f"[TIME] All mapping jobs completed, total elapsed time: {end - start:.2f} seconds")



def report_mapped_info(ref_dir, ref_aln_dir):
    ref_names =  [os.path.splitext(fn)[0]  for fn in os.listdir(ref_aln_dir) if fn.endswith('.paf')]

    mapped_ref_info = {}

    for rn in ref_names:

        rn_fna = os.path.join(ref_dir, f"{rn}.fna")
        rn_len =sum(len(record.seq) for record in SeqIO.parse(rn_fna, "fasta"))
        chr_names = [record.id for record in SeqIO.parse(rn_fna, "fasta")]
        chrs_complete = [ ('complete genome' in record.description.lower() and 'chromosome' in record.description.lower()) for record in SeqIO.parse(rn_fna, "fasta")]
        paf_path  = os.path.join(ref_aln_dir, f"{rn}.paf")

        ctgs = {}
        with open(paf_path) as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) < 9:
                    continue
                ctg_name = row[0]
                ctg_len = int(row[1])
                a, b = int(row[2]), int(row[3])
                ctg_s, ctg_e = min(a, b), max(a, b)
                chr_name = row[5]
                a, b = int(row[7]), int(row[8])
                chr_s, chr_e = min(a, b), max(a, b)
                strand = row[4]

                if ctg_name not in ctgs:
                    ctgs[ctg_name] = {
                        "length": ctg_len,
                        "aligned": {}
                    }
                if chr_name not in ctgs[ctg_name]["aligned"]:
                    ctgs[ctg_name]["aligned"][chr_name] = {
                        'ctg_intervals': [],
                        'chr_intervals': [],
                        'strands': [],
                    }
                ctgs[ctg_name]["aligned"][chr_name]['ctg_intervals'].append([ctg_s, ctg_e])
                ctgs[ctg_name]["aligned"][chr_name]['chr_intervals'].append([chr_s, chr_e])
                ctgs[ctg_name]["aligned"][chr_name]['strands'].append(strand)
                

        chr_intervals = {}
        mapped_ctgs = []

        # if 'GCF_001606025.1_ASM160602v1_genomic' in paf_path:
        #     pass

        for ctg in ctgs:
            ctg_len = ctgs[ctg]['length']

            ctg_intervals = []
            # print(ctgs[ctg])
            # print(ctgs[ctg].items())
            for chr_name, intervals in ctgs[ctg]["aligned"].items():
            #     print(intervals)
                ctg_intervals += intervals['ctg_intervals']

            merged_ctg_intervals = merge_intervals(ctg_intervals)
            ctg_covered_length = sum(e - s for s, e in merged_ctg_intervals)

            if ctg_covered_length/ctg_len < 0.7:
                continue
            
            mapped_ctgs.append(ctg)
            for chr_name, intervals in ctgs[ctg]["aligned"].items():
                if chr_name not in chr_intervals:
                    chr_intervals[chr_name] = []
                chr_intervals[chr_name] += intervals['chr_intervals']
            pass

        total_chr_mapped = 0
        for chr,intervals in chr_intervals.items():
            merged_intervals = merge_intervals(intervals)
            chr_len = sum(e - s for s, e in merged_intervals)
            total_chr_mapped += chr_len
        
        mapped_ref_info[rn] = {
            'chrs': chr_names,
            'chrs_complete': chrs_complete,
            'mapped_ctgs': mapped_ctgs,
            'mapped_ctgs_info': ctgs,
            'length': rn_len,
            'mapped_length': total_chr_mapped,
            'mapped_percent': total_chr_mapped / rn_len if rn_len > 0 else 0,
        }

    return mapped_ref_info


def build_edge_link(mapped_ref_info, compete = 0.5):
    link_g = nx.Graph()
    for ref, info in mapped_ref_info.items():
        if len(info['mapped_ctgs']) < 2:
            continue
        if info['mapped_percent'] > compete:
            continue
        for i in range(len(info['mapped_ctgs'])):
            for j in range(i + 1, len(info['mapped_ctgs'])):
                ctg1 = info['mapped_ctgs'][i]
                ctg2 = info['mapped_ctgs'][j]
                link_g.add_edge(ctg1, ctg2)

    return link_g

def sum_bin_mapped_ref(ref_info, bin_ctgs):

    ref_len = ref_info['length']
    com_ctgs = set(ref_info['mapped_ctgs']) & set(bin_ctgs)
    chr_intvs = {chr:[] for chr in ref_info['chrs']}
     
    for ctg in com_ctgs:
        for mapped_chr in ref_info['mapped_ctgs_info'][ctg]["aligned"]:
            if mapped_chr not in chr_intvs:
                continue
            for interval in ref_info['mapped_ctgs_info'][ctg]["aligned"][mapped_chr]['chr_intervals']:
                chr_intvs[mapped_chr].append(interval)

    merged_mapped_length = 0
    for chr, intervals in chr_intvs.items():
        if len(intervals) == 0:
            continue
        merged_intervals = merge_intervals(intervals)
        chr_len = sum(e - s for s, e in merged_intervals)
        merged_mapped_length += chr_len
    
    return merged_mapped_length

def extract_refer_files_ovlp(ovlp_path, database_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    found = set()
    ids = set()
    for root, dirs, files in os.walk(ovlp_path):
        for _fn in files:
            fn = os.path.splitext(_fn)[0]  # 去掉扩展名
            ids.add(fn)

    for root, dirs, files in os.walk(database_dir):
        for fn in files:
            if not (fn.endswith(".fna") or fn.endswith(".fna.gz")):
                continue
            # print(f"Checking {fn} in {root}")
            for genome_id in ids:
                if genome_id in fn:
                    src_path = os.path.join(root, fn)
                    dst_path = os.path.join(output_dir, fn)
                    # print(f"Found {genome_id} in {src_path}, copying to {dst_path}")
                    run_command(f"cp {src_path} {dst_path}")
                    found.add(dst_path)
                    break
    
    print(f"Extracted {len(found)} / {len(ids)} unique genome IDs from  {ovlp_path}.")
    return found,ids


def extract_ids(tsv_path):
    """
    从TSV的第二列里提取所有唯一的 GCA_/GCF_ ID（不含 _genomic 后缀）。
    增加调试输出与异常行提示，防止意外格式导致数据丢失。
    """
    ids = set()
    pattern = re.compile(r"(GC[AF]_\d+\.\d+)[^/]*\.fna(?:\.gz)?")
    with open(tsv_path) as f:
        header = next(f)  # 跳过表头
        for idx, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            # 检查分列数量是否合理
            col2 = parts[1]
            # print(col2)
            m = pattern.search(col2)
            if m:
                ids.add(m.group(1))
            else:
                print(f"警告：第{idx}行未匹配ID，Genome_file字段内容为：{col2}")
    return ids


def sylph_extract_refer_files(tsv_path,database_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    found = set()
    ids = extract_ids(tsv_path)
    print(ids)
    for root, dirs, files in os.walk(database_dir):
        for fn in files:
            if not (fn.endswith(".fna") or fn.endswith(".fna.gz")):
                continue
            # print(f"Checking {fn} in {root}")
            for genome_id in ids:
                if genome_id in fn:
                    src_path = os.path.join(root, fn)
                    dst_path = os.path.join(output_dir, fn)
                    # print(f"Found {genome_id} in {src_path}, copying to {dst_path}")
                    run_command(f"cp {src_path} {dst_path}")
                    found.add(dst_path)
                    break
    
    print(f"Extracted {len(found)} / {len(ids)} unique genome IDs from  {tsv_path}.")
    return found,ids
