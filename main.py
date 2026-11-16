import os
import argparse
import shlex
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from dbscan import run_initial_binning
from src.bin_io import gfa_to_networkx_graph, read_bins, write_bins
from src.bin_io import load_contig_cov, load_contig_seq

from src.bin_split import bin_split, cov_split 
from src.bin_assembly import node_merge
from src.graph_refine import graph_refine
from src.combine_refine import combine_refine
from src.contamination import check_contamination, per_bin_redbscan, pooled_redbscan

from utils.diy_cmd import run_command
from utils.marker import detect_marker_gene, get_contigs_with_marker_genes
from utils.graph_depth_refine import update_gfa_depth, gfa_edge_to_fasta
from utils.paths import bin_output_path, clear_bin_fastas

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.exceptions import EfficiencyWarning
warnings.filterwarnings("ignore", category=EfficiencyWarning)



def phase2_split(init_g, bins_egs):
    """
    阶段 2：覆盖度分裂 (per-bin)
    对每个 bin 尝试 cov_split
    跳过已由 graph_split 处理过的 bin（含 comp{idx}_gs{i} 格式的）
    """
    import re
    split_bins = {}

    for bin_name, egs in bins_egs.items():
        if len(egs) == 0:
            continue

        # 跳过 graph_split 已拆分的 bin（形如 comp0_gs1 或 5_comp0_gs1）
        if re.search(r'(^|_)comp\d+_gs\d+', bin_name):
            split_bins[bin_name] = set(egs)
            continue

        new_bin_name = bin_name[:30]

        flag = None
        if len(egs) < 4:
            splited_bins, flag = [egs], None
        else:
            splited_bins = cov_split(init_g, list(egs))

        if len(splited_bins)>1:
            flag = 'sc'
        else :
            flag = None

        if flag:
            for i in range(len(splited_bins)):
                split_bins[f"{new_bin_name}_{flag}_{str(i)}"] = set(splited_bins[i])
        else:
            split_bins[new_bin_name] = set(egs)

    # 构建反向映射
    split_egs_bins = {}
    for bin_name, egs in split_bins.items():
        for eg in egs:
            split_egs_bins.setdefault(eg, set()).add(bin_name)

    print(f"阶段 2 完成: {len(bins_egs)} 个 bin → {len(split_bins)} 个 bin")
    return split_bins, split_egs_bins


def phase1_graph_refine(init_g, bins_egs, egs_bins):
    """
    阶段 2：图路径精炼 (per-component)
    利用装配图连通子图的路径搜索分配共享 edge
    """
    refined_bins_egs = graph_refine(init_g, bins_egs, egs_bins)

    # 重建反向映射
    refined_egs_bins = {}
    for bin_name, egs in refined_bins_egs.items():
        for eg in egs:
            refined_egs_bins.setdefault(eg, set()).add(bin_name)

    print(f"阶段 1 完成:{len(bins_egs.keys())} 个 bin → {len(refined_bins_egs)} 个 bin")
    return refined_bins_egs, refined_egs_bins


def phase3_combine(bins_egs, init_g, contig_marker):
    """
    阶段 3：覆盖度合并 (global)
    基于覆盖度三元组合并碎片 bin（edge 级别）
    """
    # 从 init_g 获取 edge 级别的 dp 和 length
    edge_covs = {}
    edge_lens = {}
    for node, data in init_g.nodes(data=True):
        edge_covs[node] = data.get('dp', 0) or 0
        edge_lens[node] = data.get('length', 0) or 0

    combined_bin, left_bin = combine_refine(bins_egs, edge_covs, edge_lens, contig_marker)

    all_bins = {**left_bin, **combined_bin}

    # 重建反向映射
    combined_egs_bins = {}
    for bin_name, egs in all_bins.items():
        for eg in egs:
            combined_egs_bins.setdefault(eg, set()).add(bin_name)

    print(f"阶段 3 完成: 合并后 {len(all_bins)} 个 bin")
    return all_bins, combined_egs_bins


def phase4_contamination(bins_egs, init_g, eg_rev, eg_emb_norm, contig_marker,
                         rebin_mode='per_bin', threshold=0.1, mode='comebin'):
    """
    阶段 4：污染检测与去重
    检查每个 bin 的 marker 污染度，对污染 bin 执行 re-DBSCAN
    """
    clean, contaminated = check_contamination(bins_egs, contig_marker, threshold)

    if len(contaminated) == 0:
        print("阶段 4 完成: 无污染 bin，跳过 re-DBSCAN")
        return bins_egs

    if rebin_mode == 'per_bin':
        rebin_result = per_bin_redbscan(contaminated, init_g, eg_rev, eg_emb_norm, contig_marker, mode=mode)
    elif rebin_mode == 'pooled':
        rebin_result = pooled_redbscan(contaminated, init_g, eg_rev, eg_emb_norm, contig_marker, mode=mode)
    else:
        print(f"未知 rebin_mode: {rebin_mode}，跳过 re-DBSCAN")
        rebin_result = contaminated

    final_bins = {}
    final_bins.update(clean)
    final_bins.update(rebin_result)

    print(f"阶段 4 完成: {len(final_bins)} 个 bin")
    return final_bins

def reverse_complement(dna):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    # 先替换成互补碱基，再反转字符串
    return ''.join(complement[base] for base in reversed(dna.upper()))


def output_bins(init_g, bins_egs, refined_bin_dir, gfa_path=None):
    """
    最终输出：对每个 bin 执行 node_merge 并写出 FASTA
    如果提供gfa_path，会合并bin中在同一contig上连续的edge
    """
    from src.bin_io import read_gfa_paths

    # 读取GFA path信息（如果提供）
    ctgs_egs = {}  # {contig_name: set(edge_names)}
    egs_ctgs = {}  # {edge_name: set(contig_names)}
    ctgs_path_order = {}  # {contig_name: [(edge_name, orientation), ...]}

    if gfa_path and os.path.exists(gfa_path):
        paths_info = read_gfa_paths(gfa_path)
        for ctg_name, edges_list in paths_info.items():
            ctgs_egs[ctg_name] = set()
            ctgs_path_order[ctg_name] = edges_list
            for edge_name, orient in edges_list:
                ctgs_egs[ctg_name].add(edge_name)
                if edge_name not in egs_ctgs:
                    egs_ctgs[edge_name] = set()
                egs_ctgs[edge_name].add(ctg_name)


    refined_bins = {}
    for bin_name, egs in bins_egs.items():
        if len(egs) == 0:
            continue
        sub_g = init_g.subgraph(egs).copy()
        merged_g = node_merge(sub_g)
        bin_seq = {}
        for node, n_attr in merged_g.nodes(data=True):
            bin_seq[node] = n_attr['sequence']

        inbin_ctgs = {}
        for node, seq in bin_seq.items():
            if node in egs_ctgs:
                for ctg in egs_ctgs[node]:
                    inbin_ctgs.setdefault(ctg, set()).add(node)
        used_egs = set()
        for ctg, eg_list in inbin_ctgs.items():
            if len(eg_list) > 1 and ctg in ctgs_egs and eg_list == ctgs_egs[ctg]:
                seq = ''
                seq_name = ctg
                for eg, ori in ctgs_path_order[ctg]:
                    seq += bin_seq[eg] if ori == '+' else reverse_complement(bin_seq[eg])
                    seq_name += '#' + eg
                    used_egs.add(eg)
                bin_seq[seq_name] = seq
                
        for eg in used_egs:
            del bin_seq[eg]

        refined_bins[bin_name] = bin_seq

    clear_bin_fastas(refined_bin_dir)
    write_bins(refined_bin_dir, refined_bins)
    print(f"输出完成: {len(refined_bins)} 个 bin → {refined_bin_dir}")





def main(args):
    """Run initial binning when given contigs, then refine the resulting bins."""
    if args.contig_file:
        print("\n====== Initial binning ======", flush=True)
        bins = run_initial_binning(
            contig_file=args.contig_file,
            emb_file=args.emb_file,
            output=args.output,
            data_csv=args.data_csv,
            refined_bins_folder=args.initial_bins_folder,
            threads=args.threads,
            beta=args.beta,
            mode=args.mode,
            use_depth=args.use_depth,
            depth_file=args.depth_file,
            marker_semibin=args.marker_semibin,
            marker_hmmout=args.marker_hmmout,
        )
        if not any(path.is_file() for path in Path(bins).glob('*.fa')):
            raise RuntimeError("Initial binning produced no bins; refinement was not started")
        args = argparse.Namespace(**vars(args))
        args.bins = bins
    print("\n====== Bin refinement ======", flush=True)

    read_path = args.reads
    gfa_path = args.gfa
    bin_path = args.bins
    output_path = args.output
    refined_bins_folder = args.refined_bins_folder
    is_first = args.first
    emb_file = args.emb_file
    cov_file_path = args.cov

    threads = args.threads

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    edge_path = os.path.join(output_path, 'edges.fasta')
    coverage_path = os.path.join(output_path, 'coverage.txt')
    updated_gfa_path = os.path.join(output_path, 'updated_graph.gfa')
    refined_bin_dir = bin_output_path(output_path, refined_bins_folder)
    if os.path.realpath(bin_path) == os.path.realpath(refined_bin_dir):
        raise ValueError("Input bins and output bins must use different directories")
    edge_aln_path = os.path.join(output_path, 'edges_aln.bam')

    # 首次运行时生成必要文件
    if args.edge_coverage or (not os.path.exists(updated_gfa_path)) or is_first:
        if not args.edge_coverage and not read_path:
            raise ValueError("Provide --edge_coverage for a contig-only run, or --reads to compute coverage")
        gfa_edge_to_fasta(gfa_path, edge_path)
        if args.edge_coverage:
            coverage_path = args.edge_coverage
        else:
            run_command(f"minimap2 -ax map-hifi -t {threads} {shlex.quote(edge_path)} {shlex.quote(read_path)} | samtools view -@ {threads} -b -F 0x100 | samtools sort -@ {threads} > {shlex.quote(edge_aln_path)}")
            run_command(f"samtools coverage {shlex.quote(edge_aln_path)} > {shlex.quote(coverage_path)}")
        update_gfa_depth(gfa_path, coverage_path, updated_gfa_path)

    # 检测 marker gene
    marker_gene_file = args.edge_marker_hmmout or os.path.join(output_path, 'edges_marker_genes.hmmout')
    if not os.path.exists(marker_gene_file):
        if args.edge_marker_hmmout:
            raise FileNotFoundError(marker_gene_file)
        if not os.path.exists(edge_path):
            gfa_edge_to_fasta(gfa_path, edge_path)
        detect_marker_gene(edge_path, threads, output_path, 'edges_marker_genes')
    marker_contig, mcc, contig_marker = get_contigs_with_marker_genes(marker_gene_file)

    # 加载 embedding
    embHeader = pd.read_csv(emb_file, sep='\t', nrows=1)
    embMat = pd.read_csv(emb_file, sep='\t', usecols=range(1, embHeader.shape[1])).values
    namelist = pd.read_csv(emb_file, sep='\t', usecols=range(1)).values[:, 0]

    # 构建装配图和 bin 映射
    init_g, egs_ctgs, ctgs_egs = gfa_to_networkx_graph(updated_gfa_path)
    contig_covs = load_contig_cov(cov_file_path) if cov_file_path else {}

    # --- semibin 模式: 拼接 log(depth_mean) 到 contig embedding ---
    mode = getattr(args, 'mode', 'comebin')
    data_csv = getattr(args, 'data_csv', None)
    use_depth = getattr(args, 'use_depth', False)
    depth_files = getattr(args, 'depth_file', None)
    if mode == 'semibin' and data_csv and os.path.exists(data_csv):
        print(f"[semibin mode] Augmenting embedding with depth from data.csv: {data_csv}")
        data_df = pd.read_csv(data_csv, index_col=0)
        data_df.index = data_df.index.astype(str)
        data_df = data_df.reindex(namelist)

        col_name = str(data_df.columns[-1]).split('_')[-1]
        is_combined = col_name != 'var'
        depth = data_df.values[:, 136:].astype(np.float32)
        if is_combined:
            n_sample = (data_df.shape[1] - 136)
        else:
            n_sample = (data_df.shape[1] - 136) // 2

        if not is_combined:
            mean_index = [2 * temp for temp in range(n_sample)]
            depth_mean = depth[:, mean_index]
            embMat = np.concatenate(
                (embMat, np.log(np.clip(depth_mean, 1e-6, None))), axis=1)
        print(f"  Augmented embedding: {embMat.shape}")

    # --- comebin 模式 + use_depth: 拼接 log(depth) 到 contig embedding ---
    # depth 在 contig 层面拼接，之后 edge embedding 取均值时自然包含 depth 维度
    contig_depth_feat = None
    if mode == 'comebin' and use_depth:
        if not depth_files:
            raise FileNotFoundError("--use_depth requires --depth_file to be provided")
        print(f"[comebin mode] Reading depth from {len(depth_files)} TSV file(s)")
        depth_feat_list = []
        for df_path in depth_files:
            if not os.path.exists(df_path):
                raise FileNotFoundError(f"depth file not found: {df_path}")
            df = pd.read_csv(df_path, sep='\t', header=None, comment='#')
            df.columns = ['contig', 'depth']
            df['contig'] = df['contig'].astype(str)
            df = df.set_index('contig').reindex(namelist)
            if df['depth'].isna().any():
                missing = df['depth'].isna().sum()
                print(f"  Warning: {missing} contigs not found in {df_path}, filling with 0")
            df['depth'] = df['depth'].fillna(0).astype(np.float32)
            depth_feat_list.append(df['depth'].values.reshape(-1, 1))
        contig_depth_feat = np.hstack(depth_feat_list)
        contig_depth_feat = np.log(np.clip(contig_depth_feat, 1e-6, None))
        print(f"  Contig depth feature shape: {contig_depth_feat.shape}")

    # 构建 embedding 索引
    namelist_rev = {ctg: i for i, ctg in enumerate(namelist)}
    eg_list = []
    eg_emb = []
    eg_emb_depth = []
    for eg in egs_ctgs:
        vecs = []
        depth_vecs = []
        for ctg in egs_ctgs[eg]:
            if ctg in namelist_rev:
                idx = namelist_rev[ctg]
                vecs.append(embMat[idx])
                if contig_depth_feat is not None:
                    depth_vecs.append(contig_depth_feat[idx])
        if len(vecs) == 0:
            continue
        eg_list.append(eg)
        eg_emb.append(np.mean(vecs, axis=0))
        if len(depth_vecs) > 0:
            eg_emb_depth.append(np.mean(depth_vecs, axis=0))

    if len(eg_emb) == 0:
        print("Warning: No valid embeddings found")
        exit(1)

    if mode == 'semibin':
        # semibin 模式: 不做 L2 归一化
        eg_emb_norm = np.vstack(eg_emb)
        print(f"[semibin mode] Using raw edge embeddings (no L2 normalize), shape={eg_emb_norm.shape}")
    else:
        # comebin 模式: L2 归一化 embedding 部分，再拼接 depth 部分（不归一化）
        eg_emb_norm = normalize(np.vstack(eg_emb))
        if len(eg_emb_depth) > 0:
            depth_arr = np.vstack(eg_emb_depth)
            eg_emb_norm = np.concatenate((eg_emb_norm, depth_arr), axis=1)
            print(f"[comebin mode] Augmented edge embeddings (L2 norm + depth): {eg_emb_norm.shape}")
    eg_rev = {eg: i for i, eg in enumerate(eg_list)}

    bins_egs, egs_bins = read_bins(bin_path, ctgs_egs)

    # ====== 阶段 1: 图路径精炼 (per-component) ======
    print("\n====== 阶段 1: 图路径精炼 (per-component) ======")
    refined_bins, refined_egs_bins = phase1_graph_refine(
        init_g, bins_egs, egs_bins)
    # refined_bins = bins_egs

    # ====== 阶段 2:覆盖深度分裂 (per-bin) ======
    print("\n====== 阶段 2: 覆盖深度分裂 (per-bin) ======")
    split_bins, split_egs_bins = phase2_split(
        init_g, refined_bins )

    # ====== 阶段 3: 覆盖度合并 (global, 可选) ======
    if args.combine:
        print("\n====== 阶段 3: 覆盖度合并 (global) ======")
        combined_bins, combined_egs_bins = phase3_combine(
            split_bins, init_g, contig_marker)
    else:
        combined_bins = split_bins

    # ====== 阶段 4: 污染检测与去重 ======
    print("\n====== 阶段 4: 污染检测与去重 ======")
    final_bins = phase4_contamination(
        combined_bins, init_g, eg_rev, eg_emb_norm, contig_marker,
        rebin_mode=args.rebin_mode, threshold=args.contam_threshold, mode=mode)

    # final_bins = refined_bins
    
    # ====== 最终输出: node_merge + write_bins ======
    print("\n====== 最终输出 ======")
    output_bins(init_g, final_bins, refined_bin_dir, gfa_path=args.gfa)
    return refined_bin_dir



def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Initial binning and assembly-graph refinement",
    )

    parser.add_argument("--first", action="store_true", help="force remapping from reads")

    parser.add_argument("--reads", type=str, help="input reads file path")
    parser.add_argument("--edge_coverage", type=str,
                        help="precomputed samtools coverage table for GFA edges; skips read mapping")
    parser.add_argument("--edge_marker_hmmout", type=str,
                        help="precomputed FragGeneScan-style HMMER domtblout for GFA edges")
    parser.add_argument("--gfa", type=str, required=True, help="assembly graph gfa file path")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--contig_file", type=str,
                        help="contig FASTA: run initial binning followed by refinement")
    inputs.add_argument("--bins", type=str,
                        help="existing initial bin directory: run refinement only")
    parser.add_argument("--output", type=str, required=True, help="output directory")
    parser.add_argument("--initial_bins_folder", type=str, default='dbscan_bin',
                        help="initial bin subdirectory within --output")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="marker F-beta score weighting for initial binning")
    parser.add_argument("--marker_hmmout", type=str,
                        help="precomputed contig marker domtblout for initial binning")
    parser.add_argument("--marker_semibin", action='store_true',
                        help="parse SemiBin-style contig marker ORF names during initial binning")
    parser.add_argument("--refined_bins_folder", type=str, default='refined_bins',
                        help="output folder name for bin refinement")
    parser.add_argument("--threads", type=int, default=48, help="number of threads")
    parser.add_argument("--emb_file", type=str, required=True, help="embedding file")
    parser.add_argument("--data_csv", type=str, default=None,
                        help="SemiBin data.csv (for depth mean, used when --mode semibin)")
    parser.add_argument("--mode", type=str, default='comebin', choices=['comebin', 'semibin'],
                        help="embedding processing mode: comebin (L2 normalize) or semibin (raw + log(depth_mean) concat)")
    parser.add_argument("--use_depth", action='store_true', default=False,
                        help="(comebin mode only) concatenate log(coverage depth) to the embedding after L2 normalization")
    parser.add_argument("--depth_file", type=str, nargs='+', default=None,
                        help="(comebin mode only) one or more aug*_datacoverage_mean.tsv files (2-col TSV: contig<TAB>depth). Used with --use_depth")

    parser.add_argument("--cov", type=str, help="contig coverage file path")
    parser.add_argument("--combine", action="store_true", help="enable optional coverage-based bin merging")
    parser.add_argument("--rebin_mode", type=str, default='per_bin',
                        choices=['per_bin', 'pooled'],
                        help="reclustering strategy for contaminated bins")
    parser.add_argument("--contam_threshold", type=float, default=0.1,
                        help="repeated-marker fraction threshold for reclustering")
    

    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.beta <= 0 or not np.isfinite(args.beta):
        parser.error("--beta must be a positive finite number")
    if args.first and (not args.reads or args.edge_coverage):
        parser.error("--first requires --reads and cannot be combined with --edge_coverage")
    if args.combine and not args.cov:
        parser.error("--combine requires --cov")
    if not 0 <= args.contam_threshold <= 1:
        parser.error("--contam_threshold must be between 0 and 1")
    for name in ('gfa', 'emb_file', 'reads', 'edge_coverage', 'edge_marker_hmmout',
                 'data_csv', 'cov', 'contig_file', 'marker_hmmout'):
        value = getattr(args, name)
        if value and not os.path.isfile(value):
            parser.error(f"--{name} file not found: {value}")
    if args.mode == 'comebin' and args.use_depth:
        if not args.depth_file:
            parser.error("--use_depth requires --depth_file")
        for path in args.depth_file:
            if not os.path.isfile(path):
                parser.error(f"--depth_file file not found: {path}")
    if args.bins:
        if not os.path.isdir(args.bins):
            parser.error(f"--bins directory not found: {args.bins}")
        if not any(path.is_file() for path in Path(args.bins).glob('*.fa')):
            parser.error("--bins must contain at least one .fa bin file")
    if args.contig_file and args.marker_semibin and not args.marker_hmmout:
        inferred_marker = Path(args.emb_file).resolve().parent / 'markers.hmmout'
        if not inferred_marker.is_file():
            parser.error(f"SemiBin contig markers not found: {inferred_marker}; provide --marker_hmmout")
    try:
        refined = Path(bin_output_path(args.output, args.refined_bins_folder))
        if args.contig_file:
            initial = Path(bin_output_path(args.output, args.initial_bins_folder))
            if initial.is_relative_to(refined) or refined.is_relative_to(initial):
                parser.error("Initial and refined bin directories must be separate and must not contain one another")
        elif refined == Path(args.bins).resolve():
            parser.error("Input bins and output bins must use different directories")
    except ValueError as error:
        parser.error(str(error))
    if not args.reads and not args.edge_coverage and not os.path.isfile(
            os.path.join(args.output, 'updated_graph.gfa')):
        parser.error("provide --edge_coverage for a contig-only run, or --reads to compute coverage")

    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")
