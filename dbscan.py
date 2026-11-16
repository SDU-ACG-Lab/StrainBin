import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.preprocessing import normalize
from sklearn.cluster import dbscan
from sklearn.neighbors import kneighbors_graph

from utils.paths import bin_output_path, clear_bin_fastas
from src.bin_io import write_bins, read_contigs
from utils.marker import detect_marker_gene, get_contigs_with_marker_genes, get_contigs_with_marker_genes_semibin

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.exceptions import EfficiencyWarning
warnings.filterwarnings("ignore", category=EfficiencyWarning)


# ---------------------------------------------------------------------------
# SemiBin long-read-style helpers (used when --mode semibin)
# ---------------------------------------------------------------------------
def norm_abundance(data):
    """Replicates SemiBin/utils.py norm_abundance logic."""
    n = data.shape[1] - 136
    if n >= 20:
        return True
    return n >= 5 and np.mean(np.sum(data[:, 136:], axis=1)) > 2


def determine_is_combined(data):
    """
    Determine whether the data uses combined features.
    Mirrors SemiBin/main.py binning_preprocess.
    """
    col_name = str(data.columns[-1]).split('_')[-1]
    return col_name != 'var'


def build_longread_embedding(emb_mat, depth=None, is_combined=False, n_sample=1):
    """
    Build the augmented embedding vector exactly as SemiBin's
    long_read_cluster.py does:
      - single mode: concatenate embedding with log(depth_mean)
      - combined mode: embedding_new = embedding (no depth concatenation)
    """
    if not is_combined and depth is not None:
        mean_index = [2 * temp for temp in range(n_sample)]
        depth_mean = depth[:, mean_index]
        embedding_new = np.concatenate(
            (emb_mat, np.log(np.clip(depth_mean, 1e-6, None))), axis=1)
    else:
        embedding_new = emb_mat
    return embedding_new


def get_best_bin(results, contig_to_marker, namelist, processed_ctg, contig_dict, minfasta, beta=1):
    max_F1 = 0
    weight_of_max = 1e9
    max_bin = None

    for res_labels in results:
        res = defaultdict(list)
        for label, name in zip(res_labels, namelist):
            if label != -1 and name not in processed_ctg:
                res[label].append(name)
        for bin_contig in res.values():
            cur_weight = sum(len(contig_dict[contig]) for contig in bin_contig)
            if cur_weight < minfasta:
                continue
            marker_list = []
            for contig in bin_contig:
                if contig in contig_to_marker:
                    marker_list.extend(contig_to_marker[contig])
            if len(marker_list) == 0:
                continue
            recall = len(set(marker_list)) / 107
            contamination = (len(marker_list) - len(set(marker_list))) / len(marker_list)
            F1 = (1 + beta * beta) * recall * (1 - contamination) / (
                        recall + (1 - contamination) * beta * beta)

            if F1 > max_F1:
                max_F1 = F1
                weight_of_max = cur_weight
                max_bin = bin_contig
            elif F1 == max_F1 and cur_weight <= weight_of_max:
                weight_of_max = cur_weight
                max_bin = bin_contig
    if max_F1 > 0:
        return max_bin


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Initial marker-guided binning")
    parser.add_argument("--contig_file", type=str, required=True, help="contig fasta file")
    parser.add_argument("--emb_file", type=str, required=True, help="embedding file (embeddings.tsv)")
    parser.add_argument("--data_csv", type=str, default=None,
                        help="SemiBin data.csv (for depth mean, used when --mode semibin)")
    parser.add_argument("--output", type=str, required=True, help="output folder name for bin refinement")
    parser.add_argument("--refined_bins_folder", type=str ,default='dbscan_bin',help="output folder name for bin refinement")
    parser.add_argument("--threads", type=int ,default=48,help="number of threads")
    parser.add_argument("--beta", type=float, default=1.0, help="marker F-beta score weighting")
    parser.add_argument("--mode", type=str, default='comebin', choices=['comebin', 'semibin'],
                        help="embedding processing mode: comebin (L2 normalize) or semibin (raw + log(depth_mean) concat)")
    parser.add_argument("--use_depth", action='store_true', default=False,
                        help="(comebin mode only) append log(coverage depth) after L2 normalization")
    parser.add_argument("--depth_file", type=str, nargs='+', default=None,
                        help="(comebin mode only) one or more aug*_datacoverage_mean.tsv files (2-col TSV: contig<TAB>depth). Used with --use_depth as an alternative to --data_csv")
    parser.add_argument("--marker_semibin", action='store_true', default=False,
                        help="parse markers from a SemiBin-style markers.hmmout (prodigal/fast-naive ORFs) instead of running FragGeneScan + hmmsearch")
    parser.add_argument("--marker_hmmout", type=str, default=None,
                        help="precomputed marker domtblout; add --marker_semibin for SemiBin ORF names")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in ('contig_file', 'emb_file', 'data_csv', 'marker_hmmout'):
        value = getattr(args, name)
        if value and not os.path.isfile(value):
            parser.error(f"--{name} file not found: {value}")

    if args.beta <= 0 or not np.isfinite(args.beta):
        parser.error("--beta must be a positive finite number")
    if args.mode == 'comebin' and args.use_depth:
        if not args.depth_file:
            parser.error("--use_depth requires --depth_file")
        for path in args.depth_file:
            if not os.path.isfile(path):
                parser.error(f"--depth_file file not found: {path}")
    try:
        bin_output_path(args.output, args.refined_bins_folder)
    except ValueError as error:
        parser.error(str(error))
    return args


def run_initial_binning(contig_file, emb_file, output, *, data_csv=None,
                        refined_bins_folder='dbscan_bin', threads=48, beta=1.0,
                        mode='comebin', use_depth=False, depth_file=None,
                        marker_semibin=False, marker_hmmout=None):
    """Generate initial bin FASTAs and return their directory.

    This is shared by the standalone dbscan.py CLI and the main.py workflow.
    """
    output_dir = output
    depth_files = depth_file
    refined_bin_dir = bin_output_path(output_dir, refined_bins_folder)
    os.makedirs(output_dir, exist_ok=True)

    ctg_seqs = read_contigs(contig_file)

    if marker_semibin:
        # SemiBin-style markers: parse markers.hmmout directly, skip FragGeneScan
        marker_gene_file = marker_hmmout
        if marker_gene_file is None:
            marker_gene_file = os.path.join(os.path.dirname(os.path.abspath(emb_file)), 'markers.hmmout')
        if not os.path.exists(marker_gene_file):
            raise FileNotFoundError(f"--marker_semibin enabled but markers.hmmout not found: {marker_gene_file}")
        print(f"[marker semibin] Parsing markers from {marker_gene_file}")
        marker_contig, mcc, contig_marker = get_contigs_with_marker_genes_semibin(marker_gene_file, ctg_seqs)
    else:
        marker_gene_file = marker_hmmout or os.path.join(output_dir,'contigs_marker_genes.hmmout')

        if not os.path.exists(marker_gene_file):
            detect_marker_gene(contig_file, threads, output_dir, hmmres_name='contigs_marker_genes')
        marker_contig,mcc,contig_marker = get_contigs_with_marker_genes(marker_gene_file)

    embHeader = pd.read_csv(emb_file, sep='\t', nrows=1)
    embMat = pd.read_csv(emb_file, sep='\t', usecols=range(1, embHeader.shape[1])).values
    namelist = pd.read_csv(emb_file, sep='\t', usecols=range(1)).values[:, 0]

    length_weight = [len(ctg_seqs[c] if c in ctg_seqs else '') for c in namelist]

    if mode == 'semibin':
        # --- SemiBin long-read style: raw embedding + log(depth_mean) concatenation ---
        if data_csv and os.path.exists(data_csv):
            print(f"[semibin mode] Using data.csv: {data_csv}")
            data = pd.read_csv(data_csv, index_col=0)
            data.index = data.index.astype(str)
            data = data.reindex(namelist)

            is_combined = determine_is_combined(data)
            depth = data.values[:, 136:].astype(np.float32)
            if is_combined:
                n_sample = (data.shape[1] - 136)
            else:
                n_sample = (data.shape[1] - 136) // 2
            print(f"  is_combined: {is_combined}, n_sample: {n_sample}")

            embedding_new = build_longread_embedding(
                emb_mat=embMat, depth=depth, is_combined=is_combined, n_sample=n_sample)
            print(f"  Augmented embedding: {embedding_new.shape}")
        else:
            print("[semibin mode] No data.csv, using raw embedding only")
            embedding_new = embMat

        # No L2 normalization, direct KNN on augmented embedding
        dist_matrix = kneighbors_graph(
            embedding_new,
            n_neighbors=min(200, embedding_new.shape[0] - 1),
            mode='distance', p=2, n_jobs=threads)

        # SemiBin long-read eps_list and min_samples=5
        eps_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
        min_samples = 5
    else:
        # --- comebin mode (original): L2 normalize embedding ---
        print("[comebin mode] L2-normalized embedding")
        norm_embeddings = normalize(embMat)

        if use_depth:
            if not depth_files:
                raise FileNotFoundError("--use_depth requires --depth_file to be provided")
            # --- comebin depth TSV format (aug*_datacoverage_mean.tsv) ---
            # 2-column TSV: contig_name<TAB>depth_value, no header
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
            depth_feat = np.hstack(depth_feat_list)
            depth_feat = np.log(np.clip(depth_feat, 1e-6, None))
            print(f"  Depth feature shape: {depth_feat.shape}")

            norm_embeddings = np.concatenate((norm_embeddings, depth_feat), axis=1)
            print(f"  Augmented embedding (after L2 norm + depth): {norm_embeddings.shape}")

        dist_matrix = kneighbors_graph(
            norm_embeddings,
            n_neighbors=min(200, norm_embeddings.shape[0] - 1),
            mode='distance', p=2, n_jobs=threads)

        eps_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        min_samples = 3

    dbscan_results = []
    for eps_value in eps_list:
        try:
            _, labels = dbscan(dist_matrix,
                    eps=eps_value, min_samples=min_samples, n_jobs=threads, metric='precomputed',
                    sample_weight=length_weight)
        except (IndexError, TypeError):
            dense_dist = np.array(dist_matrix.todense())
            np.fill_diagonal(dense_dist, 0)
            _, labels = dbscan(dense_dist,
                    eps=eps_value, min_samples=min_samples, n_jobs=1, metric='precomputed',
                    sample_weight=length_weight)
        dbscan_results.append(labels.tolist())

    best_bins = []
    processed_ctg = set()
    minfasta = 200000
    remaining_len = sum(len(ctg_seqs[c]) for c in namelist)

    while remaining_len >= minfasta:
        max_bin = get_best_bin(dbscan_results,
                                contig_marker,
                                namelist,
                                processed_ctg,
                                ctg_seqs,
                                minfasta,
                                beta)
        if not max_bin:
            break
        best_bins.append(max_bin)
        processed_ctg.update(max_bin)
        remaining_len -= sum([len(ctg_seqs[c]) for c in max_bin])

    refined_bins = {}
    for i, binctg in enumerate(best_bins):
        binctg_seq = {c: ctg_seqs[c] for c in binctg}
        refined_bins[f'{i}'] = binctg_seq

    clear_bin_fastas(refined_bin_dir)
    write_bins(refined_bin_dir, refined_bins)
    print(f"Output: {len(refined_bins)} bins → {refined_bin_dir}")
    return refined_bin_dir


if __name__ == "__main__":
    run_initial_binning(**vars(parse_args()))
