"""Marker parsing and optional detection."""

import os
from pathlib import Path
import subprocess


def detect_marker_gene(contig_file, threads, output_dir,
                       hmmres_name='contigs_marker_genes'):
    """Run the repository's FragGeneScan and HMMER with its marker profiles."""
    auxiliary = Path(__file__).resolve().parents[1] / 'Auxiliary'
    marker_profiles = auxiliary / 'marker.hmm'
    if not marker_profiles.is_file():
        raise FileNotFoundError(f'Bundled marker profiles not found: {marker_profiles}')
    fraggenescan = auxiliary / 'FragGeneScan' / 'FragGeneScan'
    hmmsearch = auxiliary / 'hmmer-3.3.2' / 'bin' / 'hmmsearch'
    for program in (fraggenescan, hmmsearch):
        if not program.is_file():
            raise FileNotFoundError(f'Bundled marker tool not found: {program}')
        if not os.access(program, os.X_OK):
            raise PermissionError(f'Bundled marker tool is not executable: {program}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f'{hmmres_name}.frag'
    proteins = Path(str(prefix) + '.faa')
    hmmout = output_dir / f'{hmmres_name}.hmmout'
    with (output_dir / f'{hmmres_name}.frag.log').open('w') as log:
        subprocess.run([fraggenescan, '-s', str(contig_file), '-o', str(prefix),
                        '-w', '0', '-t', 'complete', '-p', str(threads)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if not proteins.is_file():
        raise RuntimeError(f'FragGeneScan did not produce {proteins}')
    with (output_dir / f'{hmmres_name}.hmmsearch.log').open('w') as log:
        subprocess.run([hmmsearch, '--domtblout', str(hmmout), '--cut_tc',
                        '--cpu', str(threads), str(marker_profiles), str(proteins)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if not hmmout.is_file():
        raise RuntimeError(f'hmmsearch did not produce {hmmout}')
    for suffix in ('.faa', '.ffn', '.out', '.gff'):
        Path(str(prefix) + suffix).unlink(missing_ok=True)


# Get contigs containing marker genes
def get_contigs_with_marker_genes(
    hmm_file, dict_contigRev=None,  mg_length_threshold = 0.9
):
    '''
    contig_file: hmm file end with '.hmmout'
    mg_length_thres: thres of marker gene
    
    '''
    marker_contigs = {}
    marker_contig_counts = {}
    contig_markers = {}

    with open(hmm_file, "r") as myfile:
        for line in myfile.readlines():
            if not line.startswith("#"):
                strings = line.strip().split()

                contig = strings[0]

                # Marker gene name
                marker_gene = strings[3]

                # Marker gene length
                marker_gene_length = int(strings[5])

                # Mapped marker gene length
                mapped_marker_length = int(strings[16]) - int(strings[15])

                name_strings = contig.split("_")
                name_strings = name_strings[: len(name_strings) - 3]

                # Contig name
                contig_name = "_".join(name_strings)

                
                if (dict_contigRev != None) and (contig_name not in dict_contigRev) :
                    continue

                if mapped_marker_length > marker_gene_length * mg_length_threshold:
                    marker_repeated_in_contig = False

                    # Get marker genes in each contig
                    if contig_name not in contig_markers:
                        contig_markers[contig_name] = [marker_gene]
                    else:
                        if marker_gene not in contig_markers[contig_name]:
                            contig_markers[contig_name].append(marker_gene)

                    # Get contigs containing each marker gene
                    if marker_gene not in marker_contigs:
                        marker_contigs[marker_gene] = [contig_name]
                    else:
                        if contig_name not in marker_contigs[marker_gene]:
                            marker_contigs[marker_gene].append(contig_name)
                        else:
                            marker_repeated_in_contig = True

                    # Get contig counts for each marker
                    if marker_gene not in marker_contig_counts:
                        marker_contig_counts[marker_gene] = 1
                    else:
                        if not marker_repeated_in_contig:
                            marker_contig_counts[marker_gene] += 1

    return marker_contigs, marker_contig_counts, contig_markers


# TIGRFAM families merged in the 107-marker set (111 families -> 107),
# same normalization as SemiBin (keeps F1 denominator consistent)
TIGR_RENAME = {
    'TIGR00388': 'TIGR00389',
    'TIGR00471': 'TIGR00472',
    'TIGR00408': 'TIGR00409',
    'TIGR02386': 'TIGR02387',
}


# Get contigs containing marker genes, parsing a SemiBin-style
# markers.hmmout with the same rules as SemiBin/markers.py:get_marker
def get_contigs_with_marker_genes_semibin(
    hmm_file, contig_seqs, mg_cov_threshold=0.4, min_contig_len=1000
):
    '''
    hmm_file: domtblout produced by hmmsearch on fast-naive/prodigal ORFs
              (ORF names follow the '{contig}_{idx}' convention).
    contig_seqs: dict {contig_name: sequence}, used for the
                 >= min_contig_len filter.
    mg_cov_threshold: keep hit if (qend - qstart) > qlen * threshold.
    min_contig_len: only hits on contigs of at least this length are kept.

    Returns the same triple as get_contigs_with_marker_genes:
        (marker_contigs, marker_contig_counts, contig_markers)
    '''
    marker_contigs = {}
    marker_contig_counts = {}
    contig_markers = {}

    with open(hmm_file, "r") as myfile:
        for line in myfile.readlines():
            if not line.startswith("#"):
                strings = line.strip().split()

                orf = strings[0]

                # Marker gene name
                marker_gene = strings[3]

                # Marker gene length (HMM profile length)
                marker_gene_length = int(strings[5])

                # Mapped marker gene length (hmm coords)
                mapped_marker_length = int(strings[16]) - int(strings[15])

                # ORF name -> contig name ('{contig}_{idx}')
                contig_name = orf.rsplit('_', 1)[0]

                # Skip hits on short contigs (or contigs not in the assembly)
                if contig_name not in contig_seqs or len(contig_seqs[contig_name]) < min_contig_len:
                    continue

                if mapped_marker_length > marker_gene_length * mg_cov_threshold:
                    # Normalize TIGRFAM family names (111 -> 107)
                    marker_gene = TIGR_RENAME.get(marker_gene, marker_gene)

                    marker_repeated_in_contig = False

                    # Get marker genes in each contig
                    if contig_name not in contig_markers:
                        contig_markers[contig_name] = [marker_gene]
                    else:
                        if marker_gene not in contig_markers[contig_name]:
                            contig_markers[contig_name].append(marker_gene)

                    # Get contigs containing each marker gene
                    if marker_gene not in marker_contigs:
                        marker_contigs[marker_gene] = [contig_name]
                    else:
                        if contig_name not in marker_contigs[marker_gene]:
                            marker_contigs[marker_gene].append(contig_name)
                        else:
                            marker_repeated_in_contig = True

                    # Get contig counts for each marker
                    if marker_gene not in marker_contig_counts:
                        marker_contig_counts[marker_gene] = 1
                    else:
                        if not marker_repeated_in_contig:
                            marker_contig_counts[marker_gene] += 1

    return marker_contigs, marker_contig_counts, contig_markers
