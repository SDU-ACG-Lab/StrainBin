# StrainBin

English | [简体中文](README.zh-CN.md)

StrainBin is strain-aware binning method by integrating contig embeddings from existing binners with assembly-graph topology and coverage depth. It supports embeddings from COMEBin and SemiBin2.

## Install

Use Linux x86_64 and Python 3.12. Run these commands from this directory:

```bash
conda env create -f environment.yml
conda activate strainbin
conda install -c conda-forge -c bioconda minimap2 samtools
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

When using `--reads`, `minimap2` and `samtools` must be available on the command line. Exporting SemiBin2 embeddings requires [PyTorch](https://pytorch.org/get-started/locally/).
## Usage

Prepare contigs, a Flye GFA graph and contig embeddings. GFA node coverage is calculated from HiFi reads. If coverage is already available, replace `--reads path/to/hifi_reads.fastq.gz` with `--edge_coverage path/to/coverage.tsv`.

### COMEBin mode

```bash
python main.py \
  --contig_file path/to/assembly.fasta \
  --gfa path/to/assembly_graph.gfa \
  --emb_file path/to/embeddings.tsv \
  --reads path/to/hifi_reads.fastq.gz \
  --output output/sample \
  --threads 8
```

### SemiBin2 mode

After running SemiBin2, use `data.csv` and the trained `model.pt` to export embeddings:

```bash
python get_semibin_embedding.py \
  --data path/to/semibin/data.csv \
  --model path/to/semibin/model.pt \
  --output path/to/semibin/embeddings.tsv
```


```bash
python main.py \
  --contig_file path/to/assembly.fasta \
  --gfa path/to/assembly_graph.gfa \
  --emb_file path/to/semibin/embeddings.tsv \
  --mode semibin \
  --data_csv path/to/semibin/data.csv \
  --reads path/to/hifi_reads.fastq.gz \
  --output output/sample \
  --threads 8
```

## Run the example

The cheese sample [SRR18495437](https://www.ebi.ac.uk/ena/browser/view/SRR18495437) includes contigs, an assembly graph, embeddings and coverage. Extract the files first, then choose a mode. 

```bash
gzip -dc tests/SRR18495437/assembly.fasta.gz > tests/SRR18495437/assembly.fasta
gzip -dc tests/SRR18495437/assembly_graph.gfa.gz > tests/SRR18495437/assembly_graph.gfa
```

### COMEBin mode

```bash
python main.py \
  --contig_file tests/SRR18495437/assembly.fasta \
  --gfa tests/SRR18495437/assembly_graph.gfa \
  --emb_file tests/SRR18495437/comebin/embeddings.tsv \
  --edge_coverage tests/SRR18495437/coverage.tsv \
  --output output/cheese_comebin \
  --threads 2
```

### SemiBin2 mode

```bash
python main.py \
  --contig_file tests/SRR18495437/assembly.fasta \
  --gfa tests/SRR18495437/assembly_graph.gfa \
  --emb_file tests/SRR18495437/semibin/embeddings.tsv \
  --mode semibin \
  --data_csv tests/SRR18495437/semibin/data.csv \
  --edge_coverage tests/SRR18495437/coverage.tsv \
  --output output/cheese_semibin \
  --threads 2
```

Both commands run initial binning followed by refinement. To refine existing bins, replace `--contig_file ...` with `--bins path/to/bins`. For initial binning only, see `python dbscan.py --help`.

## Results

Results are saved under `--output`:

| Directory | Contents |
| --- | --- |
| `dbscan_bin/` | Initial bin FASTAs |
| `refined_bins/` | Refined bin FASTAs |

For example, the final COMEBin-mode bins are in `output/cheese_comebin/refined_bins/`. Rerunning a command replaces its bin files.

## Common options

| Option | Meaning |
| --- | --- |
| `--contig_file` | Contig FASTA; runs initial binning and refinement |
| `--bins` | Existing `.fa` bin directory; runs refinement only |
| `--gfa` | Matching Flye GFA graph, with segment sequences, contig paths and depth tags |
| `--emb_file` | TSV with a header; first column is the contig ID, remaining columns are embedding values |
| `--mode` | `comebin` (COMEBin, default) or `semibin` (SemiBin2) |
| `--data_csv` | SemiBin2 `data.csv`, used with `--mode semibin` |
| `--reads` | HiFi reads for coverage calculation |
| `--edge_coverage` | `samtools coverage` table: graph segment ID in column 1, mean depth in column 7; skips read mapping |
| `--threads` | Number of threads |
| `--output` | Output directory |

Choose either `--contig_file` or `--bins`. Contig IDs must match across the FASTA, embedding TSV and graph paths. Use `python main.py --help` for the full option list.
