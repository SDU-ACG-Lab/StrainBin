# StrainBin

[English](README.md) | 简体中文

StrainBin 是一种面向菌株的分箱方法，结合现有分箱工具生成的 contig embedding、组装图拓扑和覆盖深度进行分箱。支持 COMEBin 和 SemiBin2 的 embedding。

## 安装

使用 Linux x86_64 和 Python 3.12，在本目录执行：

```bash
conda env create -f environment.yml
conda activate strainbin
conda install -c conda-forge -c bioconda minimap2 samtools
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

使用 `--reads` 时，`minimap2` 和 `samtools` 需要能在命令行中直接调用。导出 SemiBin2 embedding 需要 [PyTorch](https://pytorch.org/get-started/locally/)。

## 通用命令

准备 contig、Flye GFA 组装图和 contig embedding。程序从 HiFi reads 计算 GFA 节点覆盖度。已有覆盖度文件时，将 `--reads path/to/hifi_reads.fastq.gz` 替换为 `--edge_coverage path/to/coverage.tsv`。

### COMEBin 模式

```bash
python main.py \
  --contig_file path/to/assembly.fasta \
  --gfa path/to/assembly_graph.gfa \
  --emb_file path/to/embeddings.tsv \
  --reads path/to/hifi_reads.fastq.gz \
  --output output/sample \
  --threads 8
```

### SemiBin2 模式

运行 SemiBin2 后，使用 `data.csv` 和训练好的 `model.pt` 导出 embedding：

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

## 运行示例

cheese 样本 [SRR18495437](https://www.ebi.ac.uk/ena/browser/view/SRR18495437) 包含 contig、组装图、embedding 和覆盖度。先解压文件，再选择一种模式运行。

```bash
gzip -dc tests/SRR18495437/assembly.fasta.gz > tests/SRR18495437/assembly.fasta
gzip -dc tests/SRR18495437/assembly_graph.gfa.gz > tests/SRR18495437/assembly_graph.gfa
```

### COMEBin 模式

```bash
python main.py \
  --contig_file tests/SRR18495437/assembly.fasta \
  --gfa tests/SRR18495437/assembly_graph.gfa \
  --emb_file tests/SRR18495437/comebin/embeddings.tsv \
  --edge_coverage tests/SRR18495437/coverage.tsv \
  --output output/cheese_comebin \
  --threads 2
```

### SemiBin2 模式

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

两条命令都会先分箱、再优化。已有分箱时，将 `--contig_file ...` 换成 `--bins path/to/bins`，只运行优化。只需初始分箱的用法见 `python dbscan.py --help`。

## 查看结果

结果保存在 `--output` 指定的目录下：

| 目录 | 内容 |
| --- | --- |
| `dbscan_bin/` | 初始分箱 FASTA |
| `refined_bins/` | 优化后的分箱 FASTA |

例如，COMEBin 模式的最终结果在 `output/cheese_comebin/refined_bins/`。重跑命令会覆盖对应的分箱文件。

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--contig_file` | Contig FASTA，先分箱再优化 |
| `--bins` | 已有 `.fa` 分箱目录，只运行优化 |
| `--gfa` | 对应的 Flye GFA 组装图，包含节点序列、contig 路径和深度标签 |
| `--emb_file` | 带表头的 TSV，第一列为 contig ID，其余列为 embedding 数值 |
| `--mode` | `comebin`（COMEBin，默认）或 `semibin`（SemiBin2） |
| `--data_csv` | SemiBin2 的 `data.csv`，与 `--mode semibin` 配合使用 |
| `--reads` | 用于计算覆盖度的 HiFi reads |
| `--edge_coverage` | `samtools coverage` 表，第 1 列为图节点 ID，第 7 列为平均深度；提供后跳过 reads 比对 |
| `--threads` | 线程数 |
| `--output` | 输出目录 |

`--contig_file` 和 `--bins` 二选一。FASTA、embedding TSV 和组装图路径中的 contig ID 需要一致，完整参数见 `python main.py --help`。
