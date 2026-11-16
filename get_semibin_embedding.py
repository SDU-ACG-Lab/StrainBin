#!/usr/bin/env python
"""
Generate contig embeddings from SemiBin's data.csv and a trained model.pt.

The output format matches COMEBin's embeddings.tsv:
  - Tab-separated
  - First column: contig name (index)
  - Remaining columns: embedding dimensions (header: 0, 1, 2, ...)
  - Written via pandas DataFrame.to_csv(..., sep='\t', header=True)

Usage:
    python get_semibin_embedding.py \
        --data data.csv \
        --model model.pt \
        --output embeddings.tsv \
        [--device cpu|cuda] \
        [--batch-size 2048]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Model definitions (mirrors SemiBin/semi_supervised_model.py)
# ---------------------------------------------------------------------------
from torch.nn import Linear, LeakyReLU
from torch import nn


class Semi_encoding_single(torch.nn.Module):
    """Model for k-mer features (is_combined=False)."""

    def __init__(self, num):
        super(Semi_encoding_single, self).__init__()
        self.encoder1 = torch.nn.Sequential(
            Linear(num, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 100),
        )
        self.decoder1 = torch.nn.Sequential(
            Linear(100, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, num),
            nn.Softmax(dim=1),
        )

    def forward(self, input1, input2):
        return self.encoder1(input1), self.encoder1(input2)

    def embedding(self, input):
        return self.encoder1(input)


class Semi_encoding_multiple(torch.nn.Module):
    """Model for combined features (is_combined=True)."""

    def __init__(self, num):
        super(Semi_encoding_multiple, self).__init__()
        self.encoder1 = torch.nn.Sequential(
            Linear(num, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 100),
        )
        self.decoder1 = torch.nn.Sequential(
            Linear(100, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, 512),
            nn.BatchNorm1d(512),
            LeakyReLU(),
            nn.Dropout(0.2),
            Linear(512, num),
            nn.Sigmoid(),
        )

    def forward(self, input1, input2):
        return self.encoder1(input1), self.encoder1(input2)

    def embedding(self, input):
        return self.encoder1(input)


def model_load(path, device):
    """Load a SemiBin model from a .pt checkpoint."""
    saved = torch.load(path, map_location=device, weights_only=True)
    model_name = saved['model_name']
    params = saved['params']

    if model_name == 'Semi_encoding_single':
        model = Semi_encoding_single(params[0])
    elif model_name == 'Semi_encoding_multiple':
        model = Semi_encoding_multiple(params[0])
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")

    model.load_state_dict(saved['model_state_dict'])
    return model.to(device)


# ---------------------------------------------------------------------------
# Helpers
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

    Mirrors SemiBin/main.py binning_preprocess:
      - If the last column name ends with '_var' -> not combined (single mode)
      - Otherwise -> combined
    """
    col_name = str(data.columns[-1]).split('_')[-1]
    return col_name != 'var'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate contig embeddings from SemiBin data.csv and model.pt')
    parser.add_argument('--data', required=True,
                        help='Path to SemiBin data.csv file')
    parser.add_argument('--model', required=True,
                        help='Path to trained model.pt file')
    parser.add_argument('--output', required=True,
                        help='Path to output embeddings.tsv file')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for inference (default: cpu)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size for inference (default: 2048)')
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load data ---
    print(f'Reading data from {args.data} ...')
    data = pd.read_csv(args.data, index_col=0)
    data.index = data.index.astype(str)
    print(f'  Data shape: {data.shape} ({data.shape[0]} contigs, {data.shape[1]} features)')

    # --- Determine mode ---
    is_combined = determine_is_combined(data)
    print(f'  is_combined: {is_combined}')

    # --- Prepare input features ---
    if not is_combined:
        # Single mode: use only the first 136 k-mer columns
        train_data_input = data.values[:, 0:136]
    else:
        # Combined mode: use all columns, apply normalization if needed
        train_data_input = data.values
        if norm_abundance(train_data_input):
            from sklearn.preprocessing import normalize
            print('  Applying abundance normalization (L1) ...')
            norm = np.sum(train_data_input, axis=0)
            train_data_input = train_data_input / norm
            train_data_input = normalize(train_data_input, axis=1, norm='l1')

    # --- Load model ---
    print(f'Loading model from {args.model} ...')
    model = model_load(args.model, device)
    model.eval()

    # Verify input dimension matches model
    expected_in_features = model.encoder1[0].in_features
    actual_in_features = train_data_input.shape[1]
    if expected_in_features != actual_in_features:
        print(f'  WARNING: Model expects {expected_in_features} input features, '
              f'but data has {actual_in_features}.')
        if not is_combined and actual_in_features > expected_in_features:
            # Trim to expected size
            print(f'  Trimming input to first {expected_in_features} features ...')
            train_data_input = train_data_input[:, :expected_in_features]
        else:
            print('  ERROR: Input dimension mismatch cannot be resolved automatically.')
            sys.exit(1)

    # --- Generate embeddings ---
    print('Generating embeddings ...')
    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(train_data_input), args.batch_size):
            end = min(start + args.batch_size, len(train_data_input))
            batch = torch.from_numpy(train_data_input[start:end]).to(device)
            emb = model.embedding(batch.float()).detach().cpu().numpy()
            all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f'  Embeddings shape: {embeddings.shape}')

    # --- Save to TSV (COMEBin format) ---
    print(f'Writing embeddings to {args.output} ...')
    embeddings_df = pd.DataFrame(embeddings, index=data.index)
    embeddings_df.to_csv(args.output, sep='\t', header=True)
    print('Done.')


if __name__ == '__main__':
    main()
