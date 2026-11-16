"""Output handling shared by the two command-line entry points."""

from pathlib import Path


def bin_output_path(output, folder):
    root = Path(output).resolve()
    target = (root / folder).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("--refined_bins_folder must name a subdirectory within --output")
    return str(target)


def clear_bin_fastas(directory):
    """Replace bin FASTAs without deleting unrelated files or subdirectories."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = list(directory.iterdir())
    if any(p.suffix != '.fa' or not p.is_file() or p.is_symlink() for p in entries):
        raise ValueError(f"Bin output directory contains files other than bin FASTAs: {directory}")
    for path in entries:
        path.unlink()
