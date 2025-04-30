#!/usr/bin/env python
# download_fineweb_parquet.py
# Copyright Lightning AI. Apache-2.0
"""
Download the FineWeb *sample-10BT* Parquet files once and store them locally.
Example:
    python litgpt/data/download_fineweb_parquet.py --output_dir dataset/fineweb_parquet
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from litgpt.utils import CLI

def download_fineweb_parquet(
    output_dir: Path = Path("dataset/fineweb_parquet"),                       # ← 必須 CLI 引数
    repo_id: str = "HuggingFaceFW/fineweb",
    sample_name: str = "sample/10BT",       # 他に sample/100BT など
    use_symlinks: bool = False,
) -> Path:
    """
    Download FineWeb sample-10BT parquet shards into *output_dir* and return the path.
    If the directory already contains *.parquet files, nothing is downloaded.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.glob("*.parquet")):
        print(f"Parquet already present in {output_dir}. Skipping download.")
        return output_dir

    #pattern = f"{sample_name}/*.parquet"
    pattern = f"{sample_name}/000_00000.parquet"
    print(f"Downloading `{pattern}` from {repo_id} …")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=output_dir,
        local_dir_use_symlinks=use_symlinks,
        resume_download=True,
    )
    print(f"Finished. Parquet stored under {output_dir}")
    return output_dir


if __name__ == "__main__":
    # jsonargparse 互換の CLI を自動生成
    CLI(download_fineweb_parquet)
