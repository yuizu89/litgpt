# Copyright Lightning AI. Licensed under the Apache License 2.0
"""Prepare FineWeb *sample‑10BT* for LitGPT pre‑training (litdata style).

CLI signature and internal design mirror `prepare_starcoder.py` shipped with
LitGPT. Works with any tokenizer path (GPT‑2, Llama, etc.). On litdata v0.2 we
must ensure:
* `num_workers + num_downloaders == 1` (avoid weight=0 error)
* `get_item_weight` *or* `get_item_size` returns **positive** value.

Example (dev run):
```bash
python litgpt/data/prepare_fineweb_sample.py \
    --input_dir dataset/fineweb_parquet/sample/10BT \
    --output_dir dataset/fineweb_sample10b \
    --val_split_fraction 0.0001 \
    --fast_dev_run true
```
"""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
from lightning_utilities.core.imports import RequirementCache

from litgpt.tokenizer import Tokenizer
from litgpt.utils import CLI, extend_checkpoint_dir
from litdata.processing.data_processor import DataChunkRecipe, DataProcessor

_LITDATA_AVAILABLE = RequirementCache("litdata")
if not _LITDATA_AVAILABLE:
    raise ImportError("litdata>=0.2 が必要です")

class FineWebDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int,
        mode: str,
        val_split_fraction: float,  # 例: 0.0001 = 0.01%
    ):
        """
        tokenizer: LitGPT のトークナイザー
        chunk_size: チャンクあたりのトークン数上限
        mode: "train" or "val"
        val_split_fraction: バリデーション用に回す割合 (0 < val_split_fraction < 1)
        """
        super().__init__(chunk_size)
        if not (0.0 < val_split_fraction < 1.0):
            raise ValueError(f"val_split_fraction must be in (0,1), got {val_split_fraction}")
        # 外部指定の割合からモジュロ基数を計算して内部で使う
        self.val_split_fraction =  val_split_fraction
        self.tokenizer = tokenizer
        self.mode = mode

    def prepare_structure(self, input_dir: str):
        return [str(p) for p in Path(input_dir).rglob("*.parquet")]

    def prepare_item(self, item_metadata):
        import pyarrow.parquet as pq

        # MD5ハッシュ値を整数化してモジュロ判定するのに利用
        mod_base = int(round(1.0 / self.val_split_fraction))
        parquet_file = pq.ParquetFile(item_metadata)
        for batch in parquet_file.iter_batches(batch_size=8192, columns=["text"]):
            for text in batch.column("text").to_pylist():
                # MD5ハッシュ値を整数化してモジュロ判定
                h = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest(), "big")
                is_val = (h % mod_base) == 0
                tokens = self.tokenizer.encode(text, bos=False, eos=True)

                if self.mode == "val" and is_val:
                    yield tokens
                elif self.mode == "train" and not is_val:
                    yield tokens
        parquet_file.close()


def prepare(
    input_dir: Path = Path("dataset/fineweb_parquet/sample/10BT"),
    output_dir: Path = Path("dataset/fineweb_sample10b"),
    tokenizer_path: Path = Path("checkpoints/meta-llama/Meta-Llama-3.1-8B"),
    chunk_size: int = (2049 * 16384),
    val_split_fraction: float = 0.0001,  # 0.01% をバリデーションに回す
    fast_dev_run: bool = False,
):
    tokenizer_path = extend_checkpoint_dir(tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)

    # train 用レシピ (val_split_fraction を直接渡す)
    train_recipe = FineWebDataRecipe(
        tokenizer, chunk_size, mode="train", val_split_fraction=val_split_fraction
    )
    dp_train = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(Path(output_dir) / "train"),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count() // 2,
        num_downloaders=1,
    )
    dp_train.run(train_recipe)

    # val 用レシピ
    val_recipe = FineWebDataRecipe(
        tokenizer, chunk_size, mode="val", val_split_fraction=val_split_fraction
    )
    dp_val = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(Path(output_dir) / "val"),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count() // 2,
        num_downloaders=1,
    )
    dp_val.run(val_recipe)

    print(f"→ train data: {output_dir}/train")
    print(f"→ val   data: {output_dir}/val")


if __name__ == "__main__":
    CLI(prepare)

