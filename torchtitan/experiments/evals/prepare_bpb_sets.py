# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the held-out BPB sets for BPBValidator (the training-loop scorer).

Writes, under ``--output-folder``:

- ``inloop/<name>.parquet``: a fixed random slice of each set's validation
  split, at most ``--max-bytes`` of text (8 MB is about 2M tokens), drawn once
  with ``--seed`` so every run scores the same documents;
- ``manifest.json``: the sets, read by ``bpb_sets_from_manifest``.

The eval job scores the full validation splits (``trainstation_ppl``).

Sets: the nine Paloma sources in trainstation_paloma, Paloma PTB and
WikiText-103 (logged, not averaged), and OpenWebMath (Proof-Pile-2
open-web-math). Paloma is gated: accept its terms on Hugging Face first.
OpenWebMath is read from compressed JSON Lines and needs ``zstandard``.

    python -m torchtitan.experiments.evals.prepare_bpb_sets --output-folder /data/bpb_sets
"""

import argparse
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

PALOMA_CORE = [
    "c4_en",
    "c4_100_domains",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "redpajama",
]
PALOMA_LEGACY = ["ptb", "wikitext_103"]
OPENWEBMATH_FILES = (
    "hf://datasets/EleutherAI/proof-pile-2/open-web-math/validation/*.jsonl.zst"
)


def load_sets() -> dict[str, tuple[list[str], bool]]:
    """Set name -> (documents, included in its prefix's average)."""
    sets: dict[str, tuple[list[str], bool]] = {}
    for source in PALOMA_CORE + PALOMA_LEGACY:
        texts = load_dataset("allenai/paloma", source, split="val")["text"]
        sets[f"paloma/{source}"] = (texts, source in PALOMA_CORE)
    owm = load_dataset(
        "json", data_files={"validation": OPENWEBMATH_FILES}, split="validation"
    )
    sets["openwebmath"] = (owm["text"], True)
    return sets


def sample_slice(texts: list[str], *, max_bytes: int, seed: int) -> list[str]:
    """A prefix of the documents in a fixed random order, up to ``max_bytes``.

    Whole documents only. The slice ends at the first document that would cross
    the cap; skipping it and filling the rest with shorter documents would bias
    the slice toward short documents, whose BPB is higher. The first document
    is always kept, so no slice is empty.
    """
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    chosen: list[str] = []
    used = 0
    for index in order:
        size = len(texts[index].encode("utf-8"))
        if chosen and used + size > max_bytes:
            break
        chosen.append(texts[index])
        used += size
    return chosen


def num_bytes(texts: list[str]) -> int:
    return sum(len(t.encode("utf-8")) for t in texts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--max-bytes", type=int, default=8_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = []
    for name, (texts, in_average) in load_sets().items():
        sliced = sample_slice(texts, max_bytes=args.max_bytes, seed=args.seed)
        path = os.path.join("inloop", name.replace("/", "_") + ".parquet")
        os.makedirs(os.path.join(args.output_folder, "inloop"), exist_ok=True)
        pq.write_table(
            pa.table({"text": sliced}), os.path.join(args.output_folder, path)
        )
        slice_bytes, full_bytes = num_bytes(sliced), num_bytes(texts)
        manifest.append(
            {
                "name": name,
                # Relative to the output folder (see bpb_sets_from_manifest).
                "path": path,
                "in_average": in_average,
                "num_documents": len(sliced),
                "num_bytes": slice_bytes,
                "full_num_documents": len(texts),
                "full_num_bytes": full_bytes,
            }
        )
        print(
            f"{name}: {len(sliced)}/{len(texts)} documents, "
            f"{slice_bytes / 1e6:.1f}/{full_bytes / 1e6:.1f} MB in the loop"
        )
    with open(os.path.join(args.output_folder, "manifest.json"), "w") as f:
        json.dump(
            {"max_bytes": args.max_bytes, "seed": args.seed, "sets": manifest},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
