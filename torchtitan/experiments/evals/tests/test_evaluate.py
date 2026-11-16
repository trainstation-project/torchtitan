# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("lm_eval")

from lm_eval.tasks import TaskManager

from torchtitan.experiments.evals.evaluate import (
    expand_tasks,
    TASKS_DIR,
    write_on_rank0,
)

QUICK = [
    "trainstation_paloma",
    "trainstation_paloma_ptb",
    "trainstation_paloma_wikitext_103",
    "trainstation_openwebmath",
    "trainstation_mc5",
    "trainstation_mmlu_cloze",
    "lambada_openai",
    "triviaqa_gold_bpb",
    "gsm8k_gold_bpb",
    "math500_gold_bpb",
    "humaneval_gold_bpb",
    "mbpp_gold_bpb",
]


@pytest.fixture(scope="module")
def task_manager():
    return TaskManager(include_path=TASKS_DIR)


def test_expand_tasks(task_manager):
    # Suites expand recursively into their members. Groups that aggregate
    # (trainstation_paloma, trainstation_mc5, the MMLU groups) stay whole.
    assert expand_tasks(["trainstation_quick"], task_manager) == QUICK
    assert expand_tasks(["trainstation_full"], task_manager) == QUICK + [
        "trainstation_mmlu"
    ]
    # A task listed twice, directly and through a suite, is kept once.
    assert expand_tasks(["gsm8k_gold_bpb", "trainstation_quick"], task_manager) == [
        "gsm8k_gold_bpb"
    ] + [unit for unit in QUICK if unit != "gsm8k_gold_bpb"]
    assert expand_tasks(["mmlu"], task_manager) == ["mmlu"]


def _write_worker(rank: int, init_file: str, path: str, errors_dir: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        write_on_rank0(path, {"step": 1})
    except RuntimeError as e:
        with open(os.path.join(errors_dir, f"rank{rank}"), "w") as f:
            f.write(str(e))
    dist.destroy_process_group()


def test_failed_write_raises_on_every_rank():
    with tempfile.TemporaryDirectory() as tmp:
        # The parent of the results folder is a file, so makedirs fails on rank 0.
        blocker = os.path.join(tmp, "not_a_dir")
        open(blocker, "w").close()
        path = os.path.join(blocker, "step-1", "hellaswag.json")
        mp.spawn(
            _write_worker,
            args=(os.path.join(tmp, "rdzv"), path, tmp),
            nprocs=2,
            join=True,
        )
        for rank in range(2):
            with open(os.path.join(tmp, f"rank{rank}")) as f:
                assert "Rank 0 failed to write" in f.read()
