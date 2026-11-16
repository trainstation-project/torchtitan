# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BPBValidator must report the BPB the eval job's rolling scorer reports."""

import contextlib
import json
import math
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.evals.lm import TrainstationLM
from torchtitan.experiments.evals.validator import (
    bpb_sets_from_manifest,
    BPBSet,
    BPBValidator,
)
from torchtitan.models.llama3 import model_registry
from torchtitan.tools import utils

TOKENIZER_PATH = "./tests/assets/tokenizer"
MAX_CONTEXT_LENGTH = 64
NUM_TOKENS_PER_BATCH = 4 * MAX_CONTEXT_LENGTH

SETS = {
    "paloma/web": [
        "A short document.",
        "Rolling windows cover long documents. " * 12,
        "Another page, with punctuation; numbers like 3.14 and café.",
    ],
    "paloma/code": ["def f(x):\n    return x + 1\n" * 5],
    "paloma/legacy": ["An old benchmark document."],
    "openwebmath": ["Let $x^2 = 4$. Then $x = \\pm 2$. " * 4],
}


class _Logger:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int]] = []

    def log(self, metrics: dict, step: int) -> None:
        self.logged.append((metrics, step))


@pytest.fixture(scope="module", autouse=True)
def process_group():
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group(
            backend="gloo", init_method=f"file://{tmp}/rdzv", rank=0, world_size=1
        )
        yield
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer(
        HuggingFaceTokenizer.Config(), tokenizer_path=TOKENIZER_PATH
    )


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    model = model_registry("debugmodel", seq_len=MAX_CONTEXT_LENGTH).model.build()
    model.init_weights()
    return model.train()


@pytest.fixture()
def set_files(tmp_path):
    paths = {}
    for name, texts in SETS.items():
        path = tmp_path / f"{name.replace('/', '_')}.jsonl"
        path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
        paths[name] = str(path)
    return paths


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    # The validator runs on the trainer's device type, which falls back to
    # "cuda" on a machine without accelerators.
    monkeypatch.setattr(utils, "device_type", "cpu")


PARALLEL_DIMS = ParallelDims(
    dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
)


def _make_validator(tokenizer, set_files, logger, *, dp_rank=0, dp_world_size=1):
    config = BPBValidator.Config(
        enable=True,
        training_loss=False,
        sets=[
            BPBSet(name=name, path=path, in_average=name != "paloma/legacy")
            for name, path in set_files.items()
        ],
    )
    return config.build(
        parallelism=ParallelismConfig(),
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=PARALLEL_DIMS,
        loss_fn=lambda pred, labels: (cross_entropy_loss(pred, labels), {}),
        validation_context=contextlib.nullcontext,
        metrics_processor=SimpleNamespace(logger=logger),
        seq_len=MAX_CONTEXT_LENGTH,
        num_tokens_per_batch=NUM_TOKENS_PER_BATCH,
    )


def _reference_bpb(model, tokenizer, texts: list[str]) -> float:
    """BPB from TrainstationLM's rolling scorer, as lm-eval's bits_per_byte."""
    lm = TrainstationLM(
        model,
        tokenizer,
        parallel_dims=PARALLEL_DIMS,
        parallelism=ParallelismConfig(),
        max_context_length=MAX_CONTEXT_LENGTH,
        num_tokens_per_batch=NUM_TOKENS_PER_BATCH,
        device=torch.device("cpu"),
    )
    try:
        requests = [
            Instance("loglikelihood_rolling", {}, (text,), i)
            for i, text in enumerate(texts)
        ]
        logprobs = lm.loglikelihood_rolling(requests, disable_tqdm=True)
    finally:
        model._skip_lm_head = False
    num_bytes = sum(len(t.encode("utf-8")) for t in texts)
    return -sum(logprobs) / (math.log(2) * num_bytes)


def test_bpb_matches_rolling_scorer(model, tokenizer, set_files):
    logger = _Logger()
    _make_validator(tokenizer, set_files, logger).validate([model], step=7)

    # The model is handed back in training mode.
    assert model.training
    ((metrics, step),) = logger.logged
    assert step == 7
    try:
        for name, texts in SETS.items():
            expected = _reference_bpb(model.eval(), tokenizer, texts)
            assert metrics[f"validation_metrics/{name}/bpb"] == pytest.approx(
                expected, rel=1e-4
            )
    finally:
        model.train()


def test_average_skips_excluded_sets(model, tokenizer, set_files):
    logger = _Logger()
    _make_validator(tokenizer, set_files, logger).validate([model], step=1)

    ((metrics, _),) = logger.logged
    included = [metrics[f"validation_metrics/paloma/{s}/bpb"] for s in ("web", "code")]
    assert metrics["validation_metrics/paloma/avg/bpb"] == pytest.approx(
        sum(included) / 2
    )
    # A single set has no average of its own.
    assert "validation_metrics/openwebmath/avg/bpb" not in metrics


@pytest.mark.parametrize("name", ["paloma/web", "paloma/code"])
def test_ranks_share_windows(tokenizer, set_files, name):
    # Two data-parallel ranks, prepared without communication (world size 1
    # here). Their windows partition the set; a one-document set ("paloma/code"
    # is one long document) still gives both ranks work.
    shards = [
        _make_validator(
            tokenizer, set_files, _Logger(), dp_rank=rank, dp_world_size=2
        )._prepare_set(BPBSet(name=name, path=set_files[name]))
        for rank in range(2)
    ]
    expected_targets = sum(
        len(tokenizer.encode(t, add_bos=False, add_eos=False)) for t in SETS[name]
    )
    assert all(shard.num_targets == expected_targets for shard in shards)
    assert expected_targets == sum(
        len(targets) for shard in shards for _, targets in shard.sequences
    )
    assert all(shard.sequences for shard in shards)


def test_empty_buffers_add_nothing(model, tokenizer, set_files):
    # A rank with fewer buffers than the slowest rank runs fully ignored
    # buffers to keep the collectives in lockstep; they must score zero.
    validator = _make_validator(tokenizer, set_files, _Logger())
    prepared = validator._prepare_set(
        BPBSet(name="paloma/web", path=set_files["paloma/web"])
    )
    model.eval()
    try:
        with torch.no_grad():
            nll = validator._score_set(model, prepared)
            prepared.num_buffers += 2
            padded = validator._score_set(model, prepared)
        assert padded == pytest.approx(nll, rel=1e-6)
    finally:
        model.train()


def test_manifest_paths_follow_the_folder(tmp_path, set_files):
    folder = tmp_path / "moved"
    (folder / "inloop").mkdir(parents=True)
    (folder / "inloop" / "web.jsonl").write_text(json.dumps({"text": "x"}) + "\n")
    manifest = {
        "sets": [{"name": "paloma/web", "path": "inloop/web.jsonl", "in_average": True}]
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (bpb_set,) = bpb_sets_from_manifest(str(folder))
    assert bpb_set.path == str(folder / "inloop" / "web.jsonl")


def test_rejects_bad_set_names():
    with pytest.raises(ValueError, match="unique"):
        BPBValidator.Config(
            sets=[BPBSet(name="a", path="x.jsonl"), BPBSet(name="a", path="y.jsonl")]
        )
    with pytest.raises(ValueError, match="/avg"):
        BPBValidator.Config(sets=[BPBSet(name="paloma/avg", path="x.jsonl")])
    with pytest.raises(ValueError, match="parquet or .jsonl"):
        BPBValidator.Config(sets=[BPBSet(name="a", path="x.json")])


def test_rejects_missing_set_files(tokenizer, set_files):
    set_files = {**set_files, "paloma/web": "/nonexistent/web.jsonl"}
    with pytest.raises(ValueError, match="not found"):
        _make_validator(tokenizer, set_files, _Logger())


def test_in_loop_slice_is_fixed_and_capped():
    from torchtitan.experiments.evals.prepare_bpb_sets import sample_slice

    texts = [f"document {i} " * (i + 1) for i in range(50)]
    first = sample_slice(texts, max_bytes=500, seed=0)
    assert first == sample_slice(texts, max_bytes=500, seed=0)
    # Every document after the first fits under the cap.
    sizes = [len(t.encode("utf-8")) for t in first]
    assert sum(sizes[1:]) <= 500 - min(sizes[0], 500)
    # A cap smaller than every document still keeps one document.
    assert len(sample_slice(texts, max_bytes=1, seed=0)) == 1


def test_rejects_tokenizer_without_bos_or_eos(set_files):
    tokenizer = SimpleNamespace(bos_id=None, eos_id=None)
    with pytest.raises(ValueError, match="BOS or EOS"):
        _make_validator(tokenizer, set_files, _Logger())


def test_rejects_a_set_with_no_text(tokenizer, set_files, tmp_path):
    validator = _make_validator(tokenizer, set_files, _Logger())
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no text"):
        validator._prepare_set(BPBSet(name="empty", path=str(empty)))


def _two_rank_worker(rank: int, init_file: str, set_files: dict, out: str) -> None:
    import torchtitan.distributed.parallel_dims as parallel_dims_module

    parallel_dims_module.device_type = "cpu"
    utils.device_type = "cpu"
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=2, cp=1, tp=1, pp=1, ep=1, world_size=2
        )
        parallel_dims.build_mesh()
        torch.manual_seed(0)
        model = model_registry("debugmodel", seq_len=MAX_CONTEXT_LENGTH).model.build()
        model.init_weights()
        tokenizer = HuggingFaceTokenizer(
            HuggingFaceTokenizer.Config(), tokenizer_path=TOKENIZER_PATH
        )
        logger = _Logger()
        config = BPBValidator.Config(
            enable=True,
            training_loss=False,
            sets=[BPBSet(name=n, path=p) for n, p in set_files.items()],
        )
        config.build(
            parallelism=ParallelismConfig(),
            dp_world_size=2,
            dp_rank=rank,
            tokenizer=tokenizer,
            parallel_dims=parallel_dims,
            loss_fn=lambda pred, labels: (cross_entropy_loss(pred, labels), {}),
            validation_context=contextlib.nullcontext,
            metrics_processor=SimpleNamespace(logger=logger),
            seq_len=MAX_CONTEXT_LENGTH,
            num_tokens_per_batch=NUM_TOKENS_PER_BATCH,
        ).validate([model], step=1)
        ((metrics, _),) = logger.logged
        with open(f"{out}/rank{rank}.json", "w") as f:
            json.dump(metrics, f)
    finally:
        dist.destroy_process_group()


def test_two_ranks_report_the_single_rank_bpb(model, tokenizer, set_files, tmp_path):
    # "paloma/legacy" is one short document: one window, so rank 1 runs only a
    # fully ignored buffer while rank 0 scores it.
    expected_logger = _Logger()
    _make_validator(tokenizer, set_files, expected_logger).validate([model], step=1)
    ((expected, _),) = expected_logger.logged
    torch.multiprocessing.spawn(
        _two_rank_worker,
        args=(str(tmp_path / "rdzv"), set_files, str(tmp_path)),
        nprocs=2,
    )
    for rank in range(2):
        metrics = json.loads((tmp_path / f"rank{rank}.json").read_text())
        for name in SETS:
            key = f"validation_metrics/{name}/bpb"
            assert metrics[key] == pytest.approx(expected[key], rel=1e-6)
