# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bits per byte on held-out text sets, inside the training loop.

``BPBValidator`` extends the core ``Validator``: it still reports the
training-layout validation loss, then scores each configured text set the way
the eval job scores lm-eval's ``bits_per_byte`` tasks
(``TrainstationLM.loglikelihood_rolling``):

- each document on its own, conditioned on BOS;
- lm-eval's rolling windows for documents longer than the context;
- no end-of-document token is scored;
- BPB = summed negative log-likelihood in nats / (ln 2 x UTF-8 bytes of the text).

The windows are packed into buffers in the training collator's layout, with
labels only on the scored tokens, and run through the trainer's own forward
pass and loss function. Every data-parallel rank runs the same number of
forward passes (short ranks run fully ignored buffers), so FSDP, TP and EP
communicate in lockstep. Pipeline and context parallelism are not supported.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import cast

import pyarrow.parquet as pq
import torch
import torch.nn as nn

from torchtitan.components.validate import Validator
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.evals.lm import (
    build_buffer,
    pack_sequences,
    rolling_windows,
    Sequence,
    to_sequence,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.tools import utils

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class BPBSet:
    """One held-out text set scored in the loop."""

    name: str
    """Metric name, e.g. ``paloma/c4_en``. Logged as
    ``validation_metrics/<name>/bpb`` and ``validation_metrics/<name>/loss``."""

    path: str
    """Parquet or JSON Lines file with a ``text`` column (see prepare_bpb_sets.py)."""

    in_average: bool = True
    """Include this set in ``<prefix>/avg``, the unweighted mean BPB over the
    sets whose names share the prefix before the first ``/``."""


@dataclass
class _PreparedSet:
    name: str
    in_average: bool
    sequences: list[Sequence]
    """This rank's share of the set's rolling windows."""
    buffers: list[list[int]]
    """This rank's packed buffers, as indices into ``sequences``."""
    num_buffers: int
    """Buffers per rank: the maximum over data-parallel ranks."""
    num_bytes: int
    """UTF-8 bytes of the whole set's text."""
    num_targets: int
    """Scored tokens of the whole set."""


def bpb_sets_from_manifest(folder: str) -> list[BPBSet]:
    """The in-loop sets written by prepare_bpb_sets.py into ``folder``.

    Paths in the manifest are relative to ``folder``, so it can be moved.
    """
    with open(os.path.join(folder, "manifest.json")) as f:
        manifest = json.load(f)
    return [
        BPBSet(
            name=s["name"],
            path=os.path.join(folder, s["path"]),
            in_average=s["in_average"],
        )
        for s in manifest["sets"]
    ]


def read_texts(path: str) -> list[str]:
    """The ``text`` column of a Parquet or JSON Lines file."""
    if path.endswith(".parquet"):
        return pq.read_table(path, columns=["text"]).column("text").to_pylist()
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line)["text"] for line in f if line.strip()]
    raise ValueError(f"Expected a .parquet or .jsonl file, got {path}")


class BPBValidator(Validator):
    """The core validator plus bits per byte on held-out text sets."""

    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        sets: list[BPBSet] = field(default_factory=list)
        """Held-out text sets to score at every validation."""

        training_loss: bool = True
        """Also report the training-layout validation loss on ``dataloader``,
        as the core validator does."""

        def __post_init__(self) -> None:
            Validator.Config.__post_init__(self)
            if not self.sets:
                raise ValueError("BPBValidator needs at least one set in `sets`.")
            names = [s.name for s in self.sets]
            if len(set(names)) != len(names):
                raise ValueError(f"BPBValidator set names must be unique: {names}")
            # <prefix>/avg is the logged average of the sets under <prefix>.
            if any(name.endswith("/avg") for name in names):
                raise ValueError(f"BPBValidator set names cannot end in /avg: {names}")
            paths = [s.path for s in self.sets]
            if not all(p.endswith((".parquet", ".jsonl")) for p in paths):
                raise ValueError(
                    f"BPBValidator sets must be .parquet or .jsonl: {paths}"
                )

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        parallel_dims = self.parallel_dims
        if parallel_dims.pp_enabled or parallel_dims.cp_enabled:
            raise ValueError(
                "BPBValidator does not support pipeline or context parallelism."
            )
        bos_id = getattr(self.tokenizer, "bos_id", None)
        # Same conditioning token as TrainstationLM.prefix_token_id.
        self.prefix_token = bos_id if bos_id is not None else self.tokenizer.eos_id
        if self.prefix_token is None:
            raise ValueError(
                "BPBValidator requires a tokenizer with a BOS or EOS token."
            )
        # The sets are read at the first validation; fail before training.
        missing = [s.path for s in config.sets if not os.path.isfile(s.path)]
        if missing:
            raise ValueError(f"BPBValidator set files not found: {missing}")
        self.sets = config.sets
        self.training_loss = config.training_loss
        # Passed to preprocess_inputs like the trainer does; varlen attention
        # needs it.
        self.max_num_documents = self.dl_config.max_num_documents
        self._prepared: list[_PreparedSet] | None = None
        # The device the core validator and the trainer put batches on, not the
        # parameters' device, which is the CPU under FSDP CPU offload.
        self._device = torch.device(utils.device_type)

    def _prepare_set(self, bpb_set: BPBSet) -> _PreparedSet:
        texts = read_texts(bpb_set.path)
        windows: list[Sequence] = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_bos=False, add_eos=False)
            for context, continuation in rolling_windows(
                tokens, prefix_token=self.prefix_token, max_length=self.seq_len
            ):
                windows.append(
                    to_sequence(context, continuation, max_length=self.seq_len)
                )
        if not windows:
            raise ValueError(f"BPB set {bpb_set.name} has no text to score.")
        # Windows, not documents, are dealt to data-parallel ranks round robin,
        # longest first: some sets have fewer documents than ranks (Paloma PTB
        # is one document). Every rank tokenizes the whole set once.
        order = sorted(range(len(windows)), key=lambda i: -len(windows[i][0]))
        sequences = [windows[i] for i in order[self.dp_rank :: self.dp_world_size]]
        buffers = pack_sequences(
            list(range(len(sequences))),
            sequences,
            num_tokens_per_batch=self.num_tokens_per_batch,
            max_num_documents=self.max_num_documents,
        )
        return _PreparedSet(
            name=bpb_set.name,
            in_average=bpb_set.in_average,
            sequences=sequences,
            buffers=buffers,
            num_buffers=int(
                dist_utils.dist_max(
                    torch.tensor(len(buffers), device=self._device),
                    self.parallel_dims.get_optional_mesh("batch"),
                )
            ),
            num_bytes=sum(len(text.encode("utf-8")) for text in texts),
            num_targets=sum(len(targets) for _, targets in windows),
        )

    def _score_set(self, model: nn.Module, prepared: _PreparedSet) -> float:
        """Summed negative log-likelihood of the set, over all ranks, in nats."""
        # float64: tens of millions of per-token losses are summed.
        total = torch.zeros((), dtype=torch.float64, device=self._device)
        for index in range(prepared.num_buffers):
            members = prepared.buffers[index] if index < len(prepared.buffers) else []
            batch, rows, ids = build_buffer(
                [prepared.sequences[i] for i in members],
                size=self.num_tokens_per_batch,
                max_context_length=self.seq_len,
            )
            batch["labels"][rows] = torch.tensor(ids, dtype=torch.int64)
            batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}
            inputs, labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
                batch,
                parallel_dims=self.parallel_dims,
                parallelism=self.parallelism,
                max_num_documents=self.max_num_documents,
                max_context_length=self.seq_len,
            )
            with self.validation_context():
                predictions = model(inputs, **extra_kwargs)
                loss_sum, _ = self.loss_fn(predictions, labels)
            total += loss_sum.detach().double()
        if self.parallel_dims.dp_cp_enabled:
            return dist_utils.dist_sum(
                total, self.parallel_dims.get_optional_mesh("loss")
            )
        return float(total.item())

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        if self.training_loss:
            super().validate(model_parts, step)
        assert len(model_parts) == 1
        model = model_parts[0]
        if self._prepared is None:
            self._prepared = [self._prepare_set(s) for s in self.sets]
        model.eval()
        metrics: dict[str, float] = {}
        averages: dict[str, list[float]] = {}
        for prepared in self._prepared:
            nll = self._score_set(model, prepared)
            bpb = nll / (math.log(2) * prepared.num_bytes)
            metrics[f"validation_metrics/{prepared.name}/bpb"] = bpb
            metrics[f"validation_metrics/{prepared.name}/loss"] = (
                nll / prepared.num_targets
            )
            if prepared.in_average and "/" in prepared.name:
                averages.setdefault(prepared.name.split("/", 1)[0], []).append(bpb)
        # Unweighted mean over the sets sharing a prefix, e.g. paloma/avg.
        for prefix, values in averages.items():
            if len(values) > 1:
                metrics[f"validation_metrics/{prefix}/avg/bpb"] = sum(values) / len(
                    values
                )
        self.metrics_processor.logger.log(metrics, step)
        logger.info(
            "BPB at step %d: %s",
            step,
            ", ".join(
                f"{k.removeprefix('validation_metrics/').removesuffix('/bpb')}={v:.4f}"
                for k, v in metrics.items()
                if k.endswith("/bpb")
            ),
        )
        model.train()
