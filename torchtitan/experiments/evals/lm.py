# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""lm-evaluation-harness backend that scores requests with a torchtitan model.

Requests are packed into fixed-size 1D token buffers in the layout the training
collator produces (documents concatenated, positions reset per document, a
padding tail). Scoring therefore goes through the model's own
``preprocess_inputs`` and attention masking, and the fixed buffer size keeps
flex attention from recompiling for every batch shape.

lm-eval splits requests across data-parallel ranks and gathers the metrics.
Within a data-parallel rank the model may be tensor parallel; its TP ranks
see the same requests, so they issue identical forward passes. Under TP the
output layer returns each rank's slice of the vocabulary, and scores are
combined across the slices the way the loss-parallel cross-entropy in
components/loss.py does, without gathering the full vocabulary.
"""

from typing import Any

import torch
import torch.distributed as dist
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
from tqdm import tqdm

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model import BaseModel


# Target rows per lm_head call: bounds the fp32 logits to
# _LOGITS_CHUNK_ROWS x vocab (about 2.5 GB for a 150k vocabulary).
_LOGITS_CHUNK_ROWS = 4096

# One scored sequence: (input tokens, target tokens). The targets are the
# tokens predicted at the last len(targets) input positions.
Sequence = tuple[list[int], list[int]]


def rolling_windows(
    tokens: list[int], *, prefix_token: int, max_length: int
) -> list[tuple[list[int], list[int]]]:
    """lm-eval's rolling windows over a document: (context, continuation) pairs.

    The continuations are disjoint and cover every token once. The first
    window is conditioned on ``prefix_token`` (BOS), each later one on the
    tokens before it, up to ``max_length`` tokens in total.
    """
    return list(
        map(
            make_disjoint_window,
            get_rolling_token_windows(
                token_list=tokens,
                prefix_token=prefix_token,
                max_seq_len=max_length,
                context_len=1,
            ),
        )
    )


def to_sequence(
    context: list[int], continuation: list[int], *, max_length: int
) -> Sequence:
    """(input tokens, targets) for one request, left-truncated to ``max_length``."""
    # An empty context would shift every target onto the wrong logit row.
    if not context:
        raise ValueError("The context must hold at least one token.")
    if len(continuation) > max_length:
        raise ValueError(
            f"Continuation of {len(continuation)} tokens exceeds "
            f"max_context_length ({max_length})."
        )
    # The last token is only ever a target.
    tokens = (context + continuation)[-(max_length + 1) :]
    return tokens[:-1], continuation


def pack_sequences(
    order: list[int],
    sequences: list[Sequence],
    *,
    num_tokens_per_batch: int,
    max_num_documents: int | None = None,
) -> list[list[int]]:
    """Group sequence indices, taken in ``order``, into buffers that fit."""
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index in order:
        length = len(sequences[index][0])
        full = used + length > num_tokens_per_batch or (
            max_num_documents is not None and len(current) == max_num_documents
        )
        if current and full:
            batches.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    if current:
        batches.append(current)
    return batches


def build_buffer(
    sequences: list[Sequence], *, size: int, max_context_length: int
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    """One packed buffer in the training collator's layout.

    Returns the batch (``labels`` all ignored), the flat indices of the rows
    whose logits predict a target, and the target token ids.
    """
    input_ids = torch.zeros(size, dtype=torch.int64)
    positions = torch.zeros(size, dtype=torch.int64)
    padding_mask = torch.ones(size, dtype=torch.bool)
    target_rows: list[int] = []
    target_ids: list[int] = []
    start = 0
    for inputs, targets in sequences:
        end = start + len(inputs)
        input_ids[start:end] = torch.tensor(inputs)
        positions[start:end] = torch.arange(len(inputs))
        padding_mask[start:end] = False
        target_rows.extend(range(end - len(targets), end))
        target_ids.extend(targets)
        start = end
    if start < size:
        # Same padding layout as TextCollator.
        positions[start:] = torch.arange(size - start) % max_context_length
    batch = {
        "input": input_ids,
        "labels": torch.full((size,), IGNORE_INDEX, dtype=torch.int64),
        "positions": positions,
        "padding_mask": padding_mask,
    }
    return batch, target_rows, target_ids


class TrainstationLM(TemplateLM):
    """Log-likelihood scoring of lm-eval requests with a torchtitan model.

    Args:
        model: Model in eval mode on ``device``, possibly tensor parallel.
        tokenizer: The tokenizer the model was trained with.
        parallel_dims: Parallel dims of the eval job: data-parallel replicas,
            each optionally tensor parallel.
        parallelism: Parallelism config passed through to ``preprocess_inputs``.
        max_context_length: Maximum length of one scored sequence.
        num_tokens_per_batch: Size of each packed token buffer.
        max_num_documents: Cap on sequences per buffer, required by varlen
            attention configs that set ``dataloader.max_num_documents``.
        device: Device the model lives on.
    """

    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_context_length: int,
        num_tokens_per_batch: int,
        max_num_documents: int | None = None,
        device: torch.device,
    ) -> None:
        super().__init__()
        if tokenizer.eos_id is None:
            raise ValueError("lm-eval scoring requires a tokenizer with an EOS token.")
        if num_tokens_per_batch < max_context_length:
            raise ValueError(
                f"num_tokens_per_batch ({num_tokens_per_batch}) must be at least "
                f"max_context_length ({max_context_length})."
            )
        self.model = model
        # Forward returns final hidden states; see _score_buffer.
        self.model._skip_lm_head = True
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.parallelism = parallelism
        self.max_context_length = max_context_length
        self.num_tokens_per_batch = num_tokens_per_batch
        self.max_num_documents = max_num_documents
        self._device = device
        self._forward_context = dist_utils.get_spmd_context(parallel_dims=parallel_dims)
        self._tp_group = (
            parallel_dims.get_mesh("tp").get_group()
            if parallel_dims.tp_enabled
            else None
        )
        self._vocab_start: int | None = None
        self._dp_group = None
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            self._rank = batch_mesh.get_local_rank()
            self._world_size = batch_mesh.size()
            self._dp_group = batch_mesh.get_group()

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def prefix_token_id(self) -> int:
        # Training documents start with BOS (see hf_datasets/text_datasets.py),
        # so condition empty contexts and rolling windows on BOS as well.
        bos_id = getattr(self.tokenizer, "bos_id", None)
        return bos_id if bos_id is not None else self.eot_token_id

    @property
    def max_length(self) -> int:
        return self.max_context_length

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        add_bos = True if add_special_tokens is None else add_special_tokens
        return self.tokenizer.encode(string, add_bos=add_bos, add_eos=False)

    def loglikelihood_rolling(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[float]:
        windows: list[tuple[None, list[int], list[int]]] = []
        owners: list[int] = []
        for index, (string,) in enumerate(req.args for req in requests):
            tokens = self.tok_encode(string, add_special_tokens=False)
            for context, continuation in rolling_windows(
                tokens, prefix_token=self.prefix_token_id, max_length=self.max_length
            ):
                windows.append((None, context, continuation))
                owners.append(index)

        totals = [0.0] * len(requests)
        scores = self._loglikelihood_tokens(windows, disable_tqdm=disable_tqdm)
        for index, (logprob, _) in zip(owners, scores, strict=True):
            totals[index] += logprob
        return totals

    def generate_until(self, requests: list[Any], disable_tqdm: bool = False):
        raise NotImplementedError(
            "TrainstationLM only supports log-likelihood tasks; generative "
            "tasks are not supported yet."
        )

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[Any, list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        # A whitespace-only context encodes to no tokens with a tokenizer that
        # adds no BOS (_encode_pair moves the whitespace onto the
        # continuation); condition it on the prefix token, as lm-eval does for
        # an empty context.
        sequences = [
            to_sequence(
                context_enc or [self.prefix_token_id],
                continuation_enc,
                max_length=self.max_length,
            )
            for _, context_enc, continuation_enc in requests
        ]

        # Longest first so buffers fill tightly.
        order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i][0]))
        results: list[tuple[float, bool] | None] = [None] * len(sequences)
        progress = tqdm(
            total=len(sequences),
            disable=disable_tqdm or self.rank != 0,
            desc="Scoring requests",
        )
        for batch in self._pack(order, sequences):
            for index, result in zip(
                batch, self._score_buffer([sequences[i] for i in batch]), strict=True
            ):
                results[index] = result
            progress.update(len(batch))
        progress.close()
        return results  # pyrefly: ignore [bad-return]

    def _pack(self, order: list[int], sequences: list[Sequence]) -> list[list[int]]:
        return pack_sequences(
            order,
            sequences,
            num_tokens_per_batch=self.num_tokens_per_batch,
            max_num_documents=self.max_num_documents,
        )

    @torch.no_grad()
    def _score_buffer(self, sequences: list[Sequence]) -> list[tuple[float, bool]]:
        # target_rows: flat indices of the logits that predict each target.
        batch, target_rows, target_ids = build_buffer(
            sequences,
            size=self.num_tokens_per_batch,
            max_context_length=self.max_context_length,
        )
        batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}
        rows = torch.tensor(target_rows, dtype=torch.int64, device=self._device)
        ids = torch.tensor(target_ids, dtype=torch.int64, device=self._device)
        with self._forward_context():
            inputs, _, extra_kwargs = self.model.preprocess_inputs(
                batch,
                parallel_dims=self.parallel_dims,
                parallelism=self.parallelism,
                max_num_documents=self.max_num_documents,
                max_context_length=self.max_context_length,
            )
            # Like ChunkedLossWrapper: take the final hidden states, which are
            # replicated over TP, and apply lm_head to the target rows only,
            # a chunk of rows at a time. Full logits for a buffer (tokens x
            # vocab) do not fit for long contexts and large vocabularies, and
            # rolling-window requests make every row a target.
            hidden = self.model(inputs, **extra_kwargs)
            token_logprobs, is_greedy = [], []
            for chunk_rows, chunk_ids in zip(
                rows.split(_LOGITS_CHUNK_ROWS), ids.split(_LOGITS_CHUNK_ROWS)
            ):
                logits = self.model.lm_head(hidden[chunk_rows]).float()
                logprob, greedy = self._score_logits(logits, chunk_ids)
                token_logprobs.append(logprob)
                is_greedy.append(greedy)

        results = []
        offset = 0
        token_logprobs = torch.cat(token_logprobs).tolist()
        is_greedy = torch.cat(is_greedy).tolist()
        for _, targets in sequences:
            span = slice(offset, offset + len(targets))
            results.append((sum(token_logprobs[span]), all(is_greedy[span])))
            offset += len(targets)
        return results

    def _score_logits(
        self, logits: torch.Tensor, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probability of each target id, and whether it is the argmax.

        Without TP, ``logits`` covers the whole vocabulary. Under TP it is this
        rank's slice [vocab_start, vocab_start + local_size) and the softmax
        is combined across ranks with all-reduces.
        """
        if self._tp_group is None:
            logprob = logits.gather(-1, ids.unsqueeze(-1)).squeeze(
                -1
            ) - logits.logsumexp(dim=-1)
            return logprob, logits.argmax(dim=-1) == ids

        group = self._tp_group
        local_size = logits.shape[-1]
        vocab_start = self._tp_vocab_start(local_size)

        local_max, local_argmax = logits.max(dim=-1)
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
        sumexp = torch.exp(logits - global_max.unsqueeze(-1)).sum(dim=-1)
        dist.all_reduce(sumexp, op=dist.ReduceOp.SUM, group=group)

        # Only the rank whose slice holds the target contributes its logit.
        in_slice = (ids >= vocab_start) & (ids < vocab_start + local_size)
        local_ids = (ids - vocab_start).clamp(0, local_size - 1)
        target_logit = torch.where(
            in_slice, logits.gather(-1, local_ids.unsqueeze(-1)).squeeze(-1), 0.0
        )
        dist.all_reduce(target_logit, op=dist.ReduceOp.SUM, group=group)
        logprob = target_logit - global_max - sumexp.log()

        # The global argmax is the smallest vocabulary index holding the global
        # max, matching torch.argmax on the full logits.
        no_max = torch.iinfo(torch.int64).max
        argmax = torch.where(
            local_max == global_max, local_argmax + vocab_start, no_max
        )
        dist.all_reduce(argmax, op=dist.ReduceOp.MIN, group=group)
        return logprob, argmax == ids

    def _tp_vocab_start(self, local_size: int) -> int:
        """First vocabulary index of this rank's lm_head slice (slices may be uneven)."""
        if self._vocab_start is None:
            group = self._tp_group
            sizes = [0] * dist.get_world_size(group)
            dist.all_gather_object(sizes, local_size, group=group)
            self._vocab_start = sum(sizes[: dist.get_rank(group)])
        return self._vocab_start

    # Distributed primitives used by lm_eval.evaluator.evaluate(), over the
    # data-parallel group (lm-eval's ranks).

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self._dp_group)
        return torch.stack(gathered) if tensor.dim() == 0 else torch.cat(gathered)

    def gather_object(self, obj: Any, dst: int = 0) -> list[Any] | None:
        gathered = [None] * self.world_size if self.rank == dst else None
        dist.gather_object(obj, gathered, group=self._dp_group, group_dst=dst)
        return gathered

    def barrier(self) -> None:
        dist.barrier(group=self._dp_group)
