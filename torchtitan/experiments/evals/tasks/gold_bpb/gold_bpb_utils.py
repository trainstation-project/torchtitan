# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bits per byte of a gold answer, for lm-eval loglikelihood tasks.

``results[0][0]`` is the log-likelihood of ``doc_to_target``, preceded by the
prompt's trailing whitespace, which lm-eval's ``TemplateLM._encode_pair`` moves
onto the continuation. Each byte count covers exactly that text.

lm-eval passes a loglikelihood task's ``doc_to_target`` to the model unchanged
(``target_delimiter`` only joins few-shot examples), so these tasks put the
space after "Answer:" in the target and set ``target_delimiter`` to "", as
lm-eval's lambada does.
"""

from torchtitan.experiments.evals.scores import scored_text


def _bpb(loglikelihood: float, text: str) -> dict:
    return {"bits_per_byte": (loglikelihood, len(text.encode("utf-8")))}


def gsm8k_results(doc: dict, results: list) -> dict:
    return _bpb(results[0][0], " " + doc["answer"])


def math_results(doc: dict, results: list) -> dict:
    return _bpb(results[0][0], " " + doc["solution"])


def humaneval_results(doc: dict, results: list) -> dict:
    # The prompt is the whole context (humaneval_gold_bpb.yaml).
    return _bpb(results[0][0], scored_text(doc["prompt"], doc["canonical_solution"]))


def mbpp_results(doc: dict, results: list) -> dict:
    # The prompt ends in "[BEGIN]\n" (mbpp_gold_bpb.yaml).
    return _bpb(results[0][0], "\n" + doc["code"] + "\n[DONE]")


def triviaqa_results(doc: dict, results: list) -> dict:
    return _bpb(results[0][0], " " + doc["answer"]["value"])
