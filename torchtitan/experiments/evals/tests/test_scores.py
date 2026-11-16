# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gold BPB from lm-eval samples, and the per-step composites."""

import json
import math

import pytest

from torchtitan.experiments.evals import scores


def _sample(target, choices: list[str], logprobs: list[float], doc=None):
    """A multiple-choice sample in lm-eval's in-memory layout."""
    return {
        "target": target,
        "doc": doc or {},
        "arguments": [("Question: q\nAnswer:", c) for c in choices],
        "resps": [[(lp, False)] for lp in logprobs],
    }


def test_gold_index_forms():
    assert scores.gold_index(_sample(2, [" a", " b", " c"], [0, 0, 0])) == 2
    # A string target is the text of its choice: a letter or the answer.
    assert scores.gold_index(_sample("C", [" A", " B", " C"], [0, 0, 0])) == 2
    assert scores.gold_index(_sample("pear", [" apple", " pear"], [0, 0])) == 1
    # Numeric choices: lm-eval already made an index target an int, so a
    # string "2" is the choice " 2", not index 2.
    assert scores.gold_index(_sample("2", [" 1", " 2", " 3"], [0, 0, 0])) == 1
    # WinoGrande: the target is the shared continuation; the answer is 1-based,
    # even when the continuation is a number.
    winogrande = _sample(
        " 1990.",
        [" 1990.", " 1990."],
        [0, 0],
        doc={"option1": "a", "option2": "b", "answer": "2"},
    )
    assert scores.gold_index(winogrande) == 1


@pytest.mark.parametrize("target", [3, -1, True, "1", "plum"])
def test_gold_index_rejects_unknown_targets(target):
    with pytest.raises(ValueError):
        scores.gold_index(_sample(target, [" a", " b", " c"], [0, 0, 0]))


def test_scored_text_takes_the_contexts_trailing_whitespace():
    # lm-eval's _encode_pair moves it onto the continuation before scoring.
    assert scores.scored_text("[BEGIN]\n", "code") == "\ncode"
    assert scores.scored_text("Answer:", " cat") == " cat"
    assert scores.scored_text("", " cat") == " cat"


def test_gold_bpb_uses_the_correct_choice():
    samples = [
        _sample(1, [" dog", " cat"], [-1.0, -2.0]),
        _sample(0, [" a red apple", " pear"], [-3.0, -0.5]),
    ]
    nats, num_bytes = scores.task_gold_nats_and_bytes(samples)
    # The correct choices are " cat" (4 bytes) and " a red apple" (12 bytes).
    assert nats == pytest.approx(5.0)
    assert num_bytes == 16


def test_gold_bpb_counts_the_moved_whitespace():
    sample = _sample(0, ["x = 1", "y = 2"], [-4.0, -1.0])
    sample["arguments"] = [("[BEGIN]\n", c) for c in ("x = 1", "y = 2")]
    # "\nx = 1" is scored: 6 bytes.
    assert scores.task_gold_nats_and_bytes([sample]) == (4.0, 6)


@pytest.mark.parametrize("labels", [[" A", " B", " C"], [" (A)", " (B)"], ["A.", "B."]])
def test_lettered_tasks_get_no_gold_bpb(labels):
    samples = [_sample(1, labels, [-1.0] * len(labels))]
    assert scores.task_gold_nats_and_bytes(samples) is None


def test_single_digit_answers_still_get_gold_bpb():
    samples = [_sample(1, [" 3", " 4"], [-1.0, -2.0])]
    assert scores.task_gold_nats_and_bytes(samples) == (2.0, 2)


def test_add_gold_bpb_tasks_and_weighted_group():
    t2 = _sample(1, [" a", " bcdefg"], [-1.0, -6.0])  # 6 nats, 7 bytes
    results = {
        "results": {"group": {}, "t1": {}, "t2": {}, "lettered": {}, "gen": {}},
        "group_subtasks": {"group": ["t1", "t2"]},
        "configs": {
            "t1": {"output_type": "multiple_choice"},
            "t2": {"output_type": "multiple_choice"},
            "lettered": {"output_type": "multiple_choice"},
            "gen": {"output_type": "loglikelihood"},
        },
        "samples": {
            "t1": [_sample(0, [" xx", " yyyy"], [-2.0, -1.0])],  # 2 nats, 3 bytes
            "t2": [t2, t2],
            "lettered": [_sample(0, [" A", " B"], [-1.0, -1.0])],
            "gen": [{"target": "x", "doc": {}, "arguments": [("q", " x")]}],
        },
    }
    scores.add_gold_bpb(results)
    ln2 = math.log(2)
    t1_bpb, t2_bpb = 2 / (ln2 * 3), 6 / (ln2 * 7)
    assert results["results"]["t1"][scores.GOLD_BPB] == pytest.approx(t1_bpb)
    assert results["results"]["t2"][scores.GOLD_BPB] == pytest.approx(t2_bpb)
    # The group weights its tasks' gold BPB by their number of questions.
    assert results["results"]["group"][scores.GOLD_BPB] == pytest.approx(
        (t1_bpb + 2 * t2_bpb) / 3
    )
    assert scores.GOLD_BPB not in results["results"]["lettered"]
    assert scores.GOLD_BPB not in results["results"]["gen"]


def _write_unit(folder, name, results):
    (folder / f"{name}.json").write_text(json.dumps({"results": results}))


def test_summary_needs_every_member(tmp_path):
    units = ["gsm8k_gold_bpb", "math500_gold_bpb", "trainstation_paloma"]
    _write_unit(tmp_path, "gsm8k_gold_bpb", {"gsm8k_gold_bpb": {scores.BPB: 0.4}})
    summary = scores.summarize(scores.step_metrics(str(tmp_path), units))
    assert "math_gold_bpb" not in summary

    _write_unit(tmp_path, "math500_gold_bpb", {"math500_gold_bpb": {scores.BPB: 0.2}})
    _write_unit(
        tmp_path,
        "trainstation_paloma",
        {"trainstation_paloma": {scores.BPB: 0.7}},
    )
    # Clean-subset results and other suites' files in the folder are not read.
    _write_unit(tmp_path, "gsm8k_gold_bpb.clean", {"gsm8k_gold_bpb": {scores.BPB: 9}})
    _write_unit(tmp_path, "zz_milestone", {"math500_gold_bpb": {scores.BPB: 9}})

    summary = scores.summarize(scores.step_metrics(str(tmp_path), units))
    assert summary["math_gold_bpb"] == pytest.approx(0.3)
    assert summary["paloma_bpb"] == pytest.approx(0.7)
    assert "knowledge_gold_bpb" not in summary


def test_cloze_accuracy_prefers_acc_norm(tmp_path):
    results = {
        task: {"acc,none": 0.5, "acc_norm,none": 0.6} for task in scores.CLOZE_ACCURACY
    }
    # WinoGrande and Social IQa report acc only.
    results["winogrande"] = {"acc,none": 0.7}
    _write_unit(tmp_path, "mc", results)
    summary = scores.summarize(scores.step_metrics(str(tmp_path), ["mc"]))
    expected = (0.6 * (len(scores.CLOZE_ACCURACY) - 1) + 0.7) / len(
        scores.CLOZE_ACCURACY
    )
    assert summary["cloze_acc"] == pytest.approx(expected)


def test_step_summary_merges_across_suites(tmp_path):
    _write_unit(tmp_path, "gsm8k_gold_bpb", {"gsm8k_gold_bpb": {scores.BPB: 0.4}})
    _write_unit(tmp_path, "math500_gold_bpb", {"math500_gold_bpb": {scores.BPB: 0.2}})
    quick = ["gsm8k_gold_bpb", "math500_gold_bpb"]
    summary = scores.step_summary(str(tmp_path), quick)
    (tmp_path / scores.SUMMARY_FILE).write_text(json.dumps({"summary": summary}))
    # A later suite on the same step adds its composites and keeps the others.
    _write_unit(
        tmp_path, "trainstation_paloma", {"trainstation_paloma": {scores.BPB: 0.7}}
    )
    merged = scores.step_summary(str(tmp_path), ["trainstation_paloma"])
    assert merged["math_gold_bpb"] == pytest.approx(0.3)
    assert merged["paloma_bpb"] == pytest.approx(0.7)


def test_generative_gold_bpb_counts_the_scored_text():
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(scores.__file__), "tasks", "gold_bpb", "gold_bpb_utils.py"
    )
    spec = importlib.util.spec_from_file_location("gold_bpb_utils", path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    loglikelihood = -math.log(2) * 10

    def bpb(result: dict) -> float:
        ll, num_bytes = result["bits_per_byte"]
        return -ll / (math.log(2) * num_bytes)

    result = [(loglikelihood, True)]
    # " " + answer: lm-eval adds no target_delimiter to the request.
    assert bpb(utils.gsm8k_results({"answer": "4"}, result)) == pytest.approx(10 / 2)
    assert bpb(utils.math_results({"solution": "x=1"}, result)) == pytest.approx(10 / 4)
    doc = {"answer": {"value": "Paris"}}
    assert bpb(utils.triviaqa_results(doc, result)) == pytest.approx(10 / 6)
    # The MBPP prompt ends in "[BEGIN]\n"; the newline moves onto the code.
    assert bpb(utils.mbpp_results({"code": "pass"}, result)) == pytest.approx(
        10 / len("\npass\n[DONE]")
    )
    # HumanEval's prompt ends in whitespace, which moves onto the solution.
    doc = {"prompt": "def f():\n    ", "canonical_solution": "return 1\n"}
    assert bpb(utils.humaneval_results(doc, result)) == pytest.approx(
        10 / len("\n    return 1\n")
    )
