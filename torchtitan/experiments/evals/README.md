# Pretraining evals

Evaluate the checkpoints of a training run with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
while it trains, without touching the training job and without converting
checkpoints to Hugging Face format.

- **Out of band.** The eval job runs next to the training job on its own GPUs
  and only reads the `step-N` checkpoints training writes. Training is never
  paused or slowed.
- **Native checkpoints.** Each DCP checkpoint is loaded straight into the
  torchtitan model built from the training config, so what is evaluated is
  exactly what was trained. Convert to HF once, for the final model only
  (`scripts/checkpoint_conversion/convert_to_hf.py`).
- **Same numerics as training.** Requests are packed into token buffers in the
  training collator's layout and scored through the model's own
  `preprocess_inputs` and attention masking. On training documents the eval
  path reproduces the trainer's cross-entropy exactly.

## Setup

```bash
pip install -r torchtitan/experiments/evals/requirements.txt
```

This adds lm-eval to the training env; it does not touch the installed torch.

## Usage

Pass the eval options, then `--`, then the exact training arguments (the same
`--module`, `--config` and overrides as the training job, so the eval finds
its `dump_folder` and builds the same model):

```bash
TRAIN_ARGS="--module llama3 --config llama3_8b --dump_folder ./outputs/run7"

# Watch mode: evaluate each checkpoint as it lands; exits once the final
# training step (training.steps) has been evaluated, or after
# --max-idle-hours (default 24) without a new checkpoint.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_quick --watch -- $TRAIN_ARGS

# Backfill: every completed checkpoint, on the tasks it has no results for.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_full -- $TRAIN_ARGS

# Specific steps.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_full --steps 20000 40000 -- $TRAIN_ARGS
```

Results go to `<dump_folder>/evals/step-N/<task>.json`, one file per task
(lm-eval's `results`, plus step, dtype and lm-eval version). Suites are
expanded into their tasks, while groups with an aggregate score, such as
MMLU, keep one file. A task with results is skipped on later runs, so
re-running is cheap, a crashed job can simply be restarted, a task added to a
suite is backfilled on old steps, and a task in two suites runs once per step.

The step-0 checkpoint written by `checkpoint.create_seed_checkpoint` is
evaluated too. It is the model training starts from, so its scores are each
task's baseline.

### Models that do not fit on one GPU

`--tensor-parallel-degree N` shards each model replica over N GPUs with the
model's own tensor-parallel plan; the remaining GPUs are data-parallel
replicas that lm-eval splits the examples across. For example, 8 GPUs with
`--tensor-parallel-degree 4` run two replicas of four GPUs each. The model is
parallelized the same way as for training, so the checkpoint loads through
DCP resharding whatever layout it was trained with.

Other axes of the training layout (FSDP sharding, EP, CP, PP) are not used by
the eval job: lm-eval gives each data-parallel rank different requests, and
those axes communicate across ranks in the forward pass.

### W&B

`--wandb` logs every result as `eval/<task>/<metric>` to a W&B run named
`<run name>-eval-<suite>`. It reads the same environment variables as the
training job's W&B logger (`WANDB_TEAM`, `WANDB_PROJECT`, `WANDB_RUN_NAME`,
`WANDB_RUN_GROUP`). To group the eval run with the training run, set the same
`WANDB_RUN_GROUP` for both jobs. Metrics are plotted
against `train_step`, so backfilled or out-of-order results land in the right
place, and a restarted eval job resumes the same run.

### Other options

`--limit` (examples per task, for smoke tests), `--dtype`,
`--num-tokens-per-batch`, `--poll-interval`, `--max-idle-hours`,
`--output-folder`, `--decontaminate` (see [Contamination](#contamination));
see `--help`.

Rank 0 downloads the eval datasets once at startup; after that every rank
reads them from the local cache. Loading a task makes about ten Hugging Face
Hub requests even when it is cached, and the Hub allows 1000 requests per 5
minutes, so every rank loading every task at each step would be throttled.

Models with multi-token-prediction layers (the DeepSeek `_mtp` configs) are
evaluated on their main output layer; the MTP layers are not built.

Make sure `checkpoint.keep_latest_k` does not delete checkpoints before they
are evaluated; `checkpoint.purge_exempt` can keep selected steps.

## Contamination

Benchmark questions that also appear in the training data raise scores
through memorization. `contamination.py` checks for this with the 13-gram
overlap test used for GPT-3: an eval question counts as contaminated when any
13 consecutive words of it (lowercased, punctuation removed) also occur in a
training document.

It reads the training data through the run's own dataloader and decodes it
back to text, so it checks exactly what the run trains on, whatever the data
source. By default it scans the run's whole token budget
(`training.steps` x `training.num_tokens_per_train_step`). It needs no GPUs,
and each process scans its own share of the data. Run it with as many
processes as the training job's data-parallel degree: each process reads the
dataloader shard of the training rank with the same index, so with a different
count the scan covers a differently sharded stream, not exactly the tokens the
run trains on.

```bash
torchrun --nproc_per_node=32 -m torchtitan.experiments.evals.contamination \
    --tasks trainstation_full [--max-tokens 1000000000] -- $TRAIN_ARGS
```

The report, `<dump_folder>/evals/contamination.json`, lists for each task
how many questions overlap the training data, and which. Questions shorter
than 13 words cannot be checked and count as clean.

With `--decontaminate`, the eval job also scores each task on its clean
questions only, written as `step-N/<task>.clean.json` and logged to W&B under
`eval_clean/`. A gap between the full and clean scores shows how much a task
is inflated by contamination. For reading-comprehension tasks such as BoolQ,
an overlap is usually the passage (often from Wikipedia) rather than the
question and answer.

## Suites

Suites are lm-eval groups defined in [`tasks/`](./tasks); any lm-eval task or
group name also works with `--tasks`. Runs are compared on the gold-BPB
composites of the [step summary](#step-summary); accuracies are logged for
reading and for comparison with published tables.

| Suite | When | Tasks |
|---|---|---|
| `trainstation_quick` | Every checkpoint | `trainstation_ppl`, `trainstation_mc5`, `trainstation_mmlu_cloze`, LAMBADA, and gold BPB of TriviaQA, GSM8K, MATH-500, HumanEval and MBPP |
| `trainstation_full` | Milestones | `trainstation_quick` + `trainstation_mmlu` |
| `trainstation_ppl` | | `trainstation_paloma`, Paloma PTB and WikiText-103, OpenWebMath, as bits per byte |
| `trainstation_paloma` | | Nine Paloma sources (validation split) and their unweighted mean BPB |
| `trainstation_mc5` | | ARC-Challenge, ARC-Easy, HellaSwag, PIQA, WinoGrande, OpenBookQA, CommonsenseQA (cloze), Social IQa; 5-shot |
| `trainstation_mmlu_cloze` | | MMLU in cloze form (each answer's text is scored), 5-shot |
| `trainstation_mmlu` | | MMLU in the lettered format labs report, 5-shot |

Paloma (`allenai/paloma`) is gated: accept its terms on Hugging Face first.
OpenWebMath is read from Proof-Pile-2's compressed JSON Lines and needs
`zstandard` (in `requirements.txt`).

### Gold BPB

Accuracy stays at chance for small models on many benchmarks. Gold BPB, the
bits per byte of the correct answer's text given the prompt, moves smoothly
from the first checkpoint, so runs can be compared early (Signal and Noise,
arXiv 2508.13144).

- Multiple-choice tasks: `evaluate.py` logs lm-eval's samples for the units
  that contain a multiple-choice task and adds `gold_bpb` to every such task
  whose choices are text (lettered tasks are skipped). A group's `gold_bpb`
  is its tasks' `gold_bpb` weighted by their number of questions, as lm-eval
  weights MMLU accuracy. See `scores.py`.
- Generative tasks: `tasks/gold_bpb/` scores the reference solution (or, for
  TriviaQA, the canonical answer) as a log-likelihood continuation.

### Step summary

After a step's tasks are written, `evaluate.py` writes `step-N/summary.json`
with the composites in `scores.py` (logged to W&B as `eval/summary/<name>`),
each once every member has results:

| Composite | Members |
|---|---|
| `knowledge_gold_bpb` | Mean gold BPB of ARC-Challenge, ARC-Easy, HellaSwag, MMLU (cloze), TriviaQA |
| `math_gold_bpb` | GSM8K, MATH-500 |
| `code_gold_bpb` | HumanEval, MBPP |
| `cloze_acc` | Mean accuracy (`acc_norm` where defined) of the nine cloze multiple-choice tasks |
| `paloma_bpb`, `openwebmath_bpb` | The perplexity sets |

## Bits per byte in the training loop

`BPBValidator` (`validator.py`) is a `Validator` subclass, set as the
training config's validator. It reports the BPB of held-out text sets, scored
exactly as the `bits_per_byte` tasks above score them (each document from
BOS, lm-eval's rolling windows, the end-of-document token not scored),
through the trainer's own forward pass and loss. It logs `validation_metrics/<set>/bpb` and
`/loss`, and `validation_metrics/<prefix>/avg/bpb` for sets that share a
prefix, plus the core validator's loss with `training_loss=True`. Each
data-parallel rank scores every n-th rolling window of a set, and all ranks
run the same number of forward passes, so FSDP, TP and EP stay in lockstep
(tested so far with two data-parallel ranks on CPU). Pipeline and context
parallelism raise `ValueError`. The sets are
tokenized at the first validation, on every rank.

The eval job logs the same sets, on their full validation splits, to W&B as
`eval/trainstation_paloma_<source>/bits_per_byte`,
`eval/trainstation_paloma/bits_per_byte` (the nine-source mean, the in-loop
`paloma/avg`) and `eval/trainstation_openwebmath/bits_per_byte`.

Export the sets once (a fixed slice of at most 8 MB of each validation split,
and a manifest with paths relative to the output folder):

```bash
python -m torchtitan.experiments.evals.prepare_bpb_sets --output-folder /path/to/bpb_sets
```

Then set the validator in the training config:

```python
from torchtitan.experiments.evals.validator import BPBValidator, bpb_sets_from_manifest

validator=BPBValidator.Config(
    enable=True,
    freq=1000,
    sets=bpb_sets_from_manifest("/path/to/bpb_sets"),
    # BPB only. To also keep the core validation loss, leave training_loss
    # True and pass the run's validation dataloader and steps, as for
    # Validator.Config.
    training_loss=False,
)
```

## Limitations

- Log-likelihood tasks only. The eval job does not run generative benchmarks
  (exact match, pass@k); run them on the checkpoint converted to Hugging Face
  format, for example with lm-eval's `vllm` backend.
- No FSDP, EP, CP or PP in the eval job (see above), so MoE models too large
  for tensor parallelism alone are not supported yet.
