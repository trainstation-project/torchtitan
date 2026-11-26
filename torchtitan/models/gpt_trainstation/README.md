# GptTrainstation

A decoder that exists for test coverage, not for training a released model. It
has no checkpoint and no state-dict adapter.

One `debugmodel` flavor uses GQA (CosSin RoPE, QK norm) in every layer and mixes
every shared FFN type, repeating a four-layer pattern so every stage of a
2-stage pipeline split sees each type:

| layer % 4 | FFN |
|---|---|
| 0 | dense MLP |
| 1 | MoE, softmax router, routed experts only |
| 2 | MoE, sigmoid router, shared experts, aux loss |
| 3 | dense MLP |

Sharding and parallelize both reuse Qwen3's.

Integration tests live in `tests/integration_tests/models.py` (`gpt_trainstation_*`)
and cover FSDP, TP+SP, CP (KV all-gather, ptrr load balancing), PP
(1F1B, Interleaved1F1B), EP, compile, selective AC, varlen attention, SPMD type
checking, and DistMuon (`config_registry.dist_muon_optimizer`).

## muP

`MuPConfig(m_width=..., m_emb=..., m_residual=...)` describes the muP width
parameterization used by lm-engine: the embedding and LM head init from a
normal with std `initializer_range` (a `model_registry` argument, default
0.02), hidden matrices with std `initializer_range / sqrt(m_width)`, and
output projections further divided by `sqrt(2 * n_layers)`. The token
embedding output is scaled by `m_emb`, each residual branch by `m_residual`,
and the logits by `1 / m_width`. Pair it with
`config_registry.mup_adamw_optimizer`, which mirrors lm-engine's
`param-groups/mup.yml`: norms at `lr` without weight decay, the embedding and
LM head at `lr`, and every other parameter at `lr / m_width`.
The `debugmodel` flavor enables muP with arbitrary (untuned) values
`m_width=2.0, m_emb=12.0, m_residual=0.22`, and every `debugmodel` trainer
config uses `mup_adamw_optimizer`.
