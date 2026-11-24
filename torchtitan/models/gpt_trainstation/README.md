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

`model_registry(..., mup=MuPConfig(m_width=..., m_emb=..., m_residual=...))`
enables the muP width parameterization used by lm-engine: hidden matrices init
with std `0.02 / sqrt(m_width)` (output projections further divided by
`sqrt(2 * n_layers)`), the token embedding output is scaled by `m_emb`, each
residual branch by `m_residual`, and the logits by `1 / m_width`. Pair it with
`config_registry.mup_adamw_optimizer`, which trains hidden matrices at
`lr / m_width` and the embedding, LM head and norms at `lr`.
`config_registry.gpt_trainstation_debugmodel_mup` wires both together.
