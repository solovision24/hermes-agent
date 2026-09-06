# Adaptive model routing

Hermes can select a model once, immediately before the first acting-model call of a user turn. The selected model and provider remain fixed for that turn, preserving normal tool loops and prompt-cache prefixes.

This feature is opt-in and independent of `fallback_providers`, `provider_routing`, delegation model overrides, and MoA. If a configured target cannot be resolved, Hermes keeps the current model/provider for that turn.

Example in `config.yaml`:

```yaml
adaptive_model_routing:
  enabled: true
  routes:
    simple_chat:
      provider: openrouter
      model: openai/gpt-5-mini
    simple_coding:
      provider: openrouter
      model: openai/gpt-5-mini
    standard_coding:
      provider: openrouter
      model: anthropic/claude-sonnet-4
    complex_coding:
      provider: openrouter
      model: anthropic/claude-sonnet-4
    architecture_coding:
      provider: openrouter
      model: anthropic/claude-opus-4
    research:
      provider: openrouter
      model: openai/gpt-5-mini
    ops:
      provider: openrouter
      model: anthropic/claude-sonnet-4
    creative:
      provider: openrouter
      model: openai/gpt-5-mini
```

Classification is deterministic and conservative: obvious migrations, schema design, production rollouts, and security-model changes route to `architecture_coding`; coding, research, operations, and creative language select their corresponding categories. Route decisions are available in the per-turn `adaptive` metadata and logs can use that metadata for auditability.
