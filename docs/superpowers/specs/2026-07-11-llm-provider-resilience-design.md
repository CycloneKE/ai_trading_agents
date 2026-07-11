# Free-Tier LLM Provider Resilience — Design

## Problem

The agent's LLM layer floods the log with `404 Not Found` and
`429 Too Many Requests` errors and, in practice, rarely gets a usable
verdict:

- **404**: `primary_llm_provider = gemini` and the Gemini path calls
  `gemini-2.5-flash-lite`, which is not a valid/available model id on the
  `v1beta` endpoint for the configured key.
- **429**: Gemini's free tier has tight per-minute limits; the agent bursts
  past them (per-symbol validation + ~5 sector-specialist calls + the weight
  allocator) with no throttle, backoff, or cross-provider fallback.
- **Wiring dead-end**: `swarm.agents` is configured entirely with OpenRouter
  model ids, and the OpenRouter key IS set, but with gemini primary the
  OpenRouter path is never taken; OpenRouter ids passed into the Gemini path
  get force-mapped to the broken `gemini-2.5-flash-lite`. Fallback is
  one-directional (openrouter→gemini), so gemini-primary has no fallback.

Constraint: **free tiers only** — no per-call spend.

## Goals

1. Eliminate the 404 (valid free model ids per provider).
2. Eliminate the 429 storm (cross-provider fallback + per-provider cooldown +
   lower call volume).
3. Make the two free providers cooperate instead of dead-ending.
4. Preserve graceful degradation: when everything fails, trades proceed on the
   base ensemble signal (never force-held by an LLM outage), allocator/sector
   keep prior state.

## Non-goals

- Paid models / quality upgrades (explicitly free-only).
- Changing the decision pipeline or what the LLM is asked.

## Design

### 1. Valid free model ids

- Gemini default model: `gemini-2.5-flash-lite` → `gemini-2.0-flash`
  (known-good, free-tier, `v1beta`). Config-overridable via
  `gemini_model` (default `gemini-2.0-flash`).
- `swarm.agents` model ids switch from paid to free OpenRouter `:free`
  variants (e.g. `meta-llama/llama-3.1-8b-instruct:free`) so any OpenRouter
  call uses a free model. When an OpenRouter-style id (`vendor/model`) reaches
  the Gemini path, it maps to the configured `gemini_model`, not a hardcoded
  broken id.

### 2. Bidirectional provider fallback

A single internal `_complete(system, user, fallback, model_override)` helper
encapsulates the provider order:

```
providers = [primary, other]           # e.g. [gemini, openrouter]
for p in providers:
    if p on cooldown: continue
    try: return call(p)
    except 429: set_cooldown(p); continue
    except (404, timeout, parse, other): continue
return fallback                          # base signal / None
```

`validate_trade` and `propose_json` both route through `_complete` so they get
identical resilience. This doubles effective free quota and makes a 429 on one
provider transparently use the other.

### 3. Per-provider 429 cooldown (circuit-breaker)

In-memory `self._cooldown_until = {provider: epoch}`. On a 429 from a
provider, set its cooldown to `now + cooldown_seconds` (config
`llm_cooldown_seconds`, default 60). While cooling down, that provider is
skipped (the other is tried). Log the cooldown **once** when it trips, not on
every subsequent skipped call — this is what turns the 429 flood into near
silence. HTTP status is read from the `requests` exception
(`e.response.status_code`) to distinguish 429 from other errors.

### 4. Cut call volume — batch sector specialists

`SectorSpecialistManager` currently issues one LLM call per sector (~5/cycle).
Change to a **single batched call** that returns a JSON object keyed by
sector, and run the sector pass less often (config
`sector_analysis_interval`, default: once daily aligned with the existing
self-assessment cadence rather than every assessment cycle). This removes the
largest source of free-tier pressure.

### 5. Graceful degradation (mostly already correct)

- `validate_trade`: all providers fail/cooldown → return the base ensemble
  signal (trade proceeds un-vetoed). Unchanged behavior, just via `_complete`.
- `propose_json`: all fail → return None (allocator keeps prior weights,
  sector keeps prior profiles). Unchanged.
- Logging: a provider failure logs at debug; a cooldown trip logs once at
  warning. No more per-call 429 ERROR spam.

## Config additions (all optional, sane defaults)

```
"primary_llm_provider": "gemini",          # existing
"gemini_model": "gemini-2.0-flash",        # new — valid free model
"llm_cooldown_seconds": 60,                # new — 429 circuit-breaker window
"sector_analysis_interval": 390,           # new — cycles between sector passes
"swarm": { "agents": {                     # switch to free ids
    "synthesizer": "meta-llama/llama-3.1-8b-instruct:free",
    "sector_specialist": "meta-llama/llama-3.1-8b-instruct:free",
    "sentiment_analyst": "meta-llama/llama-3.1-8b-instruct:free",
    "pdf_extractor": "meta-llama/llama-3.1-8b-instruct:free"
}}
```

## Testing

- Unit: `_complete` provider ordering — primary success returns immediately;
  primary 429 sets cooldown and falls through to secondary; both down returns
  the fallback; a provider on cooldown is skipped without a call. Providers
  are stubbed (no network).
- Unit: cooldown expiry re-enables a provider after the window.
- Unit: sector-specialist batched call parses a multi-sector JSON object and
  falls back to prior profiles on failure.
- Manual/live: restart the backend and confirm the log no longer streams 404
  or 429 errors, and `/api/status` / weight allocator still function (a
  usable verdict is obtained from at least one free provider).

## Rollout

Config-only + orchestrator-internal changes; no API or dashboard change. The
decision pipeline and its inputs are untouched. Backward compatible: absent
config keys fall back to the new safe defaults.
