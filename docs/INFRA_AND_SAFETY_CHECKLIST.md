# Infra & Safety Checklist (test/startup guidance)

This file documents a few runtime flags and safe procedures to run the project locally and in CI without contacting external data providers.

Key runtime flags

- `data_manager.use_fallback_only` (boolean) — when set in the `data_manager` section of the config, the `DataManager` will skip initializing any external connectors and will use the internal fallback generator. Use this for smoke tests and CI.
- `use_fallback_only` or `test_mode` (top-level boolean) — convenience flags that are propagated into `data_manager.use_fallback_only` by the `main` application. You can set this at the top-level of a config file instead of modifying `data_manager` directly.

Recommended commands

Run the smoke-startup test locally (uses fallback-only):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agent_smoke.py -q
```

Create a temporary config and run the agent locally with fallback-only enabled (manual run):

```powershell
# Create a copy of your config and add data_manager.use_fallback_only = true
copy config\run_config.json config\run_fallback.json
(Get-Content config\run_fallback.json) -replace '"monitoring"', '"monitoring"' | Set-Content config\run_fallback.json
# (Edit config\run_fallback.json and add "data_manager": {"use_fallback_only": true} if needed)
.\.venv\Scripts\python.exe main.py --config config\run_fallback.json
```

CI notes

- The `tests/test_agent_smoke.py` test already sets `data_manager.use_fallback_only` on the temporary config it writes. The repository CI contains a manual-only (`workflow_dispatch`) smoke job that runs this test. This keeps CI safe from accidental network/API calls.
- If you add new startup/integration tests that touch connectors, make sure to add a `use_fallback_only` switch in the test or mock the connectors.

Troubleshooting

- If pytest fails during collection with `ModuleNotFoundError` for optional test libraries (e.g., `bs4`, `responses`), install test dependencies with:

```powershell
pip install -r requirements-test.txt
```

- To run only the smoke/test-data-manager tests (fast):

```powershell
.\.venv\Scripts\python.exe -m pytest -q -k "agent_smoke or data_manager_fallback"
```

Security & secrets

- Do not enable real connectors in CI. Keep secrets out of CI or add them via repository secrets when necessary for gated integration tests.
- For rotating or scrubbing secrets from history, coordinate a controlled history rewrite and rotation process — it is destructive and should be planned.

If you want, I can add a short README section instead of this file. Pick which you prefer.
# Infrastructure & Safety Checklist for ai_trading_agents

This checklist collects the operational and safety items you should satisfy before enabling online RL, ML builds, or automated deployments. Follow these steps to reduce operational risk and ensure reproducible deployments.

## 1) Secrets & Access
- Store all credentials in GitHub Secrets (or a vault like HashiCorp Vault / AWS Secrets Manager). Do NOT commit secrets to the repo.
- Required secrets (examples):
  - DOCKERHUB_USERNAME, DOCKERHUB_TOKEN
  - GCP_WORKLOAD_IDENTITY_PROVIDER, GCP_SERVICE_ACCOUNT
  - TRADING_ALPACA_API_KEY, TRADING_ALPACA_SECRET
  - TRADING_FMP_API_KEY, TRADING_FINNHUB_API_KEY
  - COINBASE_API_KEY, COINBASE_SECRET
- Rotate credentials immediately if any secret is discovered in the repo's history.

## 2) Compute & Runner Requirements
- Unit tests and CI jobs: use standard GitHub hosted runners (ubuntu-latest).
- ML heavy jobs (training / ML image build): use dedicated runners with >=32 GB RAM and GPUs (if model uses torch/tensorflow). Create a `ml` runner group or use cloud GPU instances.
- Use docker buildx + registry cache to speed repeated builds.

## 3) Data Pipelines & Storage
- Use managed time-series storage or object store (GCS/S3) for historical market data and feature store snapshots.
- Ensure data retention and backfill policies exist and are tested.
- Maintain deterministic sample datasets for CI backtest checks.

## 4) Model Checkpointing & Registry
- Save models with metadata (git commit, training data window, hyperparams, validation metrics) to a model registry or object store.
- Keep a canonical `model_manifest.json` for serving which maps model versions to artifacts and validation metrics.

## 5) Safety & Risk Controls
- Implement portfolio-level stop-loss and global circuit-breakers in `live_risk_manager` (done PoC). Circuit-breakers should disable automated trading when thresholds are exceeded.
- Add Kelly sizing with shrink and caps; set conservative defaults.
- Require manual approval to enable online RL or to accept new models in production (feature flag or gating workflow).

## 6) Monitoring & Alerts
- Export metrics via Prometheus: latency, slippage, order failures, win rate, drawdown, exposure, model metrics (validation score), and number of open positions.
- Create Grafana dashboards for live PnL, drawdown, per-strategy performance, and model drift indicators.
- Alerts (PagerDuty/Slack): severe drawdown (> configured threshold), model deployment failure, large negative PnL spikes, and data pipeline failure.

## 7) Canary & Rollback Strategy
- Use feature flags and small canary allocations for new models/strategies (e.g., 1-5% capital) and monitor real-time performance for the canary window.
- Keep a fast rollback path: un-register the model in the manifest, revert deployment, and have a documented rollback playbook.

## 8) CI & Testing
- CI should run: unit tests, linting, formatting, and deterministic backtest sanity checks (small datasets). Integration tests run only on manual dispatch when secrets are present.
- Add tests that ensure risk manager can veto trades and stop trading under simulated adverse conditions.

## 9) Retraining & Online Learning Rules
- If using online learning:
  - Use a sandboxed environment with simulated capital until models pass a defined validation gate.
  - Decide retrain frequency (e.g., weekly) and validation criteria (minimum Sharpe, max drawdown, no model drift beyond X).
  - Keep training logs and reproducible seeds for experiments.

## 10) Operational Runbooks
- Create runbooks for: emergency halt (stop trading), rotating secrets, debugging latency/slippage, and running integration tests.
- Maintain a contact list and escalation path.

## 11) Compliance & Audit
- Log all trade decisions, input features, model versions, and environment metadata for auditability.
- Redact PII and sensitive keys in logs and configure log retention policies.

---
When you're ready I can:
- Generate a PR that adds the checklist and links from `README.md`.
- Create GitHub Action templates for canary deployments and a protected manual promotion flow.
- Add Prometheus metrics hooks into `live_risk_manager` and the backtest engine.

Which of those would you like me to tackle next?