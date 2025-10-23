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