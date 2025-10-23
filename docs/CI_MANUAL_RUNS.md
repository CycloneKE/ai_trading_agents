# Manual CI jobs and how to run them

This repository exposes several manual CI jobs designed for operations that require secrets or heavy resources. These jobs are triggerable via GitHub's `workflow_dispatch` UI on the Actions page or via the GitHub API.

Jobs

- `integration` — runs integration tests marked with `@pytest.mark.integration`. Use this when you want to run live/network tests against real provider APIs. Requires repo secrets for providers (set in GitHub Settings → Secrets):
  - TRADING_ALPACA_API_KEY
  - TRADING_ALPHA_VANTAGE_API_KEY
  - TRADING_FMP_API_KEY
  - TRADING_FINNHUB_API_KEY
  - COINBASE_API_KEY

- `ml_build` — builds the Docker `ml` target (heavy ML dependencies). This job is manual to avoid consuming CI resources automatically. To push the built ML image to DockerHub, set:
  - DOCKERHUB_USERNAME
  - DOCKERHUB_TOKEN

- `docker_image_build` — a manual job that builds the production image locally on the runner; will push if DockerHub credentials are available.

How to trigger from the web UI

1. Go to the repository → Actions tab.
2. Select the `CI/CD Pipeline` workflow.
3. Click "Run workflow" and choose the branch (usually `main`).
4. Choose any inputs if present and click the green "Run workflow" button.

How to trigger using the GitHub CLI (gh)

```bash
# Trigger the workflow
gh workflow run "CI/CD Pipeline" --repo YOUR_ORG/ai_trading_agents --ref main
```

Notes

- The `integration` job will run integration tests. Ensure you have added provider secrets to GitHub before running it. If you don't want to expose secrets in the UI, use a temporary maintenance branch and rotate the secrets afterwards.
- The build jobs will always build a local image; push steps are conditional and only execute when `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are set in the repo secrets.
- For the `ml_build` job, expect long build times and high memory use. Consider running that job only on a dedicated runner if possible.

If you want I can add example `gh` commands pre-filled with your repo name, or create a small runner-setup note for `ml_build` to use a larger GitHub Actions runner. Let me know which you prefer.
