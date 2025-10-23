# How to remove committed secrets from Git history and rotate them

This document shows safe, repeatable steps to remove a secret that was accidentally committed (for example `secrets/.key`) from the repository history and rotate the credential at the provider.

Important: these operations rewrite history. Coordinate with your team, take a backup, and be prepared to force-push and ask contributors to re-clone or rebase.

## 1) Make a backup

Keep a copy of the repository directory before rewriting history:

PowerShell:

```powershell
cd ..\
Copy-Item -Recurse -Force .\ai_trading_agents .\ai_trading_agents_backup
``` 

Unix:

```bash
cd ..
cp -a ai_trading_agents ai_trading_agents_backup
```

## 2) Identify the file(s) to purge

In this repo we saw `secrets/.key`. Confirm with:

```bash
git log --all --pretty=format:%H -- secrets/.key
```

If that prints commits, the file exists in history.

## 3) Recommended: use git-filter-repo (fast and robust)

Install `git-filter-repo`:

- Debian/Ubuntu: `sudo apt-get install git-filter-repo` (may not be packaged in all distros)
- Python pip: `pip install git-filter-repo`
- On macOS: `brew install git-filter-repo`

Run the rewrite (from the repository root):

PowerShell:

```powershell
# Ensure you are on a throwaway branch or local clone and have a backup
git checkout --orphan tmp-clean-branch
git commit --allow-empty -m "tmp"
# Move back to main to perform the filter
git checkout main

# Remove the file path from history
git-filter-repo --invert-paths --paths secrets/.key

# Force-push the cleaned branch to origin (coordinate with team!)
git push --force --all
git push --force --tags
```

Unix (bash):

```bash
git checkout --orphan tmp-clean-branch
git commit --allow-empty -m "tmp"
git checkout main

git-filter-repo --invert-paths --paths secrets/.key

git push --force --all
git push --force --tags
```

Notes:
- `git-filter-repo` replaces the older `git filter-branch` and is much faster.
- If you don't have `git-filter-repo` available, see the BFG repo-scrubber steps below.

## 4) Alternative: use BFG Repo-Cleaner

Download BFG (https://rtyley.github.io/bfg-repo-cleaner/) and run::

```bash
# Replace 'secrets/.key' with the path you want to remove
java -jar bfg.jar --delete-files secrets/.key
# Then:
git reflog expire --expire=now --all
git gc --prune=now --aggressive
# Force push
git push --force --all
git push --force --tags
```

## 5) Rotate and revoke the exposed secret

Immediate steps after purging local history:
1. Log in to the provider that issued the secret (DockerHub, GCP, Alpaca, etc.).
2. Revoke or rotate the exposed key/token.
3. Add the new credential to the repository's secret storage (GitHub -> Settings -> Secrets) or your secret manager.
4. Update CI variables to use the new secret values.

## 6) Notify collaborators

Because history was rewritten, all contributors must re-clone the repo or reset their local branches. Recommended workflow for contributors:

```bash
# safest: re-clone
cd ..
rm -rf ai_trading_agents
git clone git@github.com:YOUR_ORG/ai_trading_agents.git
```

Or if they must preserve local work, follow the rebase/fetch flow described in GitHub docs.

## 7) Verify

After pushing the cleaned history and rotating secrets, verify that the secret no longer appears:

```bash
git log --all --pretty=format:%H -- secrets/.key
# Should print nothing
```

## 8) Optional: create a follow-up PR to remove any references in docs

Search the repository for the secret string (if you captured it) and remove/replace it in docs, sample files, or examples.

---
If you want, I can prepare the exact `git-filter-repo` or BFG commands tailored to your repo remote URL and help coordinate a safe force-push. Let me know how you'd like to proceed.
