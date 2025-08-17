# Secrets Directory

This directory contains sensitive information such as API keys and passwords.

## Security Guidelines

1. Never commit actual secrets to version control
2. Use environment variables or the secrets manager for production
3. Files in this directory should be added to .gitignore

## Required Secret Files

- `postgres_password.txt`: Password for PostgreSQL database
- `grafana_password.txt`: Admin password for Grafana
- `api_keys.json`: API keys for external services (encrypted)

## Using the Secrets Manager

```python
from secrets_manager import get_secret

# Get an API key
api_key = get_secret('alpha_vantage.api_key')

# Use the key
if api_key:
    # Make API call
    pass
else:
    # Handle missing key
    pass
```