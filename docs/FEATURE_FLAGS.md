# Feature Flags

This project supports lightweight in-process feature flags via the `feature_flags` module.

Usage

- Read a flag in code:

```python
from feature_flags import is_enabled
if is_enabled('new_risk_manager'):
    # use new implementation
```

- Set a flag at runtime (for tests or admin scripts):

```python
from feature_flags import set_flag
set_flag('new_risk_manager', True)
```

- Environment variable override: `FF_NEW_RISK_MANAGER=true`

Notes

- These flags are in-process only and not persisted. For production feature flags, integrate a remote feature flagging service or use a configuration store.
