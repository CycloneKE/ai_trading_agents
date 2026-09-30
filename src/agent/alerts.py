"""Alerts to the operator's email, and a log of every alert raised.

The agent used to have no way to tell anyone anything: a halt, a dead feed or
a crash showed on the dashboard, if someone happened to open it. This sends
an email, from the agent itself, when something needs a person.

It is built so an alert can never make things worse:

- Sending happens on a background thread with a timeout, so a slow or broken
  mail server cannot hold up the trading loop.
- It never raises. A failure to send is logged and recorded, nothing more.
- It does not flood. The same alert (same `key`) is sent at most once per
  cooldown (15 minutes for a critical one, an hour for a warning), and
  non-critical mail is capped per hour.
- Every alert that is raised is recorded in data/alerts.jsonl whether or not
  it was sent (a repeat inside its cooldown is not recorded again), so the
  scorecard can count them and the operator can see what was raised while
  email was not set up.

Settings come from environment variables, so no password is ever written in
config.json or committed. In Coolify they go in the Environment Variables
tab, using the same values as Coolify's own email settings:

    ALERT_EMAIL_TO    who to tell (one address, or several separated by commas)
    SMTP_HOST         the mail server, for example smtp.gmail.com
    SMTP_PORT         587 for STARTTLS (the default) or 465 for SSL
    SMTP_USER         the mail account's login (often its address)
    SMTP_PASSWORD     its password, or an app password
    SMTP_FROM         the From address (defaults to SMTP_USER)
    SMTP_SECURITY     starttls (default), ssl or none
"""
import json
import logging
import os
import smtplib
import ssl
import threading
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Callable, Dict, List, Optional

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

SEVERITIES = ('info', 'warning', 'critical')
DEFAULT_COOLDOWN_S = {'info': 0, 'warning': 3600, 'critical': 900}
SUBJECT_PREFIX = '[Trading Agent]'


class AlertLog:
    """The alerts raised, one JSON object a line, newest last. Kept in the data
    volume so a redeploy does not lose it."""

    def __init__(self, path: Optional[str] = None, keep: int = 500):
        self.path = str(path or DATA_DIR / 'alerts.jsonl')
        self.keep = keep
        self._lock = threading.Lock()

    def add(self, entry: Dict[str, Any]) -> None:
        try:
            with self._lock:
                os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, default=str) + '\n')
                self._trim()
        except OSError as e:
            logger.warning(f"Could not record alert: {e}")

    def _trim(self) -> None:
        # Cheap check first: only rewrite once the file is well past its limit.
        try:
            if os.path.getsize(self.path) < self.keep * 400:
                return
            with open(self.path, encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > self.keep * 2:
                with open(self.path, 'w', encoding='utf-8') as f:
                    f.writelines(lines[-self.keep:])
        except OSError:
            pass

    def recent(self, limit: int = 50, since: Optional[str] = None) -> List[Dict[str, Any]]:
        """The latest `limit` alerts, newest first; with `since` (an ISO
        timestamp), only those raised at or after it."""
        try:
            with self._lock, open(self.path, encoding='utf-8') as f:
                lines = f.readlines()
        except OSError:
            return []
        out: List[Dict[str, Any]] = []
        for line in reversed(lines):
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            # Not `break`: a line is written when delivery finishes, under the
            # time the alert was raised, so the file is not strictly in order.
            if since and str(entry.get('ts', '')) < since:
                continue
            out.append(entry)
            if len(out) >= limit:
                break
        return out


def _settings(env: Dict[str, str]) -> Dict[str, Any]:
    port_text = (env.get('SMTP_PORT') or '').strip()
    try:
        port = int(port_text) if port_text else 587
    except ValueError:
        port = 587
    security = (env.get('SMTP_SECURITY') or '').strip().lower() or ('ssl' if port == 465 else 'starttls')
    user = (env.get('SMTP_USER') or '').strip()
    recipients = [a.strip() for a in (env.get('ALERT_EMAIL_TO') or '').split(',') if a.strip()]
    return {
        'host': (env.get('SMTP_HOST') or '').strip(), 'port': port, 'security': security,
        'user': user, 'password': env.get('SMTP_PASSWORD') or '',
        'sender': (env.get('SMTP_FROM') or '').strip() or user, 'to': recipients,
    }


def _smtp_transport(s: Dict[str, Any], message: EmailMessage, timeout: float = 20.0) -> None:
    """Deliver one message over SMTP; raises on any failure."""
    if s['security'] == 'ssl':
        server = smtplib.SMTP_SSL(s['host'], s['port'], timeout=timeout, context=ssl.create_default_context())
    else:
        server = smtplib.SMTP(s['host'], s['port'], timeout=timeout)
    try:
        if s['security'] == 'starttls':
            server.starttls(context=ssl.create_default_context())
        if s['user']:
            server.login(s['user'], s['password'])
        server.send_message(message)
    finally:
        try:
            server.quit()
        except Exception:
            pass


def _mask(address: str) -> str:
    name, _, domain = address.partition('@')
    return f"{name[:1]}***@{domain}" if domain else '***'


class EmailAlerter:
    """Raise alerts by email. See the module note for what it guarantees."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 env: Optional[Dict[str, str]] = None,
                 log: Optional[AlertLog] = None,
                 transport: Optional[Callable[[Dict[str, Any], EmailMessage], None]] = None,
                 background: bool = True,
                 clock: Callable[[], float] = time.time):
        cfg = (config or {}).get('alerts') or {}
        self.enabled = cfg.get('enabled', True)
        self.max_per_hour = int(cfg.get('max_per_hour', 20))
        self.cooldowns = {**DEFAULT_COOLDOWN_S,
                          **{k: int(v) for k, v in (cfg.get('cooldown_seconds') or {}).items()
                             if k in SEVERITIES}}
        self.settings = _settings(dict(os.environ if env is None else env))
        self.log = log or AlertLog()
        self._transport = transport or _smtp_transport
        self._background = background
        self._clock = clock
        self._lock = threading.Lock()
        self._last_sent: Dict[str, float] = {}
        self._sent_times: List[float] = []
        self.last_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(self.enabled and s['host'] and s['to'] and s['sender'])

    def status(self) -> Dict[str, Any]:
        """What the dashboard and the scorecard show about email alerts."""
        since = datetime.fromtimestamp(self._clock() - 86400, tz=timezone.utc).isoformat()
        recent = self.log.recent(limit=500, since=since)
        count = lambda st: sum(1 for a in recent if a.get('status') == st)
        return {'configured': self.configured,
                'recipients': [_mask(a) for a in self.settings['to']],
                'sent_24h': count('sent'), 'failed_24h': count('failed'),
                'suppressed_24h': count('suppressed'), 'last_error': self.last_error}

    def send(self, key: str, subject: str, body: str = '', severity: str = 'warning',
             cooldown_s: Optional[float] = None) -> str:
        """Raise an alert. Returns what became of it: 'sent' (queued for
        delivery), 'suppressed' (the same alert went out recently, or the
        hourly cap is reached) or 'not_configured' (recorded only). Never raises."""
        try:
            severity = severity if severity in SEVERITIES else 'warning'
            now = self._clock()
            entry = {'ts': datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                     'key': key, 'severity': severity, 'subject': subject, 'body': body}
            cooldown = self.cooldowns[severity] if cooldown_s is None else cooldown_s
            with self._lock:
                # The same alert inside its cooldown is dropped quietly, sent or
                # not: a condition that lasts all day is one alert, and one line
                # in the log, not one a minute.
                if now - self._last_sent.get(key, -1e12) < cooldown:
                    return 'suppressed'
                self._sent_times = [t for t in self._sent_times if now - t < 3600]
                if self.configured and severity != 'critical' and len(self._sent_times) >= self.max_per_hour:
                    self.log.add({**entry, 'status': 'suppressed', 'why': 'hourly cap'})
                    return 'suppressed'
                self._last_sent[key] = now
                if self.configured:
                    self._sent_times.append(now)
            if not self.configured:
                self.log.add({**entry, 'status': 'not_configured'})
                logger.warning(f"ALERT [{severity}] {subject} (email is not set up, so nobody was told)")
                return 'not_configured'
            message = self._compose(severity, subject, body, entry['ts'])
            if self._background:
                threading.Thread(target=self._deliver, args=(entry, message), daemon=True,
                                 name='alert-mail').start()
            else:
                self._deliver(entry, message)
            return 'sent'
        except Exception as e:                     # an alert must never break the caller
            logger.error(f"Alerting failed: {e}")
            return 'suppressed'

    def _compose(self, severity: str, subject: str, body: str, when: str) -> EmailMessage:
        s = self.settings
        try:
            from src.utils.build_info import code_version
            version = code_version()
        except Exception:
            version = 'unknown'
        msg = EmailMessage()
        msg['Subject'] = f"{SUBJECT_PREFIX} {severity.upper()}: {subject}"
        msg['From'] = s['sender']
        msg['To'] = ', '.join(s['to'])
        msg.set_content(
            f"{body}\n\n--\nRaised at {when} by the trading agent (version {version}).\n"
            f"This alert will not repeat for a while, however long the problem lasts.")
        return msg

    def _deliver(self, entry: Dict[str, Any], message: EmailMessage) -> None:
        try:
            self._transport(self.settings, message)
            self.log.add({**entry, 'status': 'sent'})
            self.last_error = None
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            logger.error(f"Could not email alert '{entry.get('subject')}': {self.last_error}")
            self.log.add({**entry, 'status': 'failed', 'why': self.last_error})
