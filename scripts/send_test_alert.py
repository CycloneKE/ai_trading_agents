#!/usr/bin/env python3
"""Send one test email through the agent's alert settings.

Run it after setting ALERT_EMAIL_TO and the SMTP_* variables (see .env.example)
to find out, in ten seconds, whether the agent will actually be able to email
you. It says exactly what is missing or what the mail server said.

Usage:
    python scripts/send_test_alert.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.alerts import AlertLog, EmailAlerter               # noqa: E402


def main(env=None, transport=None, log_path=None) -> int:
    alerter = EmailAlerter({}, env=env, transport=transport, background=False,
                           log=AlertLog(log_path) if log_path else None)
    s = alerter.settings
    missing = [name for name, ok in (('ALERT_EMAIL_TO', s['to']), ('SMTP_HOST', s['host']),
                                     ('SMTP_USER or SMTP_FROM', s['sender'])) if not ok]
    if missing:
        print('Email is not set up. Missing: ' + ', '.join(missing) + '.\n'
              'Set them in Coolify (Environment Variables), redeploy, and run this again. See .env.example.')
        return 1
    alerter.send('test', 'test alert: email is working',
                 'This is a test from the trading agent. If you can read it, the agent can email you '
                 'when it halts or something breaks.', 'critical', cooldown_s=0)
    if alerter.last_error:
        print(f"The email could not be sent: {alerter.last_error}\n"
              f"Server {s['host']}:{s['port']} ({s['security']}), login {s['user'] or '(none)'}.\n"
              "Common causes: a wrong password (Gmail needs an app password), the wrong port "
              "(587 for starttls, 465 for ssl), or SMTP_SECURITY not matching the port.")
        return 1
    print('Sent to ' + ', '.join(alerter.status()['recipients']) + '. Check that inbox (and the spam folder).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
