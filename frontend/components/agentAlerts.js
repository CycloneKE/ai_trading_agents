// What the agent's own alerts add to the Notifications page and the bell
// (/api/agent-alerts, src/agent/alerts.py). No imports, so it can be tested
// on its own.
//
// Two kinds of item:
//  - what is wrong right now (a worker down, the loop failing, symbols with no
//    price). These are counted while they are true and drop off by themselves
//    when the agent has fixed them.
//  - alerts the agent raised (a halt, a restart, a crash) that nobody has
//    marked as read yet. Critical and warning ones need attention; info ones
//    are for information. "Mark as read" remembers, in this browser, the time
//    of the newest alert seen.

export const SEEN_KEY = 'agent_alerts_seen_at';
export const MAX_UNREAD_SHOWN = 30;

const SEVERITY = { critical: 'high', warning: 'medium', info: 'info' };
const DAY_MS = 24 * 60 * 60 * 1000;

const when = (ts) => {
  const t = Date.parse(ts);
  return Number.isNaN(t) ? null : t;
};

const sentence = (text) => {
  const t = String(text || '').trim();
  return t ? t[0].toUpperCase() + t.slice(1) : '';
};

// Alerts newer than the last "mark as read". With nothing marked yet, only
// the last day's count as unread, so a new browser is not greeted by a week of them.
export function unreadAlerts(alerts, seenAt, now = Date.now()) {
  const cutoff = (seenAt && when(seenAt)) ?? now - DAY_MS;
  return (alerts || []).filter((a) => {
    const t = when(a.ts);
    return t !== null && t > cutoff;
  });
}

// The time to remember when the operator marks everything read.
export function newestAlertTime(alerts) {
  let best = null;
  (alerts || []).forEach((a) => {
    const t = when(a.ts);
    if (t !== null && (best === null || t > best.t)) best = { t, ts: a.ts };
  });
  return best ? best.ts : new Date().toISOString();
}

export function agentAlertItems({ agentAlerts, seenAt, status = {}, now = Date.now() }) {
  if (!agentAlerts) return [];
  const items = [];
  const live = agentAlerts.live || {};

  (live.workers_down || []).forEach((w) => items.push({
    key: `worker-${w}`, severity: 'high', attention: true,
    message: `A background worker is not running: ${w.replace(/_/g, ' ')}.`,
    hint: 'The agent tries to restart it by itself. If it stays down, look at the logs or redeploy.',
  }));

  if ((live.loop_errors || 0) >= 3) {
    items.push({
      key: 'loop-errors', severity: 'high', attention: true,
      message: `The trading loop has failed ${live.loop_errors} times in a row, so nothing is being decided.`,
      hint: live.last_loop_error || 'It keeps retrying every cycle. Look at the logs; a restart may clear it.',
    });
  }

  const noPrice = Object.keys(live.symbols_without_price || {});
  if (noPrice.length) {
    const shown = noPrice.slice(0, 6).join(', ') + (noPrice.length > 6 ? ` and ${noPrice.length - 6} more` : '');
    items.push({
      key: 'no-price', severity: 'medium', attention: true,
      message: `No real price for ${noPrice.length} symbol${noPrice.length > 1 ? 's' : ''} for over 30 minutes: ${shown}.`,
      hint: 'The agent will not trade on made-up prices, so these are idle. Usually the price vendor is throttling or down; it clears by itself when prices return.',
    });
  }

  unreadAlerts(agentAlerts.alerts, seenAt, now)
    .filter((a) => !(a.key === 'heal:halt' && status.trading_halted))   // the halt item already says it
    .slice(0, MAX_UNREAD_SHOWN)
    .forEach((a) => {
      const severity = SEVERITY[a.severity] || 'medium';
      items.push({
        key: `alert-${a.id}`, severity, attention: a.severity !== 'info',
        message: sentence(a.subject), hint: a.body ? String(a.body).slice(0, 400) : undefined,
        lastAt: a.ts, lastLabel: 'Raised',
      });
    });
  return items;
}
