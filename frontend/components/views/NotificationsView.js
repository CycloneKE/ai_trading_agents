// Notifications: everything the agent wants the operator to know, in one
// place, split into what needs a decision and what is only for information.
// Built by the dashboard shell (notificationItems) from the anomaly scan,
// the approval queue, the halt switch, the dashboard's version check and the
// agent's own alerts (agentAlerts.js), which are also listed here in full.
import { useState } from 'react';
import { Bell, AlertTriangle, Info } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { card, SectionHeader, Empty, localTime } from '../ui';
import { agentAlertItems, unreadAlerts } from '../agentAlerts';

const DOT = { high: theme.colors.danger, medium: theme.colors.warning, low: theme.colors.textMuted, info: theme.colors.accent };

const PROVIDER = { anthropic: 'Claude', groq: 'Groq', gemini: 'Gemini', openrouter: 'OpenRouter' };
const clock = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : '');

// The settings the daily AI review may comment on (self_assessment.ADJUSTABLE).
const SETTING = {
  'risk_limits.stop_loss_pct': ['the fixed stop-loss', 'pct'],
  'risk_limits.trailing_stop_pct': ['the fixed trailing stop', 'pct'],
  'risk_limits.stop_loss_atr_mult': ['the stop-loss distance', 'atr'],
  'risk_limits.trailing_stop_atr_mult': ['the trailing stop distance', 'atr'],
};
const settingValue = (kind, v) => (kind === 'pct' ? `${+(v * 100).toFixed(1)}%` : `${v} x ATR`);

// The daily AI review is advice: its checked suggestions are shown here and
// nothing changes unless config.json is changed.
function reviewItems(review) {
  if (!review || !(review.suggestions || []).length) return [];
  return review.suggestions.map((s, i) => {
    const [name, kind] = SETTING[s.target] || [s.target, 'atr'];
    return {
      key: `review-${i}`, severity: 'info', attention: false,
      message: `AI review suggests changing ${name} from ${settingValue(kind, s.current)} to ${settingValue(kind, s.proposed)}${s.reasoning ? `: ${s.reasoning}` : '.'}`,
      hint: `Based on ${review.closed_trades ?? review.executed_trades} closed trades. Advice only: nothing changes unless config.json is changed.`,
      lastAt: review.at, lastLabel: 'Reviewed',
    };
  });
}

// What the AI services are doing: failing ones need a look when nothing else
// answers; otherwise they are information, as are the ones answering.
function aiItems(ai) {
  if (!ai) return [];
  if (!ai.enabled || !(ai.providers || []).length) {
    return [{
      key: 'ai-none', severity: 'medium', attention: true,
      message: 'No AI service is set up: trades are reviewed and pictures read without AI.',
      hint: 'Add GROQ_API_KEY or GEMINI_API_KEY (both free) in Coolify, and ANTHROPIC_API_KEY if you want Claude.',
    }];
  }
  const answering = ai.providers.filter((p) => p.status === 'ok');
  const items = ai.providers.filter((p) => p.status === 'failing').map((p) => ({
    key: `ai-${p.provider}`, severity: answering.length ? 'info' : 'high', attention: !answering.length,
    message: `${PROVIDER[p.provider] || p.provider} is not answering${p.model ? ` (${p.model})` : ''}: ${p.last_error || 'no reply'}.`,
    hint: `${p.advice || ''} ${answering.length
      ? `${answering.map((a) => PROVIDER[a.provider] || a.provider).join(' and ')} is answering, so reviews continue.`
      : 'Until one answers, trades go ahead without an AI review and pictures cannot be read.'}`.trim(),
  }));
  if (answering.length) {
    items.push({
      key: 'ai-ok', severity: 'info', attention: false,
      message: `AI answering: ${answering.map((a) => `${PROVIDER[a.provider] || a.provider}${a.model ? ` (${a.model})` : ''}, last at ${clock(a.last_ok)}${a.calls_today ? `, ${a.calls_today} calls and ${a.tokens_today ? `${Math.round(a.tokens_today / 1000)}k tokens` : 'some tokens'} used today` : ''}`).join('; ')}.`,
    });
  }
  return items;
}

// The one list the bell counts and this page shows.
export function notificationItems({ anomalies = [], pendingApprovals = 0, status = {}, staleDashboard = false, ai = null,
  agentAlerts = null, alertsSeenAt = null }) {
  const items = [];
  if (staleDashboard) {
    items.push({
      key: 'stale', severity: 'high', attention: true,
      message: 'This dashboard is out of date: the server is running different code.',
      hint: 'In Coolify, redeploy the dashboard (frontend) service, then reload this page with Ctrl + Shift + R.',
    });
  }
  if (status.trading_halted) {
    items.push({
      key: 'halted', severity: 'high', attention: true,
      message: `Trading is halted${status.halt_reason ? ` (${status.halt_reason})` : ''}: no new orders are being placed.`,
      hint: 'Protective stop-losses still work. Press RESUME in the header when you are ready.',
    });
  }
  if (pendingApprovals > 0) {
    items.push({
      key: 'approvals', severity: 'medium', attention: true, go: 'research',
      message: `${pendingApprovals} research rating${pendingApprovals > 1 ? 's are' : ' is'} waiting for your decision in the approval queue.`,
      hint: 'Open Research to approve or reject.',
    });
  }
  items.push(...agentAlertItems({ agentAlerts, seenAt: alertsSeenAt, status }));
  items.push(...aiItems(ai));
  items.push(...reviewItems(ai && ai.review));
  anomalies.forEach((a, i) => items.push({
    key: `a${i}`, severity: a.severity, attention: a.attention ?? ['high', 'medium'].includes(a.severity),
    message: a.message, hint: a.hint, symbol: a.symbol, lastAt: a.detail?.last_at,
  }));
  return items;
}

function Item({ item, onDrill, onGo }) {
  const act = item.symbol ? () => onDrill(item.symbol) : item.go ? () => onGo(item.go) : null;
  return (
    <div onClick={act || undefined} style={{
      display: 'flex', gap: '12px', padding: '12px 4px', borderBottom: `1px solid ${theme.colors.border}`,
      cursor: act ? 'pointer' : 'default', alignItems: 'flex-start',
    }}>
      <span style={{ width: '9px', height: '9px', borderRadius: '50%', background: DOT[item.severity] || theme.colors.textMuted,
                     flexShrink: 0, marginTop: '5px' }} />
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ fontSize: '13px', color: theme.colors.text }}>{item.message}</div>
        {item.hint && <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginTop: '3px', whiteSpace: 'pre-line' }}>{item.hint}</div>}
        {item.lastAt && <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '2px' }}>{item.lastLabel || 'Last seen'} {localTime(item.lastAt)}</div>}
      </div>
      {act && (
        <span style={{ fontSize: '11px', color: theme.colors.textSecondary, whiteSpace: 'nowrap' }}>
          {item.symbol ? 'Open chart ↗' : 'Open ↗'}
        </span>
      )}
    </div>
  );
}

const LEVELS = [['all', 'All'], ['critical', 'Critical'], ['warning', 'Warning'], ['info', 'Info']];
const LEVEL_DOT = { critical: theme.colors.danger, warning: theme.colors.warning, info: theme.colors.accent };

const chip = (on) => ({
  background: on ? `${theme.colors.primary}22` : 'rgba(255,255,255,0.04)',
  border: `1px solid ${on ? `${theme.colors.primary}80` : theme.colors.border}`,
  color: on ? '#fff' : theme.colors.textSecondary, borderRadius: '999px',
  padding: '5px 12px', fontSize: '11px', fontWeight: 800, cursor: 'pointer',
});

// Everything the agent has raised in the last week, newest first, whether or
// not email is set up.
function AgentAlerts({ agentAlerts, seenAt, isOperator, mobile }) {
  const [level, setLevel] = useState('all');
  if (!agentAlerts) return null;
  const alerts = agentAlerts.alerts || [];
  const unread = new Set(unreadAlerts(alerts, seenAt).map((a) => a.id));
  const shown = level === 'all' ? alerts : alerts.filter((a) => a.severity === level);
  const email = agentAlerts.email || {};
  return (
    <div style={card(mobile)}>
      <SectionHeader title="Recent agent alerts" icon={Bell} />
      <div style={{ fontSize: '12px', color: email.configured ? theme.colors.textMuted : theme.colors.warning, marginBottom: '12px', lineHeight: 1.5 }}>
        {email.configured
          ? `Email alerts are on (${(email.recipients || []).join(', ')}); ${email.sent_24h || 0} sent in the last day.`
          : 'Email alerts are off, so you see these here, and only while this page is open. To also be emailed, set ALERT_EMAIL_TO and the SMTP_ variables in Coolify and redeploy.'}
      </div>
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '10px' }}>
        {LEVELS.map(([id, label]) => {
          const n = id === 'all' ? alerts.length : alerts.filter((a) => a.severity === id).length;
          return <button key={id} onClick={() => setLevel(id)} style={chip(level === id)}>{label} {n}</button>;
        })}
      </div>
      {shown.length === 0 && <Empty>{alerts.length ? 'None at this level.' : 'The agent has raised no alerts in the last week.'}</Empty>}
      {shown.map((a) => (
        <div key={a.id} style={{ display: 'flex', gap: '12px', padding: '12px 4px', borderBottom: `1px solid ${theme.colors.border}`, alignItems: 'flex-start' }}>
          <span style={{ width: '9px', height: '9px', borderRadius: '50%', background: LEVEL_DOT[a.severity] || theme.colors.textMuted, flexShrink: 0, marginTop: '5px' }} />
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: '13px', color: theme.colors.text, fontWeight: unread.has(a.id) ? 800 : 500 }}>
              {a.subject}{unread.has(a.id) && <span style={{ marginLeft: '8px', fontSize: '10px', color: theme.colors.warning }}>NEW</span>}
            </div>
            {isOperator && a.body && (
              <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginTop: '3px', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{a.body}</div>
            )}
            <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '2px' }}>
              {localTime(a.ts)}{a.emailed === false ? ' · email could not be sent' : a.emailed === true ? ' · emailed' : ''}
            </div>
          </div>
        </div>
      ))}
      {!isOperator && alerts.length > 0 && (
        <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '10px' }}>The details of each alert are shown to the operator only.</div>
      )}
    </div>
  );
}

export default function NotificationsView({ items, isOperator, mobile, onDrill, onGo, agentAlerts = null, alertsSeenAt = null, onMarkRead }) {
  const attention = items.filter((i) => i.attention);
  const info = items.filter((i) => !i.attention);
  const unreadCount = agentAlerts ? unreadAlerts(agentAlerts.alerts, alertsSeenAt).length : 0;
  const markRead = unreadCount > 0 && onMarkRead ? (
    <button onClick={onMarkRead} style={chip(false)}>Mark {unreadCount} alert{unreadCount > 1 ? 's' : ''} as read</button>
  ) : null;
  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      <div style={card(mobile)}>
        <SectionHeader title="Needs your attention" icon={AlertTriangle} right={markRead} />
        {attention.length
          ? attention.map((i) => <Item key={i.key} item={i} onDrill={onDrill} onGo={onGo} />)
          : <Empty>Nothing needs you right now.</Empty>}
      </div>
      <div style={card(mobile)}>
        <SectionHeader title="For your information" icon={Info} />
        {info.length
          ? info.map((i) => <Item key={i.key} item={i} onDrill={onDrill} onGo={onGo} />)
          : <Empty>Nothing to report.</Empty>}
      </div>
      <AgentAlerts agentAlerts={agentAlerts} seenAt={alertsSeenAt} isOperator={isOperator} mobile={mobile} />
      <div style={card(mobile)}>
        <SectionHeader title="How this page works" icon={Bell} />
        <div style={{ fontSize: '12px', color: theme.colors.textSecondary, display: 'grid', gap: '6px' }}>
          <div>
            The agent checks every stock about once a minute and logs the outcome when it changes, and every half hour
            while it does not. A &quot;check&quot; below is one of those log lines, not an order: nothing is sent to a broker unless
            a trade is actually placed. Only the last 12 hours of checks count, so a problem that has been fixed drops off.
          </div>
          <div>
            <strong>Needs your attention</strong>{' '}means something stopped a trade the agent should have made, or a decision
            is waiting for you. <strong>For your information</strong>{' '}means the agent&apos;s own rules held a signal back on
            purpose, such as not adding to a holding until it has earned it, or never betting on a fall.
          </div>
          <div>
            <strong>Agent alerts</strong>{' '}are what the agent tells you about itself: a halt, a worker restarted, no price for
            a symbol, a crash. New ones stay under &quot;Needs your attention&quot; until you press &quot;Mark as read&quot;;
            things that are wrong right now clear by themselves when the agent has fixed them. All of them stay listed in
            &quot;Recent agent alerts&quot; for a week.
          </div>
          {!isOperator && <div>Signal-level notices are shown to the operator only.</div>}
          <div>The bell in the header counts the items that need your attention.</div>
        </div>
      </div>
    </div>
  );
}
