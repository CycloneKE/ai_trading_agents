// Notifications: everything the agent wants the operator to know, in one
// place, split into what needs a decision and what is only for information.
// Built by the dashboard shell (notificationItems) from the anomaly scan,
// the approval queue, the halt switch and the dashboard's version check.
import { Bell, AlertTriangle, Info } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { card, SectionHeader, Empty, localTime } from '../ui';

const DOT = { high: theme.colors.danger, medium: theme.colors.warning, low: theme.colors.textMuted, info: theme.colors.accent };

const PROVIDER = { anthropic: 'Claude', groq: 'Groq', gemini: 'Gemini', openrouter: 'OpenRouter' };
const clock = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : '');

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
      message: `AI answering: ${answering.map((a) => `${PROVIDER[a.provider] || a.provider}${a.model ? ` (${a.model})` : ''}, last at ${clock(a.last_ok)}`).join('; ')}.`,
    });
  }
  return items;
}

// The one list the bell counts and this page shows.
export function notificationItems({ anomalies = [], pendingApprovals = 0, status = {}, staleDashboard = false, ai = null }) {
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
  items.push(...aiItems(ai));
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
        {item.hint && <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginTop: '3px' }}>{item.hint}</div>}
        {item.lastAt && <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '2px' }}>Last seen {localTime(item.lastAt)}</div>}
      </div>
      {act && (
        <span style={{ fontSize: '11px', color: theme.colors.textSecondary, whiteSpace: 'nowrap' }}>
          {item.symbol ? 'Open chart ↗' : 'Open ↗'}
        </span>
      )}
    </div>
  );
}

export default function NotificationsView({ items, isOperator, mobile, onDrill, onGo }) {
  const attention = items.filter((i) => i.attention);
  const info = items.filter((i) => !i.attention);
  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      <div style={card(mobile)}>
        <SectionHeader title="Needs your attention" icon={AlertTriangle} />
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
          {!isOperator && <div>Signal-level notices are shown to the operator only.</div>}
          <div>The bell in the header counts the items that need your attention.</div>
        </div>
      </div>
    </div>
  );
}
