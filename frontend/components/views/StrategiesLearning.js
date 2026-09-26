// Strategies & Learning: which strategies vote in which market, their
// weight in the blend, their results so far, and what the agent has changed
// about them on its own (/api/strategies/learning).
import { useEffect, useState } from 'react';
import { Brain, History } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { apiGet, card, money, gainColor, SectionHeader, Empty, Note, th, td, tableHeadRow } from '../ui';

const MARKET = { us_equity: 'US', crypto: 'Crypto', nse: 'NSE' };

const params = (p) => (p && typeof p === 'object'
  ? Object.entries(p).map(([k, v]) => `${k} ${v}`).join(', ')
  : String(p ?? '—'));

function when(ts) {
  if (!ts) return 'never';
  // The tuner's older log entries are UTC without a zone mark; read them as UTC.
  const iso = typeof ts === 'string' && !/(Z|[+-]\d\d:?\d\d)$/.test(ts) ? `${ts}Z` : ts;
  const d = typeof iso === 'number' ? new Date(iso * 1000) : new Date(iso);
  return Number.isNaN(d.getTime()) ? String(ts) : d.toLocaleString();
}

export default function StrategiesLearning({ mobile }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    apiGet('/api/strategies/learning')
      .then(({ ok, status, json }) => (ok ? setData(json) : setError((json && json.error) || `HTTP ${status}`)))
      .catch((e) => setError(e.message));
  }, []);

  if (!data) {
    return <div style={card(mobile)}>{error ? <Empty>This view could not load ({error}).</Empty> : <Empty>Loading the strategies…</Empty>}</div>;
  }
  const exclusive = Object.entries(data.strategy_markets || {});
  const log = data.tuner?.log || [];
  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      <div style={card(mobile)}>
        <SectionHeader title="Strategies" icon={Brain} />
        <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '10px' }}>
          Each strategy votes buy, sell or hold on the markets shown. Votes are blended by weight
          ({data.ensemble_method || 'weighted'}); a trade needs a blended confidence of at least
          {' '}{Math.round((data.min_trade_confidence || 0.5) * 100)}%. A strategy with no view abstains rather than voting hold.
          {data.regime_filter ? ' Momentum votes only in trending markets and mean reversion only in ranging ones.' : ''}
          {exclusive.length > 0 && ` ${exclusive.map(([m, names]) => `${MARKET[m] || m} is traded only by ${names.join(', ')}`).join('; ')}.`}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={tableHeadRow}>
                <th style={th}>STRATEGY</th><th style={th}>MARKETS</th><th style={th}>WEIGHT</th>
                {!mobile && <th style={th}>NEEDS</th>}
                <th style={th}>CLOSED TRADES</th><th style={th}>WIN RATE</th><th style={th}>REALISED</th>
              </tr>
            </thead>
            <tbody>
              {data.strategies.map((s) => (
                <tr key={s.name} title={s.note || ''} style={{ borderBottom: `1px solid ${theme.colors.border}` }}>
                  <td style={{ ...td, fontWeight: 700 }}>{s.name}<div style={{ fontSize: '10px', fontWeight: 400, color: theme.colors.textMuted }}>{s.type}</div></td>
                  <td style={td}>{s.markets.length ? s.markets.map((m) => MARKET[m] || m).join(', ') : '—'}</td>
                  <td style={td}>{s.weight_pct == null ? '—' : `${s.weight_pct}%`}</td>
                  {!mobile && <td style={{ ...td, whiteSpace: 'nowrap' }}>{s.history_days ? `${s.history_days} days` : '—'}</td>}
                  <td style={td}>{s.closed_trades}</td>
                  <td style={td}>{s.closed_trades ? `${Math.round((s.win_rate || 0) * 100)}%` : '—'}</td>
                  <td style={{ ...td, color: gainColor(s.realized_pnl), whiteSpace: 'nowrap' }}>{s.closed_trades ? money(s.realized_pnl) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '8px' }}>
          Weights shift towards strategies whose closed trades earn more. Realised profit is in US dollars, with NSE
          results converted at the live KES rate. Hover over a row for the strategy's description.
        </div>
      </div>

      <div style={card(mobile)}>
        <SectionHeader title="What the agent has learned" icon={History} />
        <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '10px' }}>
          The tuner re-tests each strategy's settings on held-out history, weekly or sooner for a losing strategy, and
          adopts a change only when it beats the current settings. Last run: {when(data.tuner?.last_run)}.
        </div>
        {Object.keys(data.tuner?.params || {}).length > 0 && (
          <div style={{ marginBottom: '12px' }}>
            <Note>
              <strong>Settings in use:</strong>{' '}
              {Object.entries(data.tuner.params).map(([n, p]) => `${n} ${Object.entries(p).map(([k, v]) => `${k}=${v}`).join(', ')}`).join('; ')}
            </Note>
          </div>
        )}
        {data.tuner?.restricted ? (
          <Empty>The tuner's settings and changes are shown to the operator only.</Empty>
        ) : log.length ? (
          <div style={{ display: 'grid', gap: '6px' }}>
            {log.map((e, i) => (
              <div key={i} style={{ fontSize: '12px', padding: '6px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
                <span style={{ color: theme.colors.textMuted }}>{when(e.at)}</span>{' '}
                <strong>{e.strategy}</strong>{': '}
                {e.outcome === 'adopted'
                  ? <span style={{ color: theme.colors.primary }}>changed {params(e.before)} to {params(e.after)}</span>
                  : <span style={{ color: theme.colors.textMuted }}>{e.outcome || 'checked'}</span>}
              </div>
            ))}
          </div>
        ) : (
          <Empty>No tuning runs recorded yet. The first runs a week after start, once each strategy has closed trades to learn from.</Empty>
        )}
      </div>
    </div>
  );
}
