// Market Scan: every NSE stock screened from its real history, and the
// short list the agent trades this week (/api/nse/scan, nse_screener.py).
import { useEffect, useState } from 'react';
import { Search } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { apiGet, card, pct, gainColor, SectionHeader, Empty, Note, th, td, tableHeadRow } from '../ui';

const ROLE = {
  holding: { label: 'HOLDING', note: 'Held in the paper account; always kept so its exits and stops keep working.' },
  top: { label: 'TOP RANKED', note: 'Among the best-ranked eligible stocks, within the sector limit.' },
  exploration: { label: 'EXPLORATION', note: 'Ranked just below the cut; rotates weekly so the agent learns beyond the obvious names.' },
  fallback: { label: 'CONFIGURED', note: 'From the configured list, kept while too few stocks have enough history.' },
  configured: { label: 'CONFIGURED', note: 'The configured list; the screener has not built a short list yet.' },
};

const kes = (v) => (v == null ? '—' : `KES ${Math.round(v).toLocaleString()}`);

function Shortlist({ scan, mobile, onDrill }) {
  const sl = scan.shortlist;
  const bySym = Object.fromEntries((scan.stocks || []).map((s) => [s.symbol, s]));
  return (
    <div style={card(mobile, { marginBottom: mobile ? '16px' : '24px' })}>
      <SectionHeader title="This week's short list" icon={Search} />
      <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '10px' }}>
        {sl
          ? `Built ${new Date(sl.built_at).toLocaleString()}, rebuilt every ${scan.rules.refresh_days} days. `
          : 'Not built yet: the agent builds it on its next NSE check during trading hours. '}
        Up to {scan.rules.max_symbols} stocks, at most {scan.rules.max_per_sector} from one sector.
        {' '}{scan.eligible} of {(scan.stocks || []).length} stocks are eligible now.
      </div>
      <div style={{ display: 'grid', gap: '8px' }}>
        {(scan.traded || []).map((sym) => {
          const role = ROLE[scan.roles?.[sym]] || ROLE.configured;
          const m = bySym[sym] || {};
          return (
            <div key={sym} onClick={() => onDrill(sym)} title={role.note}
                 style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', alignItems: 'center', cursor: 'pointer',
                          padding: '8px 10px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', border: `1px solid ${theme.colors.border}` }}>
              <div>
                <strong>{sym}</strong>
                <span style={{ marginLeft: '8px', fontSize: '11px', color: theme.colors.textMuted }}>{m.name || ''}{m.sector ? ` · ${m.sector}` : ''}</span>
              </div>
              <span style={{ fontSize: '9px', fontWeight: 800, padding: '2px 6px', borderRadius: '4px', whiteSpace: 'nowrap',
                             background: `${theme.colors.accent}25`, color: theme.colors.accent }}>{role.label}</span>
            </div>
          );
        })}
      </div>
      {sl && sl.reviewed_by_ai && (
        <div style={{ marginTop: '12px' }}>
          <Note>
            <strong>AI review:</strong>{' '}
            {sl.removed && sl.removed.length
              ? sl.removed.map((r) => `removed ${r.symbol} (${r.reason})`).join('; ')
              : 'no changes.'}
            {sl.notes ? ` ${sl.notes}` : ''}
          </Note>
        </div>
      )}
    </div>
  );
}

export default function NseMarketScan({ active, mobile, onDrill }) {
  const [scan, setScan] = useState(null);
  const [error, setError] = useState(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    if (!active) return;
    apiGet('/api/nse/scan')
      .then(({ ok, status, json }) => (ok ? setScan(json) : setError((json && json.error) || `HTTP ${status}`)))
      .catch((e) => setError(e.message));
  }, [active]);

  if (!scan) {
    return <div style={card(mobile)}>{error ? <Empty>The scan could not load ({error}).</Empty> : <Empty>Screening the exchange…</Empty>}</div>;
  }
  const traded = new Set(scan.traded || []);
  const stocks = (scan.stocks || []).filter((s) => showAll || s.eligible || traded.has(s.symbol));
  return (
    <div>
      {!scan.enabled && (
        <div style={{ marginBottom: '12px' }}>
          <Note tone={theme.colors.warning}>The screener is off (nse_screener.enabled); the agent trades the configured list.</Note>
        </div>
      )}
      <Shortlist scan={scan} mobile={mobile} onDrill={onDrill} />
      <div style={card(mobile)}>
        <SectionHeader title="Market Scan" icon={Search} />
        <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '10px' }}>
          Eligible: at least {scan.rules.min_history_days} days of real prices and {kes(scan.rules.min_avg_value_kes)} traded a day.
          Score (0 to 1): half momentum, a quarter trend (price above its 50-day average), a quarter liquidity, each ranked
          against the other eligible stocks. Momentum is the return over about six months, skipping the latest month,
          or over the history there is.
        </div>
        <button onClick={() => setShowAll((v) => !v)} style={{
          marginBottom: '12px', background: 'rgba(255,255,255,0.05)', color: theme.colors.textSecondary, border: 'none',
          padding: '6px 12px', borderRadius: '6px', fontSize: '11px', fontWeight: 800, cursor: 'pointer' }}>
          {showAll ? 'ELIGIBLE ONLY' : `SHOW ALL ${(scan.stocks || []).length}`}
        </button>
        {stocks.length ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={tableHeadRow}>
                  <th style={th}>#</th><th style={th}>STOCK</th><th style={th}>SCORE</th><th style={th}>MOMENTUM</th>
                  {!mobile && <th style={th}>1 MONTH</th>}
                  {!mobile && <th style={th}>TREND</th>}
                  {!mobile && <th style={th}>TRADED A DAY</th>}
                  {!mobile && <th style={th} title="From the latest AIB-AXYS Market Pulse uploaded">P/E · YIELD</th>}
                  <th style={th}>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {stocks.map((s) => (
                  <tr key={s.symbol} onClick={() => onDrill(s.symbol)} style={{ borderBottom: `1px solid ${theme.colors.border}`, cursor: 'pointer',
                                                                             background: traded.has(s.symbol) ? `${theme.colors.accent}12` : 'transparent' }}>
                    <td style={{ ...td, color: theme.colors.textMuted }}>{s.rank || '—'}</td>
                    <td style={{ ...td, fontWeight: 700 }}>
                      {s.symbol}
                      <div style={{ fontSize: '10px', fontWeight: 400, color: theme.colors.textMuted }}>{s.name !== s.symbol ? s.name : ''}{s.sector ? ` · ${s.sector}` : ''}</div>
                    </td>
                    <td style={{ ...td, fontWeight: 700 }}>{s.score == null ? '—' : s.score.toFixed(2)}</td>
                    <td style={{ ...td, color: gainColor(s.momentum_pct) }} title={s.momentum_window || ''}>{s.momentum_pct == null ? '—' : pct(s.momentum_pct, 1)}</td>
                    {!mobile && <td style={{ ...td, color: gainColor(s.return_1m_pct) }}>{s.return_1m_pct == null ? '—' : pct(s.return_1m_pct, 1)}</td>}
                    {!mobile && <td style={td}>{s.above_sma50 == null ? '—' : (s.above_sma50 ? 'Up' : 'Down')}</td>}
                    {!mobile && <td style={{ ...td, whiteSpace: 'nowrap' }}>{kes(s.avg_value_kes)}</td>}
                    {!mobile && (
                      <td style={{ ...td, whiteSpace: 'nowrap', color: theme.colors.textSecondary }}
                          title={s.fundamentals_as_of ? `AIB-AXYS Market Pulse, ${s.fundamentals_as_of}` : 'Upload a Market Pulse to fill this in'}>
                        {s.fundamentals_as_of ? `${s.pe ? `${s.pe.toFixed(1)}x` : 'loss'} · ${(s.dividend_yield_pct || 0).toFixed(1)}%` : '—'}
                      </td>
                    )}
                    <td style={{ ...td, fontSize: '11px', color: s.eligible ? theme.colors.primary : theme.colors.textMuted }}>
                      {traded.has(s.symbol) ? (ROLE[scan.roles?.[s.symbol]] || ROLE.configured).label : (s.eligible ? 'Eligible' : s.reason)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No stock is eligible yet. Stocks qualify as their real price history reaches {scan.rules.min_history_days} days.</Empty>
        )}
      </div>
    </div>
  );
}
