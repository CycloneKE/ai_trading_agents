// Does the AI's trade review help? Signals it approved versus those it
// vetoed, measured 5 and 20 trading days later (/api/ai/scorecard).
import { useEffect, useState } from 'react';
import { Scale } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { apiGet, card, pct, gainColor, SectionHeader, Empty, Note, th, td, tableHeadRow } from '../ui';

const cell = (g) => (g && g.n
  ? <><strong style={{ color: gainColor(g.avg_return_pct) }}>{pct(g.avg_return_pct)}</strong>
      <span style={{ color: theme.colors.textMuted }}> · {Math.round((g.win_rate || 0) * 100)}% up · {g.n}</span></>
  : <span style={{ color: theme.colors.textMuted }}>no results yet</span>);

// Which AI reviews trades, and this month's spending against the cap
// (/api/ai/budget). Claude is the paid tier, off without ANTHROPIC_API_KEY.
function budgetLine(b) {
  if (!b) return null;
  if (!b.paid_tier) return b.note;
  const used = b.cap_usd ? Math.round((b.spent_usd / b.cap_usd) * 100) : 0;
  return `Paid tier on: ${b.review_model} reviews trades, ${b.volume_model} does volume work. `
    + `${b.month}: $${Number(b.spent_usd || 0).toFixed(2)} of $${Number(b.cap_usd || 0).toFixed(2)} spent (${used}%) over ${b.calls} call${b.calls === 1 ? '' : 's'}`
    + (b.remaining_usd <= 0 ? '; the cap is reached, so the free models review trades until next month.' : '.');
}

export default function AiScorecard({ mobile }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [budget, setBudget] = useState(null);

  useEffect(() => {
    apiGet('/api/ai/scorecard')
      .then(({ ok, status, json }) => (ok ? setData(json) : setError((json && json.error) || `HTTP ${status}`)))
      .catch((e) => setError(e.message));
    apiGet('/api/ai/budget')
      .then(({ ok, json }) => ok && setBudget(json))
      .catch(() => {});
  }, []);

  return (
    <div style={card(mobile, { marginBottom: mobile ? '16px' : '24px' })}>
      <SectionHeader title="Does the AI's review help?" icon={Scale} />
      {budget && (
        <div style={{ fontSize: '12px', color: budget.paid_tier && budget.remaining_usd <= 0 ? theme.colors.warning : theme.colors.textMuted, marginBottom: '10px' }}>
          {budgetLine(budget)}
        </div>
      )}
      {!data && (error ? <Empty>The scorecard could not load ({error}).</Empty> : <Empty>Scoring past reviews…</Empty>)}
      {data && (
        <>
          <Note tone={data.verdict.startsWith('The review is costing') ? theme.colors.warning : null}>
            {data.verdict}
          </Note>
          <div style={{ fontSize: '12px', color: theme.colors.textMuted, margin: '10px 0' }}>
            The AI reviewed {data.reviewed} signal{data.reviewed === 1 ? '' : 's'}: {data.approved} approved, {data.vetoed} vetoed.
            Returns are in each signal's direction, before costs; a sell followed by a fall counts as a gain.
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={tableHeadRow}>
                  <th style={th}>AFTER</th><th style={th}>APPROVED</th><th style={th}>VETOED</th><th style={th}>DIFFERENCE</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.horizons).map(([h, row]) => (
                  <tr key={h} style={{ borderBottom: `1px solid ${theme.colors.border}` }}>
                    <td style={{ ...td, fontWeight: 700 }}>{h} trading days</td>
                    <td style={td}>{cell(row.approved)}</td>
                    <td style={td}>{cell(row.vetoed)}</td>
                    <td style={{ ...td, color: gainColor(row.veto_edge_pct), fontWeight: 700 }}>
                      {row.veto_edge_pct == null ? '—' : `${pct(row.veto_edge_pct)} pts`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
