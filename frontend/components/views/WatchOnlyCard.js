// Watch-only markets (/api/watch/commodities, src/agent/commodity_watch.py):
// commodities and the UK, Japanese, French and South African stock markets,
// followed with no orders, and three patterns scored against what prices then did. Backfill is replayed history (a candidate, not
// proof); live is what has happened since the watch began (the real test).
import { useEffect, useState } from 'react';
import { Eye } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { apiGet, card, SectionHeader, Empty, Note } from '../ui';

const PATTERN = {
  trend_200d: 'Trend (200-day average)',
  momentum_12m: 'Momentum (12 months)',
  seasonal: 'Seasonal (same month, earlier years)',
};

const VERDICT = {
  candidate: { label: 'CANDIDATE', color: () => theme.colors.primary },
  no_clear_edge: { label: 'no clear edge', color: () => theme.colors.textSecondary },
  wrong_way: { label: 'WRONG WAY', color: () => theme.colors.danger },
  too_few: { label: 'too few signals', color: () => theme.colors.textMuted },
};

const KIND = { futures: 'futures', fund: 'fund', stock: 'stock', index: 'index' };
const cell = { padding: '6px 8px' };
const head = { ...cell, textAlign: 'left' };
const signed = (v, digits = 1) => (v == null ? 'n/a' : `${v > 0 ? '+' : ''}${Number(v).toFixed(digits)}`);
const label = (s) => String(s).replace(/_/g, ' ');
// A stock market added later has its own start day; say so when they differ.
const costsText = (data) => {
  const c = data.costs_pct || {};
  const parts = Object.keys(c).map((k) => `${label(k)} ${c[k]}%`);
  return `commodities ${data.assumed_cost_pct}%${parts.length ? `, ${parts.join(', ')}` : ''}`;
};
const startedBy = (data) => {
  const g = data.started_by_group || {};
  const parts = Object.keys(g).filter((k) => g[k] !== data.started).map((k) => `${label(k)} from ${g[k]}`);
  return parts.length ? ` (${parts.join(', ')})` : '';
};

const Table = ({ children }) => (
  <div style={{ overflowX: 'auto', marginTop: '8px' }}>
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>{children}</table>
  </div>
);

const Head = ({ cols }) => (
  <thead>
    <tr style={{ color: theme.colors.textMuted, fontSize: '11px', borderBottom: `1px solid ${theme.colors.border}` }}>
      {cols.map((c) => <th key={c} style={head}>{c}</th>)}
    </tr>
  </thead>
);

export const PatternTable = ({ rows }) => (
  <Table>
    <Head cols={['PATTERN', 'GROUP', 'SIGNALS', 'RIGHT', 'CHANCE', 'Z', 'EARLY / LATE HALF', 'NET MOVE', 'VERDICT']} />
    <tbody>
      {rows.map((r) => {
        const v = VERDICT[r.verdict] || VERDICT.no_clear_edge;
        const pctOf = (x) => (x == null ? 'n/a' : `${x > 0 ? '+' : ''}${(x * 100).toFixed(0)}%`);
        return (
          <tr key={`${r.source}-${r.market}`} style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary }}>
            <td style={{ ...cell, whiteSpace: 'nowrap' }}>{PATTERN[r.source] || label(r.source)}</td>
            <td style={{ ...cell, whiteSpace: 'nowrap' }}>{label(r.market)}</td>
            <td style={cell}>{r.n}</td>
            <td style={cell}>{Math.round(r.hit_rate * 100)}%</td>
            <td style={cell}>{Math.round(r.chance * 100)}%</td>
            <td style={cell}>{signed(r.z)}</td>
            <td style={{ ...cell, whiteSpace: 'nowrap' }}>{pctOf(r.early_lift)} / {pctOf(r.late_lift)}</td>
            <td style={cell}>{signed(r.net_move_pct, 2)}%</td>
            <td style={{ ...cell, fontWeight: 700, color: v.color(), whiteSpace: 'nowrap' }}>{v.label}</td>
          </tr>
        );
      })}
    </tbody>
  </Table>
);

const Reading = ({ s }) => {
  if (!s.enough_history) return <span style={{ color: theme.colors.textMuted }}>not enough history</span>;
  return (
    <span>
      {s.above_200d ? 'above' : 'below'} its 200-day average ({signed(s.pct_from_200d)}%)
    </span>
  );
};

export const Instruments = ({ items }) => (
  <Table>
    <Head cols={['INSTRUMENT', 'MARKET', 'KIND', 'LAST PRICE', 'TREND', '12-MONTH MOVE', 'THIS MONTH, EARLIER YEARS']} />
    <tbody>
      {items.map((s) => (
        <tr key={s.symbol} style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary }}>
          <td style={{ ...cell, whiteSpace: 'nowrap' }}>{s.name} <span style={{ color: theme.colors.textMuted }}>({s.symbol})</span></td>
          <td style={{ ...cell, whiteSpace: 'nowrap' }}>{label(s.group || '')}</td>
          <td style={cell}>{KIND[s.kind] || s.kind}</td>
          <td style={{ ...cell, whiteSpace: 'nowrap' }}>
            {s.last_close ?? 'n/a'}{s.quote && s.last_close != null ? ` ${s.quote}` : ''}
            {s.unit_fixes > 0 && (
              <span title={`The price source switched units ${s.unit_fixes} time(s) in this history (for example pounds and pence); it was corrected before scoring.`}
                style={{ color: theme.colors.warning }}> ⚠</span>
            )}
          </td>
          <td style={cell}><Reading s={s} /></td>
          <td style={cell}>{s.momentum_12m_pct == null ? 'n/a' : `${signed(s.momentum_12m_pct)}%`}</td>
          <td style={cell}>
            {s.seasonal
              ? `up in ${Math.round(s.seasonal.share_up * 100)}% of ${s.seasonal.years} years, average ${signed(s.seasonal.avg_pct)}%`
              : 'n/a'}
          </td>
        </tr>
      ))}
    </tbody>
  </Table>
);

export default function WatchOnlyCard({ mobile }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = () => apiGet('/api/watch/commodities')
      .then(({ ok, status, json }) => {
        if (!alive) return;
        if (ok && json && json.backfill) { setData(json); setError(null); }
        else setError((json && json.error) || `HTTP ${status}`);
      })
      .catch((e) => alive && setError(e.message));
    load();
    const id = setInterval(load, 600000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!data) {
    return (
      <div style={card(mobile)}>
        <SectionHeader title="Watch only: commodities and stock markets" icon={Eye} />
        {error ? <Empty>The market watch could not load ({error}).</Empty> : <Empty>Loading the market watch…</Empty>}
      </div>
    );
  }
  if (!data.enabled) {
    return (
      <div style={card(mobile)}>
        <SectionHeader title="Watch only: commodities and stock markets" icon={Eye} />
        <Empty>The market watch is switched off (watch_only in config.json).</Empty>
      </div>
    );
  }

  const live = data.live || { signals: 0, table: [] };
  const back = data.backfill || { signals: 0, table: [] };
  const started = data.started || 'today';
  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      <div style={card(mobile)}>
        <SectionHeader title="Watch only: commodities and stock markets (no orders are placed)" icon={Eye} />
        <Note tone={theme.colors.accent}>
          The agent follows commodities and the UK, Japanese, French and South African stock markets and records
          what three well-known patterns would have said, then what each actually did {data.horizon} trading days
          later, against how often it simply rose anyway. Nothing here trades: no broker reaches these markets.
          Watching began {started}{startedBy(data)}.
        </Note>
        <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '10px', lineHeight: 1.5 }}>
          &quot;Candidate&quot; means the pattern beat chance by two standard errors in all the history and in both
          halves of it. It earns a place on the watch, nothing more: only the live record below confirms it. Many
          rows are tested at once, so expect a few to look good by luck.
        </div>
      </div>

      <div style={card(mobile)}>
        <SectionHeader title={`Live record since ${started} (the real test)`} icon={Eye} />
        {live.table.length === 0
          ? <Empty>No signal has a known outcome yet. Each outcome arrives {data.horizon} trading days after its signal,
              and the patterns signal about once a month per commodity, so expect weeks before anything shows here.</Empty>
          : <PatternTable rows={live.table} />}
      </div>

      <div style={card(mobile)}>
        <SectionHeader title="Replayed history (a candidate, not proof)" icon={Eye} />
        {back.table.length === 0
          ? <Empty>The history has not been replayed yet. It runs shortly after the agent starts.</Empty>
          : <>
              <PatternTable rows={back.table} />
              <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '8px', lineHeight: 1.5 }}>
                {back.signals} signals replayed. Markets move together, so the true count is lower than it looks.
                Net move takes off each market&apos;s assumed cost per signal ({costsText(data)}); the stock-market costs
                are unverified assumptions. Futures prices jump when a contract rolls over, and oil and gas funds lose
                value from rolling, so judge a fund by its own price. The companies listed are today&apos;s large ones,
                which flatters their history (survivorship): the index and index-fund rows are the fairer test. Results are
                in each market&apos;s own currency, before the exchange rate a dollar-based investor would also bear.
              </div>
            </>}
      </div>

      <div style={card(mobile)}>
        <SectionHeader title="What each pattern says right now" icon={Eye} />
        {data.instruments.length === 0
          ? <Empty>No prices have been loaded yet.</Empty>
          : <Instruments items={data.instruments} />}
        {data.updated_at && (
          <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '8px' }}>Updated {data.updated_at.replace('T', ' ')}</div>
        )}
      </div>
    </div>
  );
}
