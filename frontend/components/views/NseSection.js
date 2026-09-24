// NSE Kenya: Market Watch, the agent's Paper Account and, for operators,
// the manual Order Tickets, as three tabs of one section.
import { useCallback, useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Activity, Shield, Globe, Flag, Layers } from 'lucide-react';
import { theme } from '../DashboardStyles';
import {
  apiGet, card, money, pct, gainColor, fxSourceNote, SectionHeader, HUDCard, HudRow, SubTabs, Empty, Note,
  th, td, tableHeadRow,
} from '../ui';
import NsePaperAccount from './NsePaperAccount';
import { getApiBase } from '../../utils/apiBase';

// Sources the scraper stamps on prices it actually fetched. Mirrors
// REAL_NSE_SOURCES in src/connectors/nse_scraper.py.
const REAL_NSE_SOURCES = ['nse_ticker', 'nse_pricelist', 'nse_website', 'afx_kwayisi', 'afx_history'];
const NSE_SOURCE_NOTE = {
  synthetic: 'Generated seed data, not a market price. No live source has answered for this symbol yet.',
  none: 'No price available for this symbol.',
};
const isRealNsePrice = (q) => REAL_NSE_SOURCES.includes(q.source);

// The session, trusting the backend's phase ('closed' | 'preopen' | 'open').
// Recomputing hours from the clock used to disagree with the backend through
// every morning's pre-open auction and showed "CLOSED · HOLIDAY" on ordinary
// trading days.
function nseSession(nseData) {
  const now = new Date();
  const eatMin = ((now.getUTCHours() + 3) % 24) * 60 + now.getUTCMinutes();
  const day = (now.getUTCDay() + (now.getUTCHours() + 3 >= 24 ? 1 : 0)) % 7;
  const OPEN = 9 * 60 + 30;
  const CLOSE = 15 * 60;
  const PRE = 9 * 60;
  const phase = nseData.market_phase;
  if (day === 0 || day === 6) return { label: 'CLOSED', detail: 'Weekend', color: theme.colors.textMuted };
  if (phase === 'open') {
    return { label: 'OPEN', detail: `Closes in ${Math.floor((CLOSE - eatMin) / 60)}h ${(CLOSE - eatMin) % 60}m`, color: theme.colors.primary };
  }
  if (phase === 'preopen') {
    return { label: 'PRE-OPEN', detail: `Trading in ${Math.max(0, OPEN - eatMin)}m`, color: theme.colors.warning };
  }
  const inHoursLocally = eatMin >= PRE && eatMin < CLOSE;
  if (!phase) {
    // Status not loaded yet, or the request failed. Never infer a holiday
    // from missing data.
    return { label: inHoursLocally ? 'UNKNOWN' : 'CLOSED', detail: inHoursLocally ? 'Status unavailable' : '', color: theme.colors.textMuted };
  }
  // Backend says closed on a weekday inside trading hours: a gazetted closure.
  if (inHoursLocally) return { label: 'CLOSED', detail: 'Holiday', color: theme.colors.textMuted };
  if (eatMin < PRE) return { label: 'CLOSED', detail: `Pre-open in ${Math.floor((PRE - eatMin) / 60)}h ${(PRE - eatMin) % 60}m`, color: theme.colors.textMuted };
  return { label: 'CLOSED', detail: 'Opens 09:30 EAT', color: theme.colors.warning };
}

// One line that answers "are these prices real?".
function ProvenanceBanner({ nseData }) {
  const p = nseData.provenance;
  if (!p) return null;
  const notReal = (p.synthetic || 0) + (p.missing || 0) + (p.unverified || 0);
  const total = (p.real || 0) + notReal;
  const s = nseData.scraper || {};
  const lastReal = s.last_real_price_at ? new Date(s.last_real_price_at).toLocaleString() : 'never';
  return (
    <div style={{ marginBottom: '12px' }}>
      <Note tone={notReal > 0 ? theme.colors.danger : null}>
        <strong>{p.real || 0} of {total} prices are live market data.</strong>
        {p.synthetic ? ` ${p.synthetic} synthetic.` : ''}
        {p.missing ? ` ${p.missing} with no data.` : ''}
        {p.unverified ? ` ${p.unverified} of unrecorded origin.` : ''}
        {' '}Last real price received: {lastReal}.
        {notReal > 0 && ' The agent will not evaluate a synthetic price.'}
      </Note>
    </div>
  );
}

function MarketWatch({ nseData, holdings, mobile, onDrill }) {
  const [sort, setSort] = useState({ key: 'symbol', dir: 1 });
  const [filter, setFilter] = useState('all'); // all | held | movers
  const held = Object.fromEntries((holdings || []).map((h) => [h.symbol, h]));
  const quotes = nseData.quotes || [];
  const rows = quotes
    .filter((q) => filter === 'all' || (filter === 'held' ? held[q.symbol] : Math.abs(q.change_pct || 0) >= 1))
    .sort((a, b) => {
      const va = a[sort.key] ?? '';
      const vb = b[sort.key] ?? '';
      return (typeof va === 'number' ? va - vb : String(va).localeCompare(String(vb))) * sort.dir;
    });
  const sortHead = (key, label) => (
    <th key={key} onClick={() => setSort((s) => ({ key, dir: s.key === key ? -s.dir : 1 }))} style={{ ...th, cursor: 'pointer', userSelect: 'none' }}>
      {label}{sort.key === key ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
    </th>
  );
  const session = nseSession(nseData);
  const fx = nseData.fx;
  const gainer = nseData.movers?.gainers?.[0];
  const loser = nseData.movers?.losers?.[0];

  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      <HudRow mobile={mobile}>
        <HUDCard title="NSE session" value={session.label} subValue={session.detail || `${quotes.length} symbols`} tone={session.color} icon={Activity} color={session.color} />
        <HUDCard title="KES per USD" icon={Globe} color={theme.colors.secondary}
                 value={(fx?.kes_per_usd || 1 / (nseData.kes_usd_rate || 1 / 130)).toFixed(2)}
                 subValue={fxSourceNote(fx)}
                 tone={fx && (fx.live || fx.source === 'exchangerate_api') ? theme.colors.textMuted : theme.colors.warning}
                 footer={fx?.attribution_url
                   ? <a href={fx.attribution_url} target="_blank" rel="noopener noreferrer" style={{ color: theme.colors.textMuted }}>Rates by ExchangeRate-API</a>
                   : null} />
        <HUDCard title="Top gainer" value={gainer?.symbol || '—'} subValue={gainer ? pct(gainer.change_pct) : null} icon={TrendingUp} color={theme.colors.primary}
                 onClick={gainer ? () => onDrill(gainer.symbol) : undefined} />
        <HUDCard title="Top loser" value={loser?.symbol || '—'} subValue={loser ? pct(loser.change_pct) : null} icon={TrendingDown} color={theme.colors.danger}
                 onClick={loser ? () => onDrill(loser.symbol) : undefined} />
        <HUDCard title="Paper holdings" value={(holdings || []).length} subValue={`of ${quotes.length} watched`} tone={theme.colors.textMuted} icon={Shield} color={theme.colors.accent} />
      </HudRow>

      <div style={card(mobile)}>
        <SectionHeader title="NSE Market Watch" icon={Flag} />
        <div style={{ marginBottom: '12px', fontSize: '12px', color: theme.colors.textMuted }}>
          The agent checks all {quotes.length} symbols below every cycle. Highlighted rows are held in the paper account. Tap a row for its chart.
        </div>
        <ProvenanceBanner nseData={nseData} />
        <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
          {[['all', 'ALL'], ['held', 'HELD'], ['movers', 'MOVERS ±1%']].map(([id, label]) => (
            <button key={id} onClick={() => setFilter(id)} style={{
              backgroundColor: filter === id ? theme.colors.primary : 'rgba(255,255,255,0.05)',
              color: filter === id ? '#000' : theme.colors.textSecondary,
              border: 'none', padding: '6px 12px', borderRadius: '6px', fontSize: '11px', fontWeight: 800, cursor: 'pointer',
            }}>{label}</button>
          ))}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={tableHeadRow}>
                {sortHead('symbol', 'SYMBOL')}{sortHead('price_kes', 'PRICE (KES)')}{sortHead('change_pct', 'CHANGE')}
                {!mobile && sortHead('volume', 'VOLUME')}
                <th style={th}>{mobile ? 'HELD' : 'PAPER POSITION'}</th>
              </tr>
            </thead>
            <tbody>
              {rows.length > 0 ? rows.map((q) => {
                const h = held[q.symbol];
                const bg = h ? `${theme.colors.primary}15` : 'transparent';
                return (
                  <tr key={q.symbol} onClick={() => onDrill(q.symbol)} title={`Open the ${q.symbol} chart`}
                      style={{ borderBottom: `1px solid ${theme.colors.border}`, background: bg, cursor: 'pointer' }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = `${theme.colors.primary}25`; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = bg; }}>
                    <td style={{ ...td, fontWeight: 700 }}>{q.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span></td>
                    <td style={{ ...td, color: isRealNsePrice(q) ? undefined : theme.colors.textMuted, whiteSpace: 'nowrap' }}>
                      {q.price_kes?.toFixed(2)}
                      {!isRealNsePrice(q) && (
                        <span title={NSE_SOURCE_NOTE[q.source] || 'Origin of this price is not recorded'}
                              style={{ marginLeft: '6px', fontSize: '9px', fontWeight: 800, padding: '1px 5px', borderRadius: '4px',
                                       background: q.source === 'synthetic' ? `${theme.colors.danger}30` : 'rgba(255,255,255,0.08)',
                                       color: q.source === 'synthetic' ? theme.colors.danger : theme.colors.textMuted }}>
                          {q.source === 'synthetic' ? 'SYNTHETIC' : q.source === 'none' ? 'NO DATA' : 'UNVERIFIED'}
                        </span>
                      )}
                    </td>
                    <td style={{ ...td, color: gainColor(q.change_pct) }}>{pct(q.change_pct)}</td>
                    {!mobile && <td style={td}>{q.volume?.toLocaleString()}</td>}
                    <td style={{ ...td, color: h ? gainColor(h.unrealised_pnl_pct) : theme.colors.textMuted, whiteSpace: 'nowrap' }}>
                      {h ? (mobile ? pct(h.unrealised_pnl_pct, 1) : `${h.quantity.toLocaleString()} sh · ${pct(h.unrealised_pnl_pct)}`) : '—'}
                    </td>
                  </tr>
                );
              }) : (
                <tr><td colSpan={mobile ? 4 : 5} style={{ ...td, textAlign: 'center', color: theme.colors.textMuted }}>No NSE data available</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// NSE order tickets: without a broker API, the manual workflow lists the
// agent's proposed NSE trades here; the operator keys them into the
// AIB-AXYS portal and records the fill.
function OrderTickets({ active, mobile }) {
  const [pending, setPending] = useState([]);
  const [fills, setFills] = useState([]);

  const load = useCallback(async () => {
    try {
      const { ok, json } = await apiGet('/api/operator/nse-tickets');
      if (ok && json) {
        setPending(json.pending || []);
        setFills(json.recent_fills || []);
      }
    } catch (e) { /* transient */ }
  }, []);

  useEffect(() => {
    if (!active) return undefined;
    load();
    const id = setInterval(load, 20000);
    return () => clearInterval(id);
  }, [active, load]);

  const act = async (id, verb, body) => {
    try {
      const token = localStorage.getItem('trading_token');
      const res = await fetch(`${getApiBase()}/api/operator/nse-tickets/${id}/${verb}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(body || {}),
      });
      if (res.ok) load();
      else { const j = await res.json().catch(() => ({})); window.alert(j.error || 'Action failed'); }
    } catch (e) { window.alert('Request failed'); }
  };

  const onFill = (t) => {
    const priceStr = window.prompt(`Actual fill price (KES) for ${t.side.toUpperCase()} ${t.quantity} ${t.symbol}:`, t.suggested_limit_price ?? '');
    if (priceStr === null) return;
    const qtyStr = window.prompt(`Actual fill quantity for ${t.symbol}:`, t.quantity);
    if (qtyStr === null) return;
    const fillPrice = parseFloat(priceStr);
    const fillQuantity = parseInt(qtyStr, 10);
    if (!(fillPrice > 0) || !(fillQuantity > 0)) { window.alert('Fill price and quantity must be positive numbers.'); return; }
    act(t.id, 'fill', { fill_price: fillPrice, fill_quantity: fillQuantity });
  };

  const btn = { borderRadius: '6px', padding: '6px 10px', cursor: 'pointer', fontSize: '11px' };
  return (
    <div style={card(mobile)}>
      <SectionHeader title="NSE Order Tickets" icon={Layers} />
      <div style={{ marginBottom: '12px', fontSize: '12px', color: theme.colors.textMuted }}>
        For trading real shares by hand: place the order in your broker portal, then record the fill here.
        While the paper account is on, the agent fills its own trades automatically and nothing waits here.
      </div>
      {pending.length > 0 ? (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={tableHeadRow}>
                <th style={th}>SYMBOL</th><th style={th}>SIDE</th><th style={th}>QTY</th><th style={th}>LIMIT (KES)</th>
                <th style={th}>CONF</th><th style={th}>STATUS</th><th style={th}>ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {pending.map((t) => (
                <tr key={t.id} style={{ borderBottom: `1px solid ${theme.colors.border}` }} title={t.llm_reasoning || t.rationale || ''}>
                  <td style={{ ...td, fontWeight: 700 }}>{t.symbol}</td>
                  <td style={{ ...td, color: t.side === 'buy' ? theme.colors.primary : theme.colors.danger, fontWeight: 700 }}>{t.side.toUpperCase()}</td>
                  <td style={td}>{t.quantity}</td>
                  <td style={td}>{t.suggested_limit_price?.toFixed?.(2) ?? t.suggested_limit_price}</td>
                  <td style={td}>{((t.ensemble_confidence || 0) * 100).toFixed(0)}%</td>
                  <td style={{ ...td, textTransform: 'uppercase', fontSize: '11px', color: t.status === 'placed' ? theme.colors.warning : theme.colors.textMuted }}>{t.status}</td>
                  <td style={{ ...td, display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                    {t.status === 'pending' && <button onClick={() => act(t.id, 'place')} style={{ ...btn, background: theme.colors.bgSecondary, color: '#fff', border: `1px solid ${theme.colors.border}` }}>Mark Placed</button>}
                    <button onClick={() => onFill(t)} style={{ ...btn, background: theme.colors.primary, color: '#000', border: 'none', fontWeight: 700 }}>Mark Filled</button>
                    <button onClick={() => act(t.id, 'cancel')} style={{ ...btn, background: 'transparent', color: theme.colors.danger, border: `1px solid ${theme.colors.danger}55` }}>Cancel</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No pending NSE order tickets.</Empty>
      )}
      {fills.length > 0 && (
        <div style={{ marginTop: '18px' }}>
          <div style={{ fontSize: '11px', fontWeight: 800, color: theme.colors.textMuted, marginBottom: '8px' }}>RECENT NSE FILLS</div>
          {fills.slice(0, 8).map((f) => (
            <div key={f.id} style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', padding: '6px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '12px' }}>
              <span><span style={{ fontWeight: 700 }}>{f.symbol}</span> <span style={{ color: f.side === 'buy' ? theme.colors.primary : theme.colors.danger }}>{f.side.toUpperCase()}</span> {f.fill_quantity} @ {money(f.fill_price, 'KES')}</span>
              <span style={{ color: theme.colors.textMuted, whiteSpace: 'nowrap' }}>{f.fill_at ? new Date(f.fill_at).toLocaleDateString() : ''}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function NseSection({ active, sub, onSub, nseData, isOperator, mobile, onDrill }) {
  const [paper, setPaper] = useState(null);
  const [paperError, setPaperError] = useState(null);

  // The paper account feeds both its own tab and the Market Watch holdings.
  const loadPaper = useCallback(async () => {
    try {
      const { ok, status, json } = await apiGet('/api/nse/paper');
      if (ok && json) { setPaper(json); setPaperError(null); } else setPaperError((json && json.error) || `HTTP ${status}`);
    } catch (e) { setPaperError(e.message); }
  }, []);

  useEffect(() => {
    if (!active) return undefined;
    loadPaper();
    const id = setInterval(loadPaper, 30000);
    return () => clearInterval(id);
  }, [active, loadPaper]);

  const tabs = [
    { id: 'watch', label: 'Market Watch' },
    { id: 'paper', label: 'Paper Account' },
    ...(isOperator ? [{ id: 'tickets', label: 'Order Tickets' }] : []),
  ];
  const current = tabs.some((t) => t.id === sub) ? sub : 'watch';
  const holdings = paper?.account?.holdings || [];

  return (
    <div>
      <SubTabs tabs={tabs} active={current} onChange={onSub} />
      {current === 'watch' && <MarketWatch nseData={nseData} holdings={holdings} mobile={mobile} onDrill={onDrill} />}
      {current === 'paper' && <NsePaperAccount view={paper} error={paperError} mobile={mobile} onDrill={onDrill} />}
      {current === 'tickets' && <OrderTickets active={active} mobile={mobile} />}
    </div>
  );
}
