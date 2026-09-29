// A paper account in US dollars: the US and crypto account (Alpaca's paper
// account) and the forex paper book. Value against the start, cash, holdings
// with the price each would be stopped out at, trades, results on closed
// trades and what the trades cost. Nothing here is real money; the figures
// come from /api/paper/<us|forex>.
import { ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Wallet, PiggyBank, Briefcase, TrendingUp, Receipt, Activity, ShieldAlert, History, BookOpen, Clock } from 'lucide-react';
import { theme } from '../DashboardStyles';
import {
  card, columns, money, signedMoney, pct, gainColor, SectionHeader, HUDCard, HudRow,
  Empty, Note, th, td, tableHeadRow,
} from '../ui';
import { labelFills } from './NsePaperAccount';

const KIND_COLOR = {
  buy: theme.colors.primary, add: theme.colors.secondary, trim: theme.colors.warning,
  sell: theme.colors.danger, stop: theme.colors.danger,
};

const isPair = (symbol) => String(symbol).includes('_');

// Prices to the precision the market quotes: currency pairs to four places.
export const price = (symbol, v) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  const digits = isPair(symbol) || v < 10 ? 4 : 2;
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
};

// Whole units of a currency pair or a share, a few places of a coin.
export const units = (v) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  return v.toLocaleString(undefined, { maximumFractionDigits: v < 1 ? 6 : (v < 100 ? 4 : 2) });
};

const when = (iso) => {
  if (!iso) return '';
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
};

const strategyName = (s) => (s ? String(s).replace(/_/g, ' ') : '');

// How the account did over the same days as each line on the chart.
export function versus(points, start) {
  const last = points[points.length - 1] || {};
  const r = (v) => (v && start ? (v / start - 1) * 100 : null);
  return { account: r(last.equity), benchmark: r(last.benchmark) };
}

function EquityCurve({ points, start, benchmarkName, mobile }) {
  if (points.length < 2) {
    return (
      <Empty>
        The curve starts at {money(start, 'USD', 0)}. One reading is kept for each day, so it takes shape
        over the coming days{benchmarkName ? `, drawn against the ${benchmarkName}` : ''}.
      </Empty>
    );
  }
  const values = points.flatMap((p) => [p.equity, p.benchmark]).filter((v) => typeof v === 'number');
  const lo = Math.min(start, ...values);
  const hi = Math.max(start, ...values);
  const pad = Math.max((hi - lo) * 0.15, start * 0.002);
  const vs = versus(points, start);
  const names = { equity: 'Account value', benchmark: benchmarkName || 'Benchmark' };
  return (
    <>
      <ResponsiveContainer width="100%" height={mobile ? 220 : 280}>
        <ComposedChart data={points} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
          <defs>
            <linearGradient id="paperBookEq" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={theme.colors.primary} stopOpacity={0.3} />
              <stop offset="95%" stopColor={theme.colors.primary} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border} vertical={false} />
          <XAxis dataKey="day" stroke={theme.colors.textMuted} fontSize={10} tickFormatter={(d) => d.slice(5)} minTickGap={24} />
          <YAxis stroke={theme.colors.textMuted} fontSize={10} width={mobile ? 52 : 66}
                 domain={[Math.floor(lo - pad), Math.ceil(hi + pad)]}
                 tickFormatter={(v) => (hi - lo < 5000 ? Math.round(v).toLocaleString() : `${(v / 1000).toFixed(1)}k`)} />
          <Tooltip contentStyle={{ backgroundColor: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, color: '#fff', fontSize: '12px' }}
                   formatter={(v, name) => [money(v, 'USD'), names[name] || name]} />
          <Legend formatter={(name) => names[name] || name} wrapperStyle={{ fontSize: '11px' }} />
          <ReferenceLine y={start} stroke={theme.colors.textMuted} strokeDasharray="4 4" />
          <Area type="monotone" dataKey="equity" stroke={theme.colors.primary} fill="url(#paperBookEq)" strokeWidth={2} />
          {benchmarkName && <Line type="monotone" dataKey="benchmark" stroke={theme.colors.secondary} dot={false} strokeWidth={1.5} connectNulls />}
        </ComposedChart>
      </ResponsiveContainer>
      {vs.account != null && (
        <div style={{ fontSize: '12px', color: theme.colors.textSecondary, marginTop: '8px' }}>
          Since the start: account <strong style={{ color: gainColor(vs.account) }}>{pct(vs.account)}</strong>
          {vs.benchmark != null && benchmarkName && <>, {benchmarkName} <strong style={{ color: gainColor(vs.benchmark) }}>{pct(vs.benchmark)}</strong></>}.
          {vs.benchmark != null && benchmarkName && (vs.account >= vs.benchmark
            ? ` The account is ahead of the ${benchmarkName}.`
            : ` The account is behind the ${benchmarkName}.`)}
        </div>
      )}
    </>
  );
}

const stopText = (h) => {
  if (h.core) return { stop: 'core', trail: 'core', title: 'Core index funds are held through falls on purpose, so they have no stops' };
  const s = h.stops;
  if (!s) return { stop: '—', trail: '—', title: '' };
  const how = s.source === 'atr' ? 'scaled to this holding\'s own daily range' : 'the fixed setting, until a daily range is known';
  return {
    stop: s.stop_loss_price != null ? price(h.symbol, s.stop_loss_price) : '—',
    trail: s.trailing_stop_price != null ? price(h.symbol, s.trailing_stop_price) : '—',
    title: `Sells everything ${(s.stop_loss_pct * 100).toFixed(1)}% below the average cost, or ${(s.trailing_stop_pct * 100).toFixed(1)}% below the high of ${s.high_since_entry != null ? price(h.symbol, s.high_since_entry) : 'n/a'}; ${how}`,
  };
};

function HoldingCard({ h, onDrill }) {
  const st = stopText(h);
  return (
    <div onClick={() => onDrill(h.symbol)} style={{ padding: '12px 0', borderBottom: `1px solid ${theme.colors.border}`, cursor: 'pointer' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontWeight: 800, fontSize: '15px' }}>{h.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>{h.region} ↗</span></span>
        <span style={{ fontWeight: 800, color: gainColor(h.unrealised_pnl) }}>{pct(h.unrealised_pnl_pct)}</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: theme.colors.textSecondary, marginTop: '4px' }}>
        <span>{units(h.quantity)} @ {price(h.symbol, h.avg_cost)}</span>
        <span>{money(h.market_value)} <span style={{ color: gainColor(h.unrealised_pnl) }}>({signedMoney(h.unrealised_pnl)})</span></span>
      </div>
      <div title={st.title} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: theme.colors.textMuted, marginTop: '4px' }}>
        <span>Last {price(h.symbol, h.last_price)} · {h.weight_pct.toFixed(1)}% of the account</span>
        <span>Stop {st.stop} · Trail {st.trail}</span>
      </div>
    </div>
  );
}

function HoldingsTable({ holdings, onDrill }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
        <thead>
          <tr style={tableHeadRow}>
            <th style={th}>SYMBOL</th><th style={th}>QUANTITY</th><th style={th}>AVG COST</th><th style={th}>LAST</th>
            <th style={th}>VALUE</th><th style={th} title="Share of the account's value">WEIGHT</th><th style={th}>PROFIT / LOSS</th>
            <th style={th} title="Sells everything if the price falls to this level">STOP-LOSS</th>
            <th style={th} title="Sells everything if the price falls to this level, measured from the highest price since buying">TRAILING STOP</th>
          </tr>
        </thead>
        <tbody>
          {holdings.map((h) => {
            const st = stopText(h);
            return (
              <tr key={h.symbol} onClick={() => onDrill(h.symbol)} style={{ borderBottom: `1px solid ${theme.colors.border}`, cursor: 'pointer' }}>
                <td style={{ ...td, fontWeight: 800 }}>{h.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>{h.region} ↗</span></td>
                <td style={td}>{units(h.quantity)}</td>
                <td style={td}>{price(h.symbol, h.avg_cost)}</td>
                <td style={td}>{price(h.symbol, h.last_price)}</td>
                <td style={td}>{money(h.market_value)}</td>
                <td style={td}>{h.weight_pct.toFixed(1)}%</td>
                <td style={{ ...td, color: gainColor(h.unrealised_pnl), fontWeight: 700, whiteSpace: 'nowrap' }}>
                  {signedMoney(h.unrealised_pnl)} ({pct(h.unrealised_pnl_pct)})
                </td>
                <td style={{ ...td, color: h.core ? theme.colors.textMuted : undefined }} title={st.title}>{st.stop}</td>
                <td style={{ ...td, color: h.core ? theme.colors.textMuted : undefined }} title={st.title}>{st.trail}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TradeHistory({ fills, mobile, onDrill, empty }) {
  if (!fills.length) return <Empty>{empty}</Empty>;
  return (
    <div style={{ maxHeight: mobile ? 'none' : '380px', overflowY: 'auto' }}>
      {fills.map((f) => (
        <div key={f.id} onClick={() => onDrill(f.symbol)}
             style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', padding: '9px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '12px', cursor: 'pointer' }}>
          <div style={{ minWidth: 0 }}>
            <span style={{ fontWeight: 800 }}>{f.symbol}</span>{' '}
            <span style={{ color: KIND_COLOR[f.kind], fontWeight: 800, textTransform: 'uppercase' }}>{f.kind}</span>{' '}
            <span style={{ color: theme.colors.textSecondary }}>{units(f.quantity)} @ {price(f.symbol, f.price)}</span>
            <div style={{ fontSize: '10px', color: theme.colors.textMuted, marginTop: '2px' }}>
              {money(f.quantity * f.price)}{f.strategy ? ` · ${strategyName(f.strategy)}` : ''}
            </div>
          </div>
          <span style={{ color: theme.colors.textMuted, fontSize: '11px', whiteSpace: 'nowrap' }}>{when(f.at)}</span>
        </div>
      ))}
    </div>
  );
}

const p0 = (v) => `${Math.round((v || 0) * 100)}%`;

// The rules, in plain language, from the settings the agent is actually using.
function Rules({ view }) {
  const r = view.rules || {};
  const costs = view.costs || {};
  const f = (v) => `${((v || 0) * 100).toFixed(2)}%`;
  const costLine = (market, label) => {
    const c = costs[market];
    if (!c) return null;
    return (
      <li key={market} style={{ marginBottom: '4px' }}>
        <strong style={{ color: '#fff' }}>{label}:</strong> {f(c.commission_pct)} fee and {f(c.slippage_pct)} slippage a side
        {c.verified ? '' : ', an estimate not yet confirmed with the broker'}.
      </li>
    );
  };
  return (
    <div style={{ fontSize: '12px', color: theme.colors.textSecondary, lineHeight: 1.6 }}>
      <p style={{ margin: '0 0 8px' }}>
        <strong style={{ color: '#fff' }}>Stops.</strong>{' '}
        A holding is sold in full if it falls {r.stop_loss_atr_mult}x its own typical daily range below its average cost, or {r.trailing_stop_atr_mult}x that range below its highest price since buying. Both stay between {p0(r.min_stop_pct)} and {p0(r.max_stop_pct)}. Until a daily range is known, the fixed {p0(r.stop_loss_pct)} and {p0(r.trailing_stop_pct)} apply.
      </p>
      <p style={{ margin: '0 0 8px' }}>
        <strong style={{ color: '#fff' }}>Size and pace.</strong>{' '}
        No holding is opened above {p0(r.max_position_size)} of the account, and at most {r.max_new_buys_per_day} new holdings are opened in a market in 24 hours. Positions are long only, with no borrowing.
      </p>
      {r.core && (
        <p style={{ margin: '0 0 8px' }}>
          <strong style={{ color: '#fff' }}>Core.</strong>{' '}
          {p0(1 - r.core.active_share)} of the account is held in {(r.core.symbols || []).join(', ')}, weighted so each carries about the same risk, with no stops. The other {p0(r.core.active_share)} is traded by the strategies.
        </p>
      )}
      <p style={{ margin: '0 0 4px' }}>
        <strong style={{ color: '#fff' }}>Costs.</strong>{' '}
        {view.kind === 'forex'
          ? 'The book fills at the live price plus the spread, so every fill and result already includes it:'
          : 'The paper account does not charge them, so the agent counts them itself in every result:'}
      </p>
      <ul style={{ margin: 0, paddingLeft: '18px' }}>
        {costLine('us_equity', 'US stocks and funds')}
        {costLine('crypto', 'Crypto')}
        {costLine('forex', 'Currency pairs')}
      </ul>
    </div>
  );
}

function WorkingOrders({ orders }) {
  if (!orders.length) return null;
  return (
    <Note tone={theme.colors.warning}>
      <strong>{orders.length} order{orders.length === 1 ? ' is' : 's are'} still waiting for a result:</strong>{' '}
      {orders.map((o) => `${o.side} ${units(o.quantity)} ${o.symbol} (${o.status}, since ${when(o.since)})`).join('; ')}.
      An order that stays here for long usually never reached the broker.
    </Note>
  );
}

// `view` is the /api/paper/<kind> reply; null while loading.
export default function PaperBook({ kind, view, error, mobile, onDrill }) {
  const name = kind === 'forex' ? 'Forex paper book' : 'US and crypto paper account';
  if (!view) {
    return <div style={card(mobile)}>{error ? <Empty>The {name.toLowerCase()} could not load ({error}).</Empty> : <Empty>Loading the {name.toLowerCase()}…</Empty>}</div>;
  }
  if (!view.enabled) {
    return (
      <div style={card(mobile)}>
        <SectionHeader title={name} icon={Wallet} />
        <Note tone={theme.colors.warning}>{view.reason || 'The account is off.'}{kind === 'forex' && ' Switch it on with brokers.forex_paper in the agent\'s configuration.'}</Note>
      </div>
    );
  }

  const a = view.account || {};
  const stats = view.stats || {};
  const holdings = view.holdings || [];
  const fills = labelFills(view.fills || []);
  const start = a.starting_capital || 0;
  const live = view.broker && view.broker.paper === false;
  const session = view.session;

  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      {live ? (
        <Note tone={theme.colors.danger}>
          <strong>This is a live account.</strong> The broker reports it is not a paper account, so orders move real money.
        </Note>
      ) : (
        <Note>
          Paper trading: no real money moves. {kind === 'forex'
            ? 'The agent trades this dollar book on live currency prices, with an estimated spread as the cost, so you can judge it before any live money is involved.'
            : `The agent trades this account (${view.broker?.name || 'broker'} paper) on real prices, so you can judge it before any live money is involved.`}
        </Note>
      )}

      <HudRow mobile={mobile}>
        <HUDCard title="Account value" value={money(a.equity)}
                 subValue={a.pnl == null ? null : `${signedMoney(a.pnl)} (${pct(a.return_pct)})`}
                 tone={gainColor(a.pnl)} icon={Wallet} color={theme.colors.primary}
                 footer={`Started with ${money(start, 'USD', 0)}${a.started_on ? `, first reading ${a.started_on}` : ''}`} />
        <HUDCard title="Cash" value={money(a.cash)} subValue={`${(a.cash_pct || 0).toFixed(0)}% of the account`}
                 tone={theme.colors.textSecondary} icon={PiggyBank} color={theme.colors.secondary} />
        <HUDCard title="Invested" value={money(a.invested)} subValue={`${holdings.length} holding${holdings.length === 1 ? '' : 's'}`}
                 tone={theme.colors.textSecondary} icon={Briefcase} color={theme.colors.accent} />
        <HUDCard title="Realised profit" value={signedMoney(stats.realised_pnl)}
                 subValue={stats.closed_trades
                   ? `${stats.closed_trades} closed trade${stats.closed_trades === 1 ? '' : 's'}, ${Math.round((stats.win_rate || 0) * 100)}% won`
                   : 'no closed trades yet'}
                 tone={theme.colors.textMuted} icon={TrendingUp} color={gainColor(stats.realised_pnl)} />
        {kind === 'forex'
          ? <HUDCard title="Costs" value="In the prices" subValue="every fill already includes the spread"
                     tone={theme.colors.textMuted} icon={Receipt} color={theme.colors.warning} />
          : <HUDCard title="Costs, modelled" value={money(stats.modelled_costs)} subValue="counted in results, not charged by the account"
                     tone={theme.colors.textMuted} icon={Receipt} color={theme.colors.warning} />}
      </HudRow>

      {session && (
        <Note tone={session.open ? null : theme.colors.warning}>
          <Clock size={12} style={{ verticalAlign: '-2px', marginRight: '6px' }} />
          {kind === 'forex' ? 'Currency market' : 'US stock market'} is <strong>{session.open ? 'open' : 'closed'}</strong> now. {session.hours}.
          {!session.open && ' New orders in this market wait for it to open.'}
        </Note>
      )}

      <WorkingOrders orders={view.working_orders || []} />

      <div style={columns('2fr 1fr', mobile)}>
        <div style={card(mobile)}>
          <SectionHeader title="Account value over time" icon={Activity} />
          <EquityCurve points={view.equity_curve || []} start={start} benchmarkName={view.benchmark_name} mobile={mobile} />
        </div>
        <div style={card(mobile)}>
          <SectionHeader title="The rules it trades by" icon={BookOpen} />
          <Rules view={view} />
        </div>
      </div>

      <div style={card(mobile)}>
        <SectionHeader title="Holdings" icon={ShieldAlert} />
        {holdings.length === 0
          ? <Empty>No holdings. All {money(a.cash)} is in cash.</Empty>
          : mobile
            ? holdings.map((h) => <HoldingCard key={h.symbol} h={h} onDrill={onDrill} />)
            : <HoldingsTable holdings={holdings} onDrill={onDrill} />}
      </div>

      <div style={card(mobile)}>
        <SectionHeader title="Trade history" icon={History}
                       right={<span style={{ fontSize: '11px', color: theme.colors.textMuted }}>{stats.fills || 0} trade{stats.fills === 1 ? '' : 's'} in all</span>} />
        <TradeHistory fills={fills} mobile={mobile} onDrill={onDrill}
                      empty={kind === 'forex'
                        ? 'No forex trades yet. The agent trades a pair when a signal passes its checks while the currency market is open.'
                        : 'No trades yet. The agent trades when a signal passes its checks while the market is open.'} />
      </div>
    </div>
  );
}
