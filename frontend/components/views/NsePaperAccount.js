// The agent's NSE paper account: KES cash, holdings with their stops, the
// trade history and a daily equity curve. Nothing here is real money; the
// account replays the agent's paper fills from /api/nse/paper.
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Wallet, PiggyBank, Briefcase, TrendingUp, Receipt, Activity, ShieldAlert, History, BookOpen } from 'lucide-react';
import { theme } from '../DashboardStyles';
import {
  card, columns, money, signedMoney, pct, gainColor, SectionHeader, HUDCard, HudRow,
  Empty, Note, th, td, tableHeadRow,
} from '../ui';

const KIND_COLOR = {
  buy: theme.colors.primary, add: theme.colors.secondary, trim: theme.colors.warning,
  sell: theme.colors.danger, stop: theme.colors.danger,
};
const STOP_STRATEGIES = ['stop_loss', 'trailing_stop'];

// What each fill did to its position (mirrors chart_data.markers on the
// server): a buy while holding is an add, a sell that leaves shares a trim.
export function labelFills(newestFirst) {
  const held = {};
  const out = [...newestFirst].reverse().map((f) => {
    const before = held[f.symbol] || 0;
    let kind;
    if (f.side === 'buy') {
      kind = before > 0 ? 'add' : 'buy';
      held[f.symbol] = before + f.quantity;
    } else {
      kind = STOP_STRATEGIES.includes(f.strategy) ? 'stop' : (before - f.quantity > 0 ? 'trim' : 'sell');
      held[f.symbol] = Math.max(before - f.quantity, 0);
    }
    return { ...f, kind };
  });
  return out.reverse();
}

const when = (iso) => {
  if (!iso) return '';
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
};

const strategyName = (s) => (s ? String(s).replace(/_/g, ' ') : '');

function EquityCurve({ points, start, mobile }) {
  if (points.length < 2) {
    return (
      <Empty>
        The curve starts at {money(start, 'KES', 0)}. One reading is added each trading day, so it
        takes shape over the coming sessions.
      </Empty>
    );
  }
  const lo = Math.min(start, ...points.map((p) => p.equity_kes));
  const hi = Math.max(start, ...points.map((p) => p.equity_kes));
  const pad = Math.max((hi - lo) * 0.15, start * 0.005);
  return (
    <ResponsiveContainer width="100%" height={mobile ? 200 : 260}>
      <AreaChart data={points} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
        <defs>
          <linearGradient id="paperEq" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={theme.colors.primary} stopOpacity={0.3} />
            <stop offset="95%" stopColor={theme.colors.primary} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border} vertical={false} />
        <XAxis dataKey="day" stroke={theme.colors.textMuted} fontSize={10} tickFormatter={(d) => d.slice(5)} minTickGap={24} />
        <YAxis stroke={theme.colors.textMuted} fontSize={10} width={mobile ? 44 : 60}
               domain={[Math.floor(lo - pad), Math.ceil(hi + pad)]}
               tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v)} />
        <Tooltip contentStyle={{ backgroundColor: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, color: '#fff', fontSize: '12px' }}
                 formatter={(v, name) => [money(v, 'KES'), name === 'equity_kes' ? 'Account value' : name]} />
        <ReferenceLine y={start} stroke={theme.colors.textMuted} strokeDasharray="4 4"
                       label={{ value: 'Start', fill: theme.colors.textMuted, fontSize: 10, position: 'insideTopLeft' }} />
        <Area type="monotone" dataKey="equity_kes" stroke={theme.colors.primary} fill="url(#paperEq)" strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function HoldingCard({ h, onDrill }) {
  const s = h.stops || {};
  return (
    <div onClick={() => onDrill(h.symbol)} style={{ padding: '12px 0', borderBottom: `1px solid ${theme.colors.border}`, cursor: 'pointer' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontWeight: 800, fontSize: '15px' }}>{h.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span></span>
        <span style={{ fontWeight: 800, color: gainColor(h.unrealised_pnl_kes) }}>{pct(h.unrealised_pnl_pct)}</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: theme.colors.textSecondary, marginTop: '4px' }}>
        <span>{h.quantity.toLocaleString()} sh @ {h.avg_cost_kes?.toFixed(2)}</span>
        <span>{money(h.market_value_kes, 'KES', 0)} <span style={{ color: gainColor(h.unrealised_pnl_kes) }}>({signedMoney(h.unrealised_pnl_kes, 'KES', 0)})</span></span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: theme.colors.textMuted, marginTop: '4px' }}>
        <span>Last {h.last_price_kes ? h.last_price_kes.toFixed(2) : 'n/a, valued at cost'}</span>
        <span>Stop {s.stop_loss_kes?.toFixed(2) ?? '—'} · Trail {s.trailing_stop_kes?.toFixed(2) ?? '—'}</span>
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
            <th style={th}>SYMBOL</th><th style={th}>SHARES</th><th style={th}>AVG COST</th><th style={th}>LAST</th>
            <th style={th}>VALUE</th><th style={th}>PROFIT / LOSS</th><th style={th} title="Sells everything if the price falls this far below the average cost">STOP-LOSS</th>
            <th style={th} title="Sells everything if the price falls this far below the highest price since buying">TRAILING STOP</th><th style={th}>ADDS</th>
          </tr>
        </thead>
        <tbody>
          {holdings.map((h) => {
            const s = h.stops || {};
            return (
              <tr key={h.symbol} onClick={() => onDrill(h.symbol)} style={{ borderBottom: `1px solid ${theme.colors.border}`, cursor: 'pointer' }}>
                <td style={{ ...td, fontWeight: 800 }}>{h.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span></td>
                <td style={td}>{h.quantity.toLocaleString()}</td>
                <td style={td}>{h.avg_cost_kes?.toFixed(2)}</td>
                <td style={{ ...td, color: h.priced ? undefined : theme.colors.textMuted }} title={h.priced ? '' : 'No real price this cycle; valued at cost'}>
                  {h.last_price_kes ? h.last_price_kes.toFixed(2) : 'at cost'}
                </td>
                <td style={td}>{money(h.market_value_kes, 'KES', 0)}</td>
                <td style={{ ...td, color: gainColor(h.unrealised_pnl_kes), fontWeight: 700, whiteSpace: 'nowrap' }}>
                  {signedMoney(h.unrealised_pnl_kes, 'KES', 0)} ({pct(h.unrealised_pnl_pct)})
                </td>
                <td style={td} title={s.stop_loss_pct ? `${(s.stop_loss_pct * 100).toFixed(1)}% below cost` : ''}>{s.stop_loss_kes?.toFixed(2) ?? '—'}</td>
                <td style={td} title={s.trailing_stop_pct ? `${(s.trailing_stop_pct * 100).toFixed(1)}% below the high of ${s.high_since_entry_kes}` : ''}>{s.trailing_stop_kes?.toFixed(2) ?? '—'}</td>
                <td style={td}>{Math.max((h.entries || 1) - 1, 0)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TradeHistory({ fills, mobile, onDrill }) {
  if (!fills.length) return <Empty>No paper trades yet. The agent trades when a signal passes its checks during NSE hours.</Empty>;
  return (
    <div style={{ maxHeight: mobile ? 'none' : '380px', overflowY: 'auto' }}>
      {fills.map((f) => (
        <div key={f.id} onClick={() => onDrill(f.symbol)} title={f.rationale || ''}
             style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', padding: '9px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '12px', cursor: 'pointer' }}>
          <div style={{ minWidth: 0 }}>
            <span style={{ fontWeight: 800 }}>{f.symbol}</span>{' '}
            <span style={{ color: KIND_COLOR[f.kind], fontWeight: 800, textTransform: 'uppercase' }}>{f.kind}</span>{' '}
            <span style={{ color: theme.colors.textSecondary }}>{f.quantity.toLocaleString()} @ {f.price.toFixed(2)}</span>
            <div style={{ fontSize: '10px', color: theme.colors.textMuted, marginTop: '2px' }}>
              {money(f.quantity * f.price, 'KES', 0)} · fees {money(f.fees_kes, 'KES', 0)}{f.strategy ? ` · ${strategyName(f.strategy)}` : ''}
            </div>
          </div>
          <span style={{ color: theme.colors.textMuted, fontSize: '11px', whiteSpace: 'nowrap' }}>{when(f.fill_at)}</span>
        </div>
      ))}
    </div>
  );
}

// "1.30% brokerage, 0.34% levies and 0.02% stamp duty (AIB-AXYS schedule)…"
function costText(costs, verified) {
  const c = costs || {};
  const f = (v) => `${((v || 0) * 100).toFixed(2)}%`;
  const b = c.breakdown;
  const fees = b
    ? `${f(b.brokerage_pct)} brokerage, ${f(b.statutory_levies_pct)} statutory levies and ${f(b.stamp_duty_pct)} stamp duty`
    : `${f(c.commission_pct)} in fees`;
  const annual = c.annual_fee_kes ? `, plus KES ${c.annual_fee_kes} a year for the account` : '';
  const slip = c.slippage_pct ? ` An estimated ${f(c.slippage_pct)} slippage is added to each price.` : '';
  return verified
    ? `Each trade pays ${fees} per side, from the AIB-AXYS schedule${annual}.${slip}`
    : `Each trade is charged an estimated ${fees} per side until the broker's schedule is confirmed.${slip}`;
}

function Rules({ rules, costsVerified }) {
  const stop = rules.stop_loss || {};
  const add = rules.add_to_winners || {};
  const exit = rules.exits || {};
  const limits = rules.limits || {};
  const p = (v) => `${Math.round((v || 0) * 100)}%`;
  return (
    <div style={{ fontSize: '12px', color: theme.colors.textSecondary, lineHeight: 1.6 }}>
      <p style={{ margin: '0 0 8px' }}>
        <strong style={{ color: '#fff' }}>Stops.</strong>{' '}
        {stop.enabled === false ? 'Off.' : `A holding is sold in full if it falls ${p(stop.stop_loss_pct)} below its average cost, or ${p(stop.trailing_stop_pct)} below its highest price since buying. Volatile stocks get wider stops, never tighter.`}
      </p>
      <p style={{ margin: '0 0 8px' }}>
        <strong style={{ color: '#fff' }}>Adding to winners.</strong>{' '}
        {add.enabled === false ? 'Off.' : `Only when the holding is in profit after selling costs and the price is at least ${p(add.min_gain_pct)} above the last purchase. Each add is ${p(add.add_size_pct)} of a normal order, at most ${add.max_adds} adds, and no stock above ${p(add.max_position_pct)} of the account.`}
      </p>
      {limits.min_holding_days != null && (
        <p style={{ margin: '0 0 8px' }}>
          <strong style={{ color: '#fff' }}>Trading limits.</strong>{' '}
          {`Each holding is kept at least ${limits.min_holding_days} days unless a stop triggers; at most ${limits.max_new_positions_per_week} new holdings a week; no order above ${p(limits.max_adv_fraction)} of the stock's average daily volume; sale proceeds are spendable ${limits.settlement_days} trading days after the sale.`}
        </p>
      )}
      <p style={{ margin: '0 0 8px' }}>
        <strong style={{ color: '#fff' }}>Selling.</strong>{' '}
        {exit.partial_exits === false ? 'A sell signal closes the whole holding.' : `A strong sell signal (${p(exit.full_exit_confidence)} confidence or more) closes the holding; a weaker one sells ${p(exit.trim_fraction)} of it, at most once a day.`}
      </p>
      <p style={{ margin: 0 }}>
        <strong style={{ color: '#fff' }}>Costs.</strong>{' '}
        {costText(rules.costs, costsVerified)}
      </p>
    </div>
  );
}

// `view` is the /api/nse/paper reply, loaded by the NSE section (which also
// shows the holdings in Market Watch); null while loading.
export default function NsePaperAccount({ view, error, mobile, onDrill }) {
  if (!view) {
    return <div style={card(mobile)}>{error ? <Empty>The paper account could not load ({error}).</Empty> : <Empty>Loading the paper account…</Empty>}</div>;
  }
  if (!view.enabled) {
    return (
      <div style={card(mobile)}>
        <SectionHeader title="NSE Paper Account" icon={Wallet} />
        <Note>The NSE paper account is switched off. Set <code>nse_paper_trading.starting_capital_kes</code> in the agent's configuration to open one.</Note>
      </div>
    );
  }

  const a = view.account || {};
  const holdings = a.holdings || [];
  const fills = labelFills(view.fills || []);
  const start = a.starting_capital_kes || 0;
  const pnl = (a.equity_kes || 0) - start;
  const unpriced = holdings.filter((h) => !h.priced).length;

  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      <HudRow mobile={mobile}>
        <HUDCard title="Account value" value={money(a.equity_kes, 'KES', 0)} subValue={`${signedMoney(pnl, 'KES', 0)} (${pct(a.return_pct)})`}
                 tone={gainColor(pnl)} icon={Wallet} color={theme.colors.primary}
                 footer={`Started with ${money(start, 'KES', 0)}${a.started_at ? ` on ${new Date(a.started_at).toLocaleDateString()}` : ''}`} />
        <HUDCard title="Cash" value={money(a.cash_kes, 'KES', 0)}
                 subValue={a.unsettled_kes > 0
                   ? `${money(a.unsettled_kes, 'KES', 0)} awaiting settlement`
                   : (start ? `${((a.cash_kes / (a.equity_kes || start)) * 100).toFixed(0)}% of the account` : null)}
                 tone={a.unsettled_kes > 0 ? theme.colors.warning : theme.colors.textSecondary} icon={PiggyBank} color={theme.colors.secondary} />
        <HUDCard title="Invested" value={money(a.holdings_value_kes, 'KES', 0)} subValue={`${holdings.length} holding${holdings.length === 1 ? '' : 's'}`}
                 tone={theme.colors.textSecondary} icon={Briefcase} color={theme.colors.accent} />
        <HUDCard title="Realised profit" value={signedMoney(a.realised_pnl_kes, 'KES', 0)}
                 subValue={a.dividends_net_kes > 0
                   ? `plus ${money(a.dividends_net_kes, 'KES', 0)} dividends after ${money(a.dividend_tax_kes, 'KES', 0)} tax`
                   : 'from closed and trimmed trades'}
                 tone={theme.colors.textMuted} icon={TrendingUp} color={gainColor(a.realised_pnl_kes)} />
        <HUDCard title="Fees paid" value={money(a.fees_paid_kes, 'KES', 0)} subValue={a.costs_verified ? 'broker schedule' : 'estimated costs'}
                 tone={theme.colors.textMuted} icon={Receipt} color={theme.colors.warning} />
      </HudRow>

      <Note>
        Paper trading: no real money moves. The agent trades this KES account on real NSE prices, with estimated
        brokerage costs, so you can judge it before any live money is involved.
        {unpriced > 0 && ` ${unpriced} holding${unpriced === 1 ? ' has' : 's have'} no real price this cycle and ${unpriced === 1 ? 'is' : 'are'} valued at cost.`}
      </Note>

      <div style={columns('2fr 1fr', mobile)}>
        <div style={card(mobile)}>
          <SectionHeader title="Account value over time" icon={Activity} />
          <EquityCurve points={view.equity_curve || []} start={start} mobile={mobile} />
        </div>
        <div style={card(mobile)}>
          <SectionHeader title="The rules it trades by" icon={BookOpen} />
          <Rules rules={view.rules || {}} costsVerified={a.costs_verified} />
        </div>
      </div>

      <div style={card(mobile)}>
        <SectionHeader title="Holdings" icon={ShieldAlert} />
        {holdings.length === 0
          ? <Empty>No holdings. All {money(a.cash_kes, 'KES', 0)} is in cash.</Empty>
          : mobile
            ? holdings.map((h) => <HoldingCard key={h.symbol} h={h} onDrill={onDrill} />)
            : <HoldingsTable holdings={holdings} onDrill={onDrill} />}
      </div>

      <div style={card(mobile)}>
        <SectionHeader title="Trade history" icon={History} right={<span style={{ fontSize: '11px', color: theme.colors.textMuted }}>{fills.length} trade{fills.length === 1 ? '' : 's'}</span>} />
        <TradeHistory fills={fills} mobile={mobile} onDrill={onDrill} />
      </div>
    </div>
  );
}
