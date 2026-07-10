import { useState, useEffect } from 'react';
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import { X, TrendingUp, TrendingDown, Activity, Layers, AlertTriangle } from 'lucide-react';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

// Human labels for the skip reasons the decision journal records.
const REASON_LABEL = {
  hold: 'No directional signal',
  below_confidence: 'Below confidence floor',
  bias_downgrade: 'Bias detector downgrade',
  llm_veto: 'LLM vetoed the trade',
  fallback_price: 'Price was synthetic fallback',
  halted: 'Trading halted (kill switch)',
  risk_limits: 'Portfolio risk limit hit',
  pdt_guard: 'Pattern-day-trader block',
  min_notional: 'Below minimum order size',
  no_account_info: 'Account info unavailable',
  no_broker: 'No connected broker',
  no_price: 'No valid price',
  duplicate: 'Duplicate decision blocked',
};

const reasonColor = (r) => {
  if (!r) return theme.colors.primary;                 // executed
  if (['halted', 'risk_limits', 'pdt_guard'].includes(r)) return theme.colors.danger;
  if (['llm_veto', 'bias_downgrade'].includes(r)) return theme.colors.accent;
  if (r === 'hold') return theme.colors.textMuted;
  return theme.colors.warning;
};

const Stat = ({ label, value, color }) => (
  <div style={{ flex: 1 }}>
    <div style={{ fontSize: '11px', color: theme.colors.textMuted, textTransform: 'uppercase', fontWeight: 800 }}>{label}</div>
    <div style={{ fontSize: '22px', fontWeight: 800, color: color || theme.colors.text }}>{value}</div>
  </div>
);

const SymbolDrilldown = ({ symbol, onClose }) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/symbol/${symbol}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) { setError(res.status === 404 ? 'not_found' : `HTTP ${res.status}`); return; }
        const json = await res.json();
        if (alive) setData(json);
      } catch (e) {
        if (alive) setError(e.message);
      }
    };
    load();
    const id = setInterval(load, 20000);
    return () => { alive = false; clearInterval(id); };
  }, [symbol]);

  const alpha = data?.alpha_vs_hold || {};
  const eq = data?.execution_quality || {};
  const books = data?.strategy_books || {};
  const decisions = data?.decisions || [];

  // Build the tape series from decisions that carry a price, oldest first.
  const tape = [...decisions]
    .filter((d) => d.price)
    .reverse()
    .map((d, i) => ({
      i,
      ts: d.ts ? new Date(d.ts).toLocaleTimeString() : '',
      price: d.price,
      executed: d.executed ? d.price : null,
      reason: d.skip_reason,
      action: d.action,
    }));

  const agentBeatsHold = alpha.buy_hold_return_pct != null &&
    alpha.agent_pnl != null; // both present -> comparison meaningful

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, backgroundColor: 'rgba(2,6,23,0.88)',
      display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
      zIndex: 1000, padding: '24px', overflowY: 'auto',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        ...glassCard, maxWidth: '900px', width: '100%', padding: '28px',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2 style={{ margin: 0, fontSize: '24px', fontWeight: 800, letterSpacing: '1px' }}>{symbol}</h2>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: theme.colors.textSecondary, cursor: 'pointer' }}><X size={22} /></button>
        </div>

        {error && (
          <div style={{ color: theme.colors.danger, padding: '20px' }}>
            {error === 'not_found'
              ? `"${symbol}" isn't a tracked symbol. Check the ticker — e.g. SCOM, EQTY (NSE) or AAPL, NVDA (US).`
              : `Failed to load: ${error}`}
          </div>
        )}
        {!data && !error && <div style={{ color: theme.colors.textMuted, padding: '40px', textAlign: 'center' }}>Loading…</div>}

        {data && (
          <>
            {/* Headline stats: the BI numbers */}
            <div style={{ display: 'flex', gap: '20px', marginBottom: '24px', flexWrap: 'wrap' }}>
              <Stat label="Current Price" value={alpha.current_price ? `$${alpha.current_price.toFixed(2)}` : '—'} />
              <Stat label="Agent P&L" value={`${alpha.agent_pnl >= 0 ? '+' : ''}$${(alpha.agent_pnl || 0).toFixed(2)}`}
                    color={alpha.agent_pnl >= 0 ? theme.colors.primary : theme.colors.danger} />
              <Stat label="Buy & Hold" value={alpha.buy_hold_return_pct != null ? `${alpha.buy_hold_return_pct >= 0 ? '+' : ''}${alpha.buy_hold_return_pct}%` : '—'}
                    color={theme.colors.secondary} />
              <Stat label="Avg Slippage" value={`${eq.avg_slippage_bps || 0} bps`}
                    color={(eq.avg_slippage_bps || 0) > 5 ? theme.colors.warning : theme.colors.textSecondary} />
            </div>

            {/* Alpha verdict banner */}
            {agentBeatsHold && (
              <div style={{
                border: `1px solid ${theme.colors.border}`, borderRadius: '10px',
                padding: '12px 16px', marginBottom: '24px', fontSize: '13px',
                color: theme.colors.textSecondary,
              }}>
                {alpha.first_entry_price
                  ? <>Since first entry at <strong>${alpha.first_entry_price.toFixed(2)}</strong>, the agent has realized+unrealized <strong style={{ color: alpha.agent_pnl >= 0 ? theme.colors.primary : theme.colors.danger }}>${alpha.agent_pnl.toFixed(2)}</strong> on {symbol}; simply holding would be <strong style={{ color: theme.colors.secondary }}>{alpha.buy_hold_return_pct}%</strong>.</>
                  : <>No agent entries on {symbol} yet — nothing to compare against buy-and-hold.</>}
              </div>
            )}

            {/* Decision tape */}
            <SectionTitle icon={Activity} title="Decision Tape" />
            {tape.length > 1 ? (
              <div style={{ height: '220px', marginBottom: '24px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={tape}>
                    <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border} />
                    <XAxis dataKey="ts" tick={{ fontSize: 10, fill: theme.colors.textMuted }} />
                    <YAxis domain={['dataMin - (dataMin * 0.05)', 'dataMax + (dataMax * 0.05)']} tick={{ fontSize: 10, fill: theme.colors.textMuted }} width={55} />
                    <Tooltip
                      contentStyle={{ background: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, borderRadius: '8px', fontSize: '12px' }}
                      formatter={(v, n) => n === 'price' ? [`$${v}`, 'Price'] : [v, n]}
                    />
                    <Line type="monotone" dataKey="price" stroke={theme.colors.secondary} dot={false} strokeWidth={2} />
                    
                    {/* Reference lines for target and entry prices */}
                    {data.target_price && (
                      <ReferenceLine y={data.target_price} stroke={theme.colors.primary} strokeDasharray="3 3" label={{ value: `Target: $${data.target_price.toFixed(2)}`, fill: theme.colors.primary, position: 'top', fontSize: 9 }} />
                    )}
                    {alpha.first_entry_price && (
                      <ReferenceLine y={alpha.first_entry_price} stroke={theme.colors.warning} strokeDasharray="3 3" label={{ value: `Entry: $${alpha.first_entry_price.toFixed(2)}`, fill: theme.colors.warning, position: 'bottom', fontSize: 9 }} />
                    )}

                    {/* Green dots where an order actually executed */}
                    <Scatter dataKey="executed" fill={theme.colors.primary}>
                      {tape.map((d, i) => <Cell key={i} fill={theme.colors.primary} />)}
                    </Scatter>
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div style={{ color: theme.colors.textMuted, padding: '20px', fontSize: '13px' }}>Not enough priced decisions to chart yet.</div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
              {/* Why not? log */}
              <div>
                <SectionTitle icon={AlertTriangle} title="Why did / didn't it trade?" />
                <div style={{ maxHeight: '280px', overflowY: 'auto' }}>
                  {data.decisions_restricted && (
                    <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>
                      Decision internals are visible to operators only.
                    </div>
                  )}
                  {!data.decisions_restricted && decisions.length === 0 && <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>No decisions recorded yet.</div>}
                  {decisions.map((d, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '12px' }}>
                      <div>
                        <span style={{ color: reasonColor(d.skip_reason), fontWeight: 700 }}>
                          {d.executed ? `EXECUTED ${d.action?.toUpperCase()}` : (REASON_LABEL[d.skip_reason] || d.skip_reason || 'hold')}
                        </span>
                        <div style={{ color: theme.colors.textMuted, fontSize: '10px' }}>
                          conf {(d.ensemble_confidence || 0).toFixed(2)}
                          {d.llm_verdict?.reasoning ? ` · LLM: ${String(d.llm_verdict.reasoning).slice(0, 40)}` : ''}
                        </div>
                      </div>
                      <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>{d.ts ? new Date(d.ts).toLocaleTimeString() : ''}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Per-strategy books */}
              <div>
                <SectionTitle icon={Layers} title="Strategy Books (this symbol)" />
                {Object.keys(books).length === 0 && <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>No fills attributed yet.</div>}
                {Object.entries(books).map(([name, b]) => {
                  const total = (b.realized_pnl || 0) + (b.unrealized_pnl || 0);
                  return (
                    <div key={name} style={{ padding: '10px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ fontWeight: 700, fontSize: '12px', textTransform: 'capitalize' }}>{name.replace(/_/g, ' ')}</span>
                        <span style={{ fontWeight: 800, color: total >= 0 ? theme.colors.primary : theme.colors.danger }}>
                          {total >= 0 ? '+' : ''}${total.toFixed(2)}
                        </span>
                      </div>
                      <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>
                        {b.closed_trades || 0} closed · {((b.win_rate || 0) * 100).toFixed(0)}% wins · {(b.open_positions || []).length} open
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const SectionTitle = ({ icon: Icon, title }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
    <Icon size={16} color={theme.colors.primary} />
    <span style={{ fontSize: '13px', fontWeight: 800, textTransform: 'uppercase', color: theme.colors.textSecondary }}>{title}</span>
  </div>
);

export default SymbolDrilldown;
