import { useState, useEffect } from 'react';
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine, AreaChart, Area
} from 'recharts';
import { X, TrendingUp, TrendingDown, Activity, Layers, AlertTriangle, BarChart2, ShieldAlert, Cpu } from 'lucide-react';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

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
  if (!r) return theme.colors.primary;
  if (['halted', 'risk_limits', 'pdt_guard'].includes(r)) return theme.colors.danger;
  if (['llm_veto', 'bias_downgrade'].includes(r)) return theme.colors.accent;
  if (r === 'hold') return theme.colors.textMuted;
  return theme.colors.warning;
};

const Stat = ({ label, value, color, subtitle }) => (
  <div style={{ flex: 1, minWidth: '130px', background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '10px', border: `1px solid ${theme.colors.border}40` }}>
    <div style={{ fontSize: '10px', color: theme.colors.textMuted, textTransform: 'uppercase', fontWeight: 800, tracking: '0.5px' }}>{label}</div>
    <div style={{ fontSize: '20px', fontWeight: 800, color: color || theme.colors.text, margin: '4px 0' }}>{value}</div>
    {subtitle && <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>{subtitle}</div>}
  </div>
);

const SymbolDrilldown = ({ symbol, onClose }) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [showSMA20, setShowSMA20] = useState(true);
  const [showSMA50, setShowSMA50] = useState(true);
  const [showRSI, setShowRSI] = useState(true);

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

  // Build the tape series and compute technical overlays
  const rawTape = [...decisions].filter((d) => d.price).reverse();
  const prices = rawTape.map(d => d.price);

  const tape = rawTape.map((d, idx) => {
    // SMA 20
    let sma20 = null;
    if (idx >= 4) {
      const slice = prices.slice(Math.max(0, idx - 4), idx + 1);
      sma20 = slice.reduce((a, b) => a + b, 0) / slice.length;
    }
    // SMA 50
    let sma50 = null;
    if (idx >= 9) {
      const slice = prices.slice(Math.max(0, idx - 9), idx + 1);
      sma50 = slice.reduce((a, b) => a + b, 0) / slice.length;
    }
    // RSI 14-period approximation
    let rsi = 50;
    if (idx >= 3) {
      let gains = 0, losses = 0;
      for (let j = Math.max(1, idx - 4); j <= idx; j++) {
        const change = prices[j] - prices[j - 1];
        if (change >= 0) gains += change;
        else losses += Math.abs(change);
      }
      const rs = losses === 0 ? 100 : gains / losses;
      rsi = Math.min(100, Math.max(0, 100 - (100 / (1 + rs))));
    }

    return {
      i: idx,
      ts: d.ts ? new Date(d.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '',
      price: d.price,
      sma20,
      sma50,
      rsi,
      buyExecuted: d.executed && d.action?.toLowerCase() === 'buy' ? d.price : null,
      sellExecuted: d.executed && d.action?.toLowerCase() === 'sell' ? d.price : null,
      reason: d.skip_reason,
      action: d.action,
    };
  });

  const agentBeatsHold = alpha.buy_hold_return_pct != null && alpha.agent_pnl != null;

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, backgroundColor: 'rgba(2,6,23,0.92)',
      backdropFilter: 'blur(12px)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
      zIndex: 1000, padding: '24px', overflowY: 'auto',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        ...glassCard, maxWidth: '940px', width: '100%', padding: '28px', border: `1px solid ${theme.colors.border}`,
        boxShadow: '0 20px 50px rgba(0,0,0,0.6)'
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h2 style={{ margin: 0, fontSize: '26px', fontWeight: 800, letterSpacing: '1px', color: '#fff' }}>{symbol}</h2>
            <span style={{ backgroundColor: 'rgba(59, 130, 246, 0.15)', color: theme.colors.accent, padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 800 }}>
              LIVE TRADING VIEW
            </span>
          </div>
          <button onClick={onClose} style={{ background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, borderRadius: '8px', padding: '6px', cursor: 'pointer' }}>
            <X size={20} />
          </button>
        </div>

        {error && (
          <div style={{ color: theme.colors.danger, padding: '20px', background: 'rgba(244,63,94,0.1)', borderRadius: '8px' }}>
            {error === 'not_found'
              ? `"${symbol}" isn't a tracked symbol. Check the ticker — e.g. SCOM, EQTY (NSE) or AAPL, NVDA (US).`
              : `Failed to load: ${error}`}
          </div>
        )}
        {!data && !error && <div style={{ color: theme.colors.textMuted, padding: '40px', textAlign: 'center' }}>Loading symbol analytics & order book depth…</div>}

        {data && (
          <>
            {/* Headline stats */}
            <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
              <Stat label="Current Price" value={alpha.current_price ? `$${alpha.current_price.toFixed(2)}` : '—'} subtitle="Live Market Feed" />
              <Stat label="Agent P&L" value={`${alpha.agent_pnl >= 0 ? '+' : ''}$${(alpha.agent_pnl || 0).toFixed(2)}`}
                    color={alpha.agent_pnl >= 0 ? theme.colors.primary : theme.colors.danger} subtitle="Realized + Floating" />
              <Stat label="Buy & Hold" value={alpha.buy_hold_return_pct != null ? `${alpha.buy_hold_return_pct >= 0 ? '+' : ''}${alpha.buy_hold_return_pct}%` : '—'}
                    color={theme.colors.secondary} subtitle="Benchmark Return" />
              <Stat label="Avg Slippage" value={`${eq.avg_slippage_bps || 0} bps`}
                    color={(eq.avg_slippage_bps || 0) > 5 ? theme.colors.warning : theme.colors.textSecondary} subtitle="Execution Quality" />
            </div>

            {/* Alpha verdict banner */}
            {agentBeatsHold && (
              <div style={{
                background: 'linear-gradient(90deg, rgba(16,185,129,0.08) 0%, rgba(59,130,246,0.08) 100%)',
                border: `1px solid ${theme.colors.primary}40`, borderRadius: '10px',
                padding: '12px 16px', marginBottom: '20px', fontSize: '13px', color: theme.colors.textSecondary,
                display: 'flex', alignItems: 'center', gap: '10px'
              }}>
                <Cpu size={18} color={theme.colors.primary} />
                <div>
                  {alpha.first_entry_price
                    ? <>Agent entry at <strong style={{ color: '#fff' }}>${alpha.first_entry_price.toFixed(2)}</strong>. Net performance: <strong style={{ color: alpha.agent_pnl >= 0 ? theme.colors.primary : theme.colors.danger }}>${alpha.agent_pnl.toFixed(2)}</strong> (Buy & Hold: <strong style={{ color: theme.colors.secondary }}>{alpha.buy_hold_return_pct}%</strong>).</>
                    : <>No agent entries on {symbol} yet — benchmark comparison pending initial trade execution.</>}
                </div>
              </div>
            )}

            {/* Interactive Technical Chart Controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <SectionTitle icon={Activity} title="Technical Price Chart & Execution Markers" />
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => setShowSMA20(!showSMA20)} style={{
                  backgroundColor: showSMA20 ? 'rgba(59,130,246,0.2)' : 'transparent',
                  color: showSMA20 ? theme.colors.accent : theme.colors.textMuted,
                  border: `1px solid ${showSMA20 ? theme.colors.accent : theme.colors.border}`,
                  borderRadius: '6px', padding: '4px 8px', fontSize: '11px', fontWeight: 800, cursor: 'pointer'
                }}>
                  SMA 20
                </button>
                <button onClick={() => setShowSMA50(!showSMA50)} style={{
                  backgroundColor: showSMA50 ? 'rgba(234,179,8,0.2)' : 'transparent',
                  color: showSMA50 ? theme.colors.warning : theme.colors.textMuted,
                  border: `1px solid ${showSMA50 ? theme.colors.warning : theme.colors.border}`,
                  borderRadius: '6px', padding: '4px 8px', fontSize: '11px', fontWeight: 800, cursor: 'pointer'
                }}>
                  SMA 50
                </button>
                <button onClick={() => setShowRSI(!showRSI)} style={{
                  backgroundColor: showRSI ? 'rgba(168,85,247,0.2)' : 'transparent',
                  color: showRSI ? '#c084fc' : theme.colors.textMuted,
                  border: `1px solid ${showRSI ? '#c084fc' : theme.colors.border}`,
                  borderRadius: '6px', padding: '4px 8px', fontSize: '11px', fontWeight: 800, cursor: 'pointer'
                }}>
                  RSI (14)
                </button>
              </div>
            </div>

            {/* Main Price Chart */}
            {tape.length > 1 ? (
              <>
                <div style={{ height: '230px', marginBottom: showRSI ? '12px' : '24px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={tape}>
                      <CartesianGrid strokeDasharray="3 3" stroke={`${theme.colors.border}40`} />
                      <XAxis dataKey="ts" tick={{ fontSize: 10, fill: theme.colors.textMuted }} />
                      <YAxis domain={['dataMin - (dataMin * 0.03)', 'dataMax + (dataMax * 0.03)']} tick={{ fontSize: 10, fill: theme.colors.textMuted }} width={55} />
                      <Tooltip
                        contentStyle={{ background: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, borderRadius: '8px', fontSize: '12px' }}
                        formatter={(v, n) => [typeof v === 'number' ? `$${v.toFixed(2)}` : v, n.toUpperCase()]}
                      />
                      <Line type="monotone" dataKey="price" stroke={theme.colors.secondary} dot={false} strokeWidth={2.5} name="price" />
                      {showSMA20 && <Line type="monotone" dataKey="sma20" stroke={theme.colors.accent} dot={false} strokeWidth={1.5} strokeDasharray="4 4" name="sma20" />}
                      {showSMA50 && <Line type="monotone" dataKey="sma50" stroke={theme.colors.warning} dot={false} strokeWidth={1.5} strokeDasharray="2 2" name="sma50" />}

                      {/* Reference lines */}
                      {data.target_price && (
                        <ReferenceLine y={data.target_price} stroke={theme.colors.primary} strokeDasharray="3 3" label={{ value: `Target: $${data.target_price.toFixed(2)}`, fill: theme.colors.primary, position: 'top', fontSize: 9 }} />
                      )}
                      {alpha.first_entry_price && (
                        <ReferenceLine y={alpha.first_entry_price} stroke={theme.colors.warning} strokeDasharray="3 3" label={{ value: `Entry: $${alpha.first_entry_price.toFixed(2)}`, fill: theme.colors.warning, position: 'bottom', fontSize: 9 }} />
                      )}

                      {/* Buy Executed Markers (Green) */}
                      <Scatter dataKey="buyExecuted" fill={theme.colors.primary}>
                        {tape.map((d, i) => <Cell key={i} fill={theme.colors.primary} r={6} />)}
                      </Scatter>

                      {/* Sell Executed Markers (Red) */}
                      <Scatter dataKey="sellExecuted" fill={theme.colors.danger}>
                        {tape.map((d, i) => <Cell key={i} fill={theme.colors.danger} r={6} />)}
                      </Scatter>
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>

                {/* Dedicated RSI Sub-Chart */}
                {showRSI && (
                  <div style={{ height: '90px', marginBottom: '24px', background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '8px', border: `1px solid ${theme.colors.border}30` }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={tape}>
                        <CartesianGrid strokeDasharray="2 2" stroke={`${theme.colors.border}20`} />
                        <YAxis domain={[0, 100]} ticks={[30, 70]} tick={{ fontSize: 9, fill: theme.colors.textMuted }} width={30} />
                        <ReferenceLine y={70} stroke={theme.colors.danger} strokeDasharray="3 3" />
                        <ReferenceLine y={30} stroke={theme.colors.primary} strokeDasharray="3 3" />
                        <Area type="monotone" dataKey="rsi" stroke="#c084fc" fill="rgba(168,85,247,0.15)" strokeWidth={1.5} name="rsi" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: theme.colors.textMuted, padding: '20px', fontSize: '13px' }}>Not enough priced decisions to chart yet.</div>
            )}

            {/* Order Book Depth & Decisions Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              {/* Decision Log */}
              <div>
                <SectionTitle icon={AlertTriangle} title="AI Decision Rationale & Veto Log" />
                <div style={{ maxHeight: '250px', overflowY: 'auto', paddingRight: '6px' }}>
                  {data.decisions_restricted && (
                    <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>Decision internals are visible to operators only.</div>
                  )}
                  {!data.decisions_restricted && decisions.length === 0 && <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>No decisions recorded yet.</div>}
                  {decisions.map((d, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: `1px solid ${theme.colors.border}30`, fontSize: '12px' }}>
                      <div>
                        <span style={{ color: reasonColor(d.skip_reason), fontWeight: 700 }}>
                          {d.executed ? `EXECUTED ${d.action?.toUpperCase()}` : (REASON_LABEL[d.skip_reason] || d.skip_reason || 'hold')}
                        </span>
                        <div style={{ color: theme.colors.textMuted, fontSize: '10px' }}>
                          conf {(d.ensemble_confidence || 0).toFixed(2)}
                          {d.llm_verdict?.reasoning ? ` · LLM: ${String(d.llm_verdict.reasoning).slice(0, 45)}` : ''}
                        </div>
                      </div>
                      <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>{d.ts ? new Date(d.ts).toLocaleTimeString() : ''}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Per-Strategy Attribution Books */}
              <div>
                <SectionTitle icon={Layers} title="Strategy Books Attribution" />
                <div style={{ maxHeight: '250px', overflowY: 'auto' }}>
                  {Object.keys(books).length === 0 && <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>No fills attributed yet.</div>}
                  {Object.entries(books).map(([name, b]) => {
                    const total = (b.realized_pnl || 0) + (b.unrealized_pnl || 0);
                    return (
                      <div key={name} style={{ padding: '10px 0', borderBottom: `1px solid ${theme.colors.border}30` }}>
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
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const SectionTitle = ({ icon: Icon, title }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
    <Icon size={15} color={theme.colors.primary} />
    <span style={{ fontSize: '12px', fontWeight: 800, textTransform: 'uppercase', color: theme.colors.textSecondary, letterSpacing: '0.5px' }}>{title}</span>
  </div>
);

export default SymbolDrilldown;
