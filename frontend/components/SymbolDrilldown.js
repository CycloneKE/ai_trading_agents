import { useState, useEffect } from 'react';
import { X, Activity, Layers, AlertTriangle, Cpu } from 'lucide-react';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';
import PriceChart from './PriceChart';
import { columns, money, signedMoney, useIsMobile } from './ui';

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
  already_held: 'Already holding it (no adding to positions)',
  no_position: 'Sell signal, but nothing held (no short selling)',
  order_pending: 'An earlier order is still working',
  position_unknown: 'Could not confirm holdings; skipped for safety',
  insufficient_cash: 'Not enough paper cash for the order',
  add_not_profitable: 'Holding not yet profitable enough to add to',
  max_adds: 'Already added to this holding the maximum times',
  position_cap: 'Holding is at its size limit',
  trimmed_today: 'Already trimmed this position today',
};

const MARKET_LABEL = { nse: 'NSE KENYA', us_equity: 'US STOCK', crypto: 'CRYPTO' };

const reasonColor = (r) => {
  if (!r) return theme.colors.primary;
  if (['halted', 'risk_limits', 'pdt_guard'].includes(r)) return theme.colors.danger;
  if (['llm_veto', 'bias_downgrade'].includes(r)) return theme.colors.accent;
  if (r === 'hold') return theme.colors.textMuted;
  return theme.colors.warning;
};

const Stat = ({ label, value, color, subtitle }) => (
  <div style={{ flex: '1 1 130px', minWidth: 0, background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '10px', border: `1px solid ${theme.colors.border}40` }}>
    <div style={{ fontSize: '10px', color: theme.colors.textMuted, textTransform: 'uppercase', fontWeight: 800, tracking: '0.5px' }}>{label}</div>
    <div style={{ fontSize: '20px', fontWeight: 800, color: color || theme.colors.text, margin: '4px 0' }}>{value}</div>
    {subtitle && <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>{subtitle}</div>}
  </div>
);

const SymbolDrilldown = ({ symbol, onClose }) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [chart, setChart] = useState(null);
  const mobile = useIsMobile();

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

  // NSE stocks are priced in shillings; the chart feed says which market
  // the symbol trades in.
  const currency = chart?.currency || 'USD';
  const lastClose = chart?.bars?.length ? chart.bars[chart.bars.length - 1].close : null;
  const currentPrice = alpha.current_price ?? lastClose;

  const agentBeatsHold = alpha.buy_hold_return_pct != null && alpha.agent_pnl != null;

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, backgroundColor: 'rgba(2,6,23,0.92)',
      backdropFilter: 'blur(12px)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
      zIndex: 1000, padding: mobile ? 0 : '24px', overflowY: 'auto',
    }}>
      {/* On a phone the panel fills the screen. */}
      <div onClick={(e) => e.stopPropagation()} style={{
        ...glassCard, maxWidth: '980px', width: '100%', padding: mobile ? '14px 12px 28px' : '28px', border: `1px solid ${theme.colors.border}`,
        boxShadow: '0 20px 50px rgba(0,0,0,0.6)', ...(mobile ? { borderRadius: 0, minHeight: '100%' } : {}),
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: mobile ? '14px' : '20px', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <h2 style={{ margin: 0, fontSize: mobile ? '22px' : '26px', fontWeight: 800, letterSpacing: '1px', color: '#fff' }}>{symbol}</h2>
            {chart?.market && (
              <span style={{ backgroundColor: 'rgba(59, 130, 246, 0.15)', color: theme.colors.accent, padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 800 }}>
                {MARKET_LABEL[chart.market] || chart.market.toUpperCase()} · {currency}
              </span>
            )}
          </div>
          <button onClick={onClose} aria-label="Close" style={{ background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, borderRadius: '8px', padding: '6px', cursor: 'pointer' }}>
            <X size={20} />
          </button>
        </div>

        {error && (
          <div style={{ color: theme.colors.danger, padding: '20px', background: 'rgba(244,63,94,0.1)', borderRadius: '8px' }}>
            {error === 'not_found'
              ? `"${symbol}" isn't a tracked symbol. Check the ticker, for example SCOM or EQTY (NSE), AAPL or NVDA (US).`
              : `Failed to load: ${error}`}
          </div>
        )}
        {!data && !error && <div style={{ color: theme.colors.textMuted, padding: '40px', textAlign: 'center' }}>Loading…</div>}

        {data && (
          <>
            {/* Headline stats */}
            <div style={{ display: 'flex', gap: mobile ? '8px' : '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
              <Stat label="Current Price" value={currentPrice ? money(currentPrice, currency) : '—'}
                    subtitle={alpha.current_price ? 'Live market feed' : lastClose ? 'Last daily close' : 'No price yet'} />
              <Stat label="Agent P&L" value={signedMoney(alpha.agent_pnl || 0, currency)}
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
                    ? <>Agent entry at <strong style={{ color: '#fff' }}>{money(alpha.first_entry_price, currency)}</strong>. Net performance: <strong style={{ color: alpha.agent_pnl >= 0 ? theme.colors.primary : theme.colors.danger }}>{signedMoney(alpha.agent_pnl, currency)}</strong> (Buy & Hold: <strong style={{ color: theme.colors.secondary }}>{alpha.buy_hold_return_pct}%</strong>).</>
                    : <>No agent entries on {symbol} yet, so there is nothing to compare with buying and holding.</>}
                </div>
              </div>
            )}

            <SectionTitle icon={Activity} title="Daily price chart and the agent's trades" />
            <div style={{ marginBottom: '24px' }}>
              <PriceChart symbol={symbol} height={mobile ? 280 : 360} onLoaded={setChart} />
            </div>

            {/* Order Book Depth & Decisions Grid */}
            <div style={columns('1fr 1fr', mobile, '20px')}>
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
                            {signedMoney(total, currency)}
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
