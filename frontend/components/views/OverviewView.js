// Overview: the headline figures, performance curve, what the agent is
// watching, top positions, allocation and strategy attribution.
import React from 'react';
import {
  ResponsiveContainer, Tooltip, PieChart, Pie, Cell,
} from 'recharts';
import { TrendingUp, Activity, Shield, Zap, Layers, RefreshCw, Globe, ArrowUpRight } from 'lucide-react';
import AgentActivity from '../AgentActivity';
import AgentFocus from '../AgentFocus';
import SleeveDashboard from '../SleeveDashboard';
import TradingViewDrawer from '../TradingViewDrawer';
import { theme } from '../DashboardStyles';
import ErrorBoundary from '../ErrorBoundary';
import { card, columns, SectionHeader, HUDCard, HudRow } from '../ui';

const PERFORMANCE_SERIES = [{ key: 'value', name: 'Account value', type: 'area', color: theme.colors.primary }];

const OverviewView = ({ data, isConnected, onDrill, mobile, heartbeat = {}, lastUpdated = null }) => {
  // The risk report keeps value at risk as a share of the account under
  // current_metrics; this card read a top-level field that does not exist.
  const varRaw = data.riskMetrics?.current_metrics?.portfolio_var;
  const varPct = typeof varRaw === 'number' && varRaw > 0 ? varRaw : null;
  const usEquity = data.performance.us_paper_value ?? data.performance.portfolio_value ?? 0;
  const daysMeasured = data.performance.days_measured ?? 0;
  const minDays = data.performance.min_days_for_ratios ?? 20;
  const ratiosReady = daysMeasured >= minDays;
  // What the agent is doing, from its own heartbeat (not a fixed "OPTIMAL").
  const agentState = (() => {
    if (!isConnected) return { label: 'Unreachable', detail: 'The dashboard cannot reach the agent', color: theme.colors.danger };
    if (data.status?.trading_halted) return { label: 'Halted', detail: data.status.halt_reason || 'New orders are paused', color: theme.colors.warning };
    if (heartbeat.triggered) return { label: 'Stopped', detail: 'Its loop stopped answering', color: theme.colors.danger };
    if (heartbeat.healthy === false) return { label: 'Slow', detail: `Last cycle ${Math.round(heartbeat.elapsed_seconds || 0)}s ago`, color: theme.colors.warning };
    return { label: 'Running', detail: `Last cycle ${Math.round(heartbeat.elapsed_seconds || 0)}s ago`, color: theme.colors.primary };
  })();
  const allocationTotal = (data.allocation || []).reduce((t, a) => t + (a.value || 0), 0);
  const performanceSeries = React.useMemo(() => [
    ...PERFORMANCE_SERIES,
    ...(data.performance.benchmark_name ? [{
      key: 'benchmark', name: data.performance.benchmark_name, type: 'line', color: theme.colors.secondary, lineStyle: 'dashed',
    }] : []),
  ], [data.performance.benchmark_name]);
  const chartMarkers = React.useMemo(() => {
    const raw = data.performance?.trade_markers || [];
    if (raw.length > 0) {
      return raw.map((m) => ({
        time: m.time,
        symbol: m.symbol,
        kind: m.side?.toLowerCase() || 'buy',
        text: `${m.side || 'TRADE'} ${m.symbol}`,
        position: m.side === 'BUY' ? 'belowBar' : 'aboveBar',
        shape: m.side === 'BUY' ? 'arrowUp' : 'arrowDown',
        color: m.side === 'BUY' ? theme.colors.primary : theme.colors.danger,
        id: `trade-${m.symbol}-${m.time}`
      }));
    }
    return (data.agentActivity || [])
      .filter((a) => (a.type === 'BUY' || a.type === 'SELL') && a.symbol)
      .map((a) => ({
        time: a.timestamp || a.time,
        symbol: a.symbol,
        kind: a.type.toLowerCase(),
        text: `${a.type} ${a.symbol}`,
        position: a.type === 'BUY' ? 'belowBar' : 'aboveBar',
        shape: a.type === 'BUY' ? 'arrowUp' : 'arrowDown',
        color: a.type === 'BUY' ? theme.colors.primary : theme.colors.danger,
        id: `activity-${a.symbol}-${a.time || Math.random()}`
      }));
  }, [data.performance?.trade_markers, data.agentActivity]);

  if (data.loading) {
    return (
      <div style={{ padding: mobile ? '40px 16px' : '80px', textAlign: 'center', color: theme.colors.textMuted }}>
        <div style={{
          width: '36px', height: '36px', margin: '0 auto 16px', borderRadius: '50%',
          border: `2px solid ${theme.colors.primary}`, borderTopColor: 'transparent',
          animation: 'spin 1s linear infinite'
        }} />
        <div style={{ fontSize: '13px', letterSpacing: '1.5px', fontWeight: 800, color: '#fff' }}>ESTABLISHING SECURE DESK LINK…</div>
        <div style={{ fontSize: '11px', marginTop: '6px' }}>Synchronizing portfolio telemetry, risk curves and market state</div>
      </div>
    );
  }
  return (
  <div style={{ display: 'grid', gap: mobile ? '14px' : '20px' }}>
    <HudRow mobile={mobile}>
      <HUDCard title="Consolidated Equity"
        value={`$${(data.performance.consolidated_equity ?? data.performance.portfolio_value ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
        subValue={data.performance.total_pnl >= 0 ? `↗ +$${(data.performance.total_pnl || 0).toFixed(2)}` : `↘ -$${Math.abs(data.performance.total_pnl || 0).toFixed(2)}`}
        footer={`US paper: $${(data.performance.us_paper_value ?? data.performance.portfolio_value ?? 0).toLocaleString()} · ${data.performance.nse_is_paper ? 'NSE paper' : 'NSE'}: $${(data.performance.nse_value_usd ?? 0).toLocaleString()}`}
        icon={TrendingUp} color={theme.colors.primary} />
      <HUDCard title="Value at risk"
        value={varPct == null ? '—' : `${(varPct * 100).toFixed(2)}%`}
        subValue={varPct == null ? 'Not measured yet' : `about $${Math.round(varPct * usEquity).toLocaleString()}`}
        footer="The most the US account should lose in a day, 19 days in 20"
        icon={Shield} color={theme.colors.warning} />
      <HUDCard title="Win rate"
        value={(data.performance.total_trades || 0) > 0 ? `${((data.performance.win_rate || 0) * 100).toFixed(1)}%` : '—'}
        subValue={`${data.performance.total_trades || 0} closed trade${data.performance.total_trades === 1 ? '' : 's'}`}
        footer={ratiosReady
          ? `Sharpe ${(data.performance.sharpe_ratio || 0).toFixed(2)} · deepest fall ${(Math.abs(data.performance.max_drawdown || 0) * 100).toFixed(1)}%`
          : `Sharpe after ${minDays} days of history (${daysMeasured} so far) · deepest fall ${(Math.abs(data.performance.max_drawdown || 0) * 100).toFixed(1)}%`}
        icon={Zap} color={theme.colors.secondary} />
      <HUDCard title="Agent"
        value={agentState.label}
        subValue={agentState.detail}
        footer={lastUpdated ? `Dashboard updated ${lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : 'Waiting for the first update'}
        icon={Activity} color={agentState.color} />
    </HudRow>

    <div style={columns('2fr 1fr', mobile)}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? '14px' : '20px', minWidth: 0 }}>
        <div style={card(mobile)}>
          <SectionHeader title="Performance Curve" icon={Activity} badge="LIVE NAV" />
          <TradingViewDrawer
            data={data.performance.portfolio_chart || []}
            markers={chartMarkers}
            onDrill={onDrill}
            height={mobile ? 240 : 300}
            currency="USD"
            timeKey="timestamp"
            series={performanceSeries}
            removeWeekendGaps={false}
            emptyMessage="No portfolio performance curve recorded yet."
          />
        </div>
        <ErrorBoundary name="What the agent is doing"><AgentFocus onDrill={onDrill} /></ErrorBoundary>
        <ErrorBoundary name="The dividend sleeve"><SleeveDashboard /></ErrorBoundary>
        <ErrorBoundary name="Agent activity"><AgentActivity activities={data.agentActivity} onDrill={onDrill} /></ErrorBoundary>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? '14px' : '20px', minWidth: 0 }}>
        {/* Top Active Positions */}
        <div style={card(mobile)}>
          <SectionHeader title="Top Positions" icon={RefreshCw} badge={`${data.positions.length} OPEN`} />
          {data.positions.length > 0 ? (
            <div style={{ display: 'grid', gap: '6px' }}>
              {data.positions.slice(0, 5).map((pos, i) => (
                <div
                  key={i}
                  onClick={() => onDrill(pos.symbol)}
                  title={`Open chart for ${pos.symbol}`}
                  style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '10px 12px', borderRadius: '10px',
                    background: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.05)',
                    cursor: 'pointer', transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.06)';
                    e.currentTarget.style.borderColor = theme.colors.primary + '50';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.02)';
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.05)';
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontWeight: '800', fontSize: '13px', color: '#fff' }}>{pos.symbol}</span>
                    <ArrowUpRight size={12} color={theme.colors.textMuted} />
                  </div>
                  <div style={{
                    padding: '2px 8px', borderRadius: '6px', fontSize: '11px', fontWeight: 800,
                    background: pos.unrealized_pl >= 0 ? `${theme.colors.primary}18` : `${theme.colors.danger}18`,
                    color: pos.unrealized_pl >= 0 ? theme.colors.primary : theme.colors.danger,
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    {pos.unrealized_pl >= 0 ? '+' : ''}{pos.unrealized_pl_pct?.toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ padding: '24px 16px', textAlign: 'center', color: theme.colors.textMuted, fontSize: '12px' }}>
              No active open positions
            </div>
          )}
        </div>
        
        {/* Asset Allocation Pie/Donut Chart */}
        <div style={card(mobile)}>
          <SectionHeader title="Asset Allocation" icon={Layers} />
          <div style={{ height: '170px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
            {data.allocation && data.allocation.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data.allocation}
                    cx="50%"
                    cy="50%"
                    innerRadius={52}
                    outerRadius={72}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {data.allocation.map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'rgba(11, 17, 32, 0.95)',
                      border: '1px solid rgba(255, 255, 255, 0.12)',
                      borderRadius: '8px', fontSize: '11px', color: '#fff',
                    }}
                    formatter={(value) => [`$${Math.round(value).toLocaleString()}`, 'Value']}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>No allocation data</div>
            )}
          </div>
          {/* Scrollable Legend list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '10px' }}>
            {data.allocation && data.allocation.map((entry, idx) => (
              <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', padding: '3px 0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: entry.color }} />
                  <span style={{ fontWeight: '600', color: theme.colors.text }}>{entry.name}</span>
                </div>
                <span style={{ color: theme.colors.textSecondary, fontVariantNumeric: 'tabular-nums', fontWeight: 700 }}>
                  ${Math.round(entry.value).toLocaleString()}
                  <span style={{ color: theme.colors.textMuted, fontWeight: 600, marginLeft: '8px', display: 'inline-block', minWidth: '38px', textAlign: 'right' }}>
                    {allocationTotal ? `${((entry.value / allocationTotal) * 100).toFixed(0)}%` : ''}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Strategy Attribution */}
        <div style={card(mobile)}>
          <SectionHeader title="Strategy Attribution" icon={Layers} />
          {data.strategies.length > 0 ? (
            <div style={{ display: 'grid', gap: '8px' }}>
              {data.strategies.map((s, i) => {
                const totalPnl = s.realized_pnl + (s.unrealized_pnl || 0);
                return (
                  <div key={i} style={{ padding: '8px 10px', borderRadius: '8px', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.04)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', alignItems: 'center' }}>
                      <span style={{ fontWeight: '800', fontSize: '12px', textTransform: 'uppercase', color: '#fff' }}>{s.name}</span>
                      <span style={{
                        fontWeight: '800', fontSize: '12px', fontVariantNumeric: 'tabular-nums',
                        color: totalPnl >= 0 ? theme.colors.primary : theme.colors.danger,
                      }}>
                        {totalPnl >= 0 ? '+' : '-'}${Math.abs(totalPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </span>
                    </div>
                    <div style={{ fontSize: '11px', color: theme.colors.textMuted }}>
                      {s.closed_trades} closed · {((s.win_rate || 0) * 100).toFixed(0)}% wins · {(s.open_positions || []).length} open
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted, fontSize: '12px' }}>No attributed trades yet</div>
          )}
        </div>

        {/* Sector Performance */}
        <div style={card(mobile)}>
          <SectionHeader title="Sector Performance" icon={Globe} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            {data.heatmap.slice(0, 4).map((s, i) => (
              <div key={i} style={{
                padding: '10px 12px', borderRadius: '10px',
                background: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.05)',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '10px', fontWeight: 800, color: theme.colors.textMuted, letterSpacing: '0.4px' }}>
                  {(s.sector || 'N/A').toUpperCase()}
                </div>
                <div style={{
                  fontWeight: '800', fontSize: '14px', marginTop: '4px',
                  fontVariantNumeric: 'tabular-nums',
                  color: (s.performance || 0) >= 0 ? theme.colors.primary : theme.colors.danger,
                }}>
                  {(s.performance || 0) >= 0 ? '+' : ''}{((s.performance || 0) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>
  );
};

export default OverviewView;
