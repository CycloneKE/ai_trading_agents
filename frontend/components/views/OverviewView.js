// Overview: the headline figures, performance curve, what the agent is
// watching, top positions, allocation and strategy attribution.
import {
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area, Line, PieChart, Pie, Cell,
} from 'recharts';
import { TrendingUp, Activity, Shield, Zap, Layers, RefreshCw, Globe } from 'lucide-react';
import AgentActivity from '../AgentActivity';
import AgentFocus from '../AgentFocus';
import SleeveDashboard from '../SleeveDashboard';
import { theme } from '../DashboardStyles';
import { card, columns, SectionHeader, HUDCard, HudRow } from '../ui';

const OverviewView = ({ data, isConnected, onDrill, mobile }) => {
  if (data.loading) {
    return (
      <div style={{ padding: mobile ? '40px 16px' : '80px', textAlign: 'center', color: theme.colors.textMuted }}>
        <div style={{ fontSize: '14px', letterSpacing: '2px' }}>ESTABLISHING SECURE LINK…</div>
        <div style={{ fontSize: '11px', marginTop: '8px' }}>Loading portfolio, risk and market state</div>
      </div>
    );
  }
  return (
  <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
    <HudRow mobile={mobile}>
      <HUDCard title="Consolidated Equity"
        value={`$${(data.performance.consolidated_equity ?? data.performance.portfolio_value ?? 0).toLocaleString()}`}
        subValue={data.performance.total_pnl > 0 ? `↗ $${data.performance.total_pnl.toFixed(2)}` : `↘ $${(data.performance.total_pnl || 0).toFixed(2)}`}
        footer={`US paper: $${(data.performance.us_paper_value ?? data.performance.portfolio_value ?? 0).toLocaleString()} · ${data.performance.nse_is_paper ? 'NSE paper' : 'NSE'}: $${(data.performance.nse_value_usd ?? 0).toLocaleString()}`}
        icon={TrendingUp} color={theme.colors.primary} />
      <HUDCard title="Exposure (VaR)" value={`$${(data.riskMetrics.portfolio_var || 0).toLocaleString()}`} subValue={`Risk Score: ${data.riskMetrics.risk_score?.toFixed(1) || '0.0'}/10`} icon={Shield} color={theme.colors.warning} />
      <HUDCard title="Win Rate" value={`${((data.performance.win_rate || 0) * 100).toFixed(1)}%`} subValue={`${data.performance.total_trades || 0} Trades`} icon={Zap} color={theme.colors.secondary} />
      <HUDCard title="System Health" value={isConnected ? 'OPTIMAL' : 'DEGRADED'} subValue={isConnected ? `${Object.keys(data.status.components || {}).length} Services Active` : 'Reconnecting…'} icon={Activity} color={isConnected ? theme.colors.accent : theme.colors.warning} />
    </HudRow>

    <div style={columns('2fr 1fr', mobile)}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? '16px' : '24px', minWidth: 0 }}>
        <div style={card(mobile)}>
          <SectionHeader title="Performance Curve" icon={Activity} />
          <ResponsiveContainer width="100%" height={mobile ? 220 : 300}>
            <ComposedChart data={data.performance.portfolio_chart || []}>
              <defs><linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={theme.colors.primary} stopOpacity={0.3}/><stop offset="95%" stopColor={theme.colors.primary} stopOpacity={0}/></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border} vertical={false} />
              <XAxis 
                dataKey="timestamp" 
                stroke={theme.colors.textMuted} 
                fontSize={10} 
                tickFormatter={(t) => {
                  try {
                    const d = new Date(t);
                    const isToday = d.toDateString() === new Date().toDateString();
                    return isToday 
                      ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
                      : d.toLocaleDateString([], { month: 'short', day: 'numeric' });
                  } catch (e) {
                    return t;
                  }
                }} 
              />
              <YAxis stroke={theme.colors.textMuted} fontSize={10} domain={['dataMin - (dataMin * 0.01)', 'dataMax + (dataMax * 0.01)']} />
              <Tooltip contentStyle={{ backgroundColor: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, color: '#fff' }} />
              <Area type="monotone" dataKey="value" name="Account value" stroke={theme.colors.primary} fill="url(#colorVal)" />
              {data.performance.benchmark_name && (
                <Line type="monotone" dataKey="benchmark" name={data.performance.benchmark_name}
                      stroke={theme.colors.secondary} dot={false} strokeWidth={1.5} strokeDasharray="5 4" connectNulls />
              )}
              {data.performance.benchmark_name && <Legend wrapperStyle={{ fontSize: '11px' }} />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <AgentFocus onDrill={onDrill} />
        <SleeveDashboard />
        <AgentActivity activities={data.agentActivity} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? '16px' : '24px', minWidth: 0 }}>
        <div style={card(mobile)}>
          <SectionHeader title="Top Positions" icon={RefreshCw} />
          {data.positions.length > 0 ? (
            data.positions.slice(0, 5).map((pos, i) => (
              <div key={i} onClick={() => onDrill(pos.symbol)} title={`Drill into ${pos.symbol}`} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: `1px solid ${theme.colors.border}`, cursor: 'pointer' }}>
                <span style={{ fontWeight: '700' }}>{pos.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span></span>
                <span style={{ color: pos.unrealized_pl >= 0 ? theme.colors.primary : theme.colors.danger }}>{pos.unrealized_pl >= 0 ? '+' : ''}{pos.unrealized_pl_pct?.toFixed(2)}%</span>
              </div>
            ))
          ) : (
            <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No active positions</div>
          )}
        </div>
        
        {/* Asset Allocation Pie/Donut Chart */}
        <div style={card(mobile)}>
          <SectionHeader title="Asset Allocation" icon={Layers} />
          <div style={{ height: '170px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', marginTop: '10px' }}>
            {data.allocation && data.allocation.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data.allocation}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={70}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {data.allocation.map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, borderRadius: '8px', fontSize: '11px', color: '#fff' }}
                    formatter={(value) => [`$${value.toLocaleString()}`, 'Value']}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>No allocation data</div>
            )}
          </div>
          {/* Scrollable Legend list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '12px', maxHeight: '110px', overflowY: 'auto', paddingRight: '4px' }}>
            {data.allocation && data.allocation.map((entry, idx) => (
              <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: entry.color }} />
                  <span style={{ fontWeight: '600' }}>{entry.name}</span>
                </div>
                <span style={{ color: theme.colors.textSecondary }}>${entry.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={card(mobile)}>
          <SectionHeader title="Strategy Attribution" icon={Layers} />
          {data.strategies.length > 0 ? (
            data.strategies.map((s, i) => (
              <div key={i} style={{ padding: '10px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ fontWeight: '700', fontSize: '12px', textTransform: 'uppercase' }}>{s.name}</span>
                  <span style={{ fontWeight: '800', color: (s.realized_pnl + (s.unrealized_pnl || 0)) >= 0 ? theme.colors.primary : theme.colors.danger }}>
                    {(s.realized_pnl + (s.unrealized_pnl || 0)) >= 0 ? '+' : ''}${(s.realized_pnl + (s.unrealized_pnl || 0)).toFixed(2)}
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: theme.colors.textMuted }}>
                  {s.closed_trades} closed · {((s.win_rate || 0) * 100).toFixed(0)}% wins · {(s.open_positions || []).length} open
                </div>
              </div>
            ))
          ) : (
            <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No attributed trades yet</div>
          )}
        </div>
        <div style={card(mobile)}>
          <SectionHeader title="Sector Performance" icon={Globe} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            {data.heatmap.slice(0, 4).map((s, i) => (
              <div key={i} style={{ padding: '10px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', textAlign: 'center' }}>
                <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>{(s.sector || 'N/A').toUpperCase()}</div>
                <div style={{ fontWeight: '800', color: (s.performance || 0) >= 0 ? theme.colors.primary : theme.colors.danger }}>{((s.performance || 0) * 100).toFixed(1)}%</div>
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
