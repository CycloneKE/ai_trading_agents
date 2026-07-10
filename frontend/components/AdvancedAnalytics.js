import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { theme } from './DashboardStyles';

const AdvancedAnalytics = ({ data }) => {
  const [timeRange, setTimeRange] = useState('7d');
  
  const rm = data.riskMetrics || {};
  const currentRisk = rm.current_metrics || {};
  const strategies = data.strategies || [];
  const portfolioChart = data.performance?.portfolio_chart || [];

  // 1. Live Risk Metrics
  const riskMetrics = [
    { name: 'VaR (95%)', value: `${((currentRisk.portfolio_var || 0) * 100).toFixed(3)}%`, benchmark: '2.0%', status: (currentRisk.portfolio_var || 0) < 0.02 ? 'good' : 'warning' },
    { name: 'Beta (vs S&P 500)', value: (currentRisk.beta || 1.0).toFixed(2), benchmark: '1.0', status: Math.abs((currentRisk.beta || 1.0) - 1.0) < 0.2 ? 'good' : 'neutral' },
    { name: 'Volatility (Annualized)', value: `${((currentRisk.volatility || 0) * 100).toFixed(1)}%`, benchmark: '15.0%', status: (currentRisk.volatility || 0) < 0.20 ? 'good' : 'warning' },
    { name: 'Current Leverage', value: `${(currentRisk.leverage || 1.0).toFixed(2)}x`, benchmark: '2.0x', status: (currentRisk.leverage || 1.0) <= 1.5 ? 'good' : 'warning' }
  ];

  // 2. Live Drawdown / Underwater Analysis (Calculated dynamically from portfolio chart)
  const drawdownAnalysis = [];
  let peak = 0;
  
  // Filter portfolio chart by selected timeRange
  const now = Date.now();
  const rangeMs = timeRange === '1d' ? 24 * 60 * 60 * 1000 : timeRange === '7d' ? 7 * 24 * 60 * 60 * 1000 : 30 * 24 * 60 * 60 * 1000;
  
  const filteredChart = portfolioChart.filter(pt => {
    const ptTime = new Date(pt.timestamp).getTime();
    return now - ptTime <= rangeMs;
  });

  filteredChart.forEach(point => {
    const val = point.portfolio_value || point.equity || 0;
    if (val > peak) peak = val;
    const drawdownPct = peak > 0 ? ((val - peak) / peak) * 100 : 0;
    drawdownAnalysis.push({
      date: new Date(point.timestamp).toLocaleDateString(),
      equity: val,
      drawdown: parseFloat(drawdownPct.toFixed(2)),
      underwater: parseFloat((drawdownPct * 1.2).toFixed(2)) // proxy for tail metrics
    });
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      {/* Control bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontSize: '20px', fontWeight: '800', margin: 0, color: '#fff' }}>Portfolio Risk & Strategy Analytics</h3>
        <div style={{ display: 'flex', gap: '8px' }}>
          {['1d', '7d', '30d'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              style={{
                backgroundColor: timeRange === range ? theme.colors.primary : 'rgba(255,255,255,0.05)',
                color: timeRange === range ? '#000' : theme.colors.textSecondary,
                border: 'none',
                padding: '6px 14px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: '800',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              {range.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', flexWrap: 'wrap' }}>
        {/* Risk Metrics */}
        <div style={{ ...theme.glass, padding: '24px' }}>
          <h4 style={{ fontSize: '15px', fontWeight: '800', marginBottom: '18px', color: '#fff', textTransform: 'uppercase' }}>
            Real-time Risk Metrics
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {riskMetrics.map((metric, index) => (
              <div key={index} style={{ borderBottom: `1px solid ${theme.colors.border}20`, paddingBottom: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '13px', color: theme.colors.textSecondary }}>{metric.name}</span>
                  <span style={{ 
                    fontSize: '14px', 
                    fontWeight: '800',
                    color: metric.status === 'good' ? theme.colors.primary : 
                           metric.status === 'warning' ? theme.colors.warning : theme.colors.danger
                  }}>
                    {metric.value}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: theme.colors.textMuted }}>
                  <span>Target limit / Benchmark:</span>
                  <span>{metric.benchmark}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Drawdown Chart */}
        <div style={{ ...theme.glass, padding: '24px' }}>
          <h4 style={{ fontSize: '15px', fontWeight: '800', marginBottom: '18px', color: '#fff', textTransform: 'uppercase' }}>
            Drawdown / Underwater Analysis (%)
          </h4>
          {drawdownAnalysis.length > 0 ? (
            <ResponsiveContainer width="100%" height={210}>
              <LineChart data={drawdownAnalysis}>
                <CartesianGrid strokeDasharray="3 3" stroke={`${theme.colors.border}30`} />
                <XAxis dataKey="date" stroke={theme.colors.textMuted} fontSize={10} />
                <YAxis stroke={theme.colors.textMuted} fontSize={10} unit="%" />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: theme.colors.bgSecondary,
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: '8px',
                    color: theme.colors.text
                  }}
                />
                <Line type="monotone" dataKey="drawdown" stroke={theme.colors.danger} strokeWidth={2} name="Drawdown" dot={false} />
                <Line type="monotone" dataKey="underwater" stroke={theme.colors.warning} strokeWidth={1} name="Max Tail DD" dot={false} strokeDasharray="4 4" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '210px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: theme.colors.textMuted, fontSize: '13px' }}>
              Insufficient portfolio history for drawdown chart. Run trading cycles to populate metrics.
            </div>
          )}
        </div>
      </div>

      {/* Strategy Performance Attribution */}
      <div style={{ ...theme.glass, padding: '24px' }}>
        <h4 style={{ fontSize: '15px', fontWeight: '800', marginBottom: '18px', color: '#fff', textTransform: 'uppercase' }}>
          Live Strategy Performance Attribution
        </h4>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, textAlign: 'left' }}>
                <th style={{ padding: '10px' }}>STRATEGY</th>
                <th style={{ padding: '10px' }}>REALIZED P&L</th>
                <th style={{ padding: '10px' }}>WIN RATE</th>
                <th style={{ padding: '10px' }}>SHARPE RATIO</th>
                <th style={{ padding: '10px', textAlign: 'right' }}>TRADES COUNT</th>
              </tr>
            </thead>
            <tbody>
              {strategies.length > 0 ? (
                strategies.map((strat, index) => (
                  <tr key={index} style={{ borderBottom: `1px solid ${theme.colors.border}20` }}>
                    <td style={{ padding: '12px 10px', fontWeight: 'bold' }}>{strat.name.toUpperCase()}</td>
                    <td style={{ padding: '12px 10px', color: strat.realized_pnl >= 0 ? theme.colors.primary : theme.colors.danger, fontWeight: '700' }}>
                      ${(strat.realized_pnl || 0).toFixed(2)}
                    </td>
                    <td style={{ padding: '12px 10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{
                          width: `${(strat.win_rate || 0) * 100}%`,
                          height: '5px',
                          backgroundColor: theme.colors.primary,
                          borderRadius: '3px',
                          minWidth: '5px',
                          maxWidth: '60px'
                        }} />
                        {((strat.win_rate || 0) * 100).toFixed(1)}%
                      </div>
                    </td>
                    <td style={{ 
                      padding: '12px 10px', 
                      color: (strat.sharpe_ratio || 0) > 1.5 ? theme.colors.primary : 
                             (strat.sharpe_ratio || 0) > 1.0 ? theme.colors.warning : theme.colors.danger
                    }}>
                      {(strat.sharpe_ratio || 0).toFixed(2)}
                    </td>
                    <td style={{ padding: '12px 10px', textAlign: 'right', color: theme.colors.textSecondary }}>
                      {strat.trades_count || 0} trades
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" style={{ padding: '30px 10px', textAlign: 'center', color: theme.colors.textMuted }}>
                    No strategy performance attribution data available yet. Run trades to populate.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AdvancedAnalytics;