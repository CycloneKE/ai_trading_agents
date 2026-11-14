import TradingViewDrawer from './TradingViewDrawer';
import { theme } from './DashboardStyles';

const DRAWDOWN_SERIES = [{
  key: 'drawdown', name: 'Below the high', type: 'area', color: theme.colors.danger,
  topColor: 'rgba(244,63,94,0.0)', bottomColor: 'rgba(244,63,94,0.3)',
}];

const AdvancedAnalytics = ({ data, mobile }) => {
  
  const rm = data.riskMetrics || {};
  const currentRisk = rm.current_metrics || {};
  const strategies = data.strategies || [];
  const portfolioChart = data.performance?.portfolio_chart || [];

  // 1. Risk measures from the risk manager. A measure it does not report is
  // shown as a dash: it used to read Beta 1.00 and Leverage 1.00x, in green,
  // when nothing had been measured at all.
  const has = (v) => typeof v === 'number' && !Number.isNaN(v);
  const riskMetrics = [
    { name: 'Value at risk (95%, one day)', raw: currentRisk.portfolio_var, value: (v) => `${(v * 100).toFixed(2)}%`, benchmark: '2.0%', good: (v) => v < 0.02 },
    { name: 'Beta (against the S&P 500)', raw: currentRisk.beta, value: (v) => v.toFixed(2), benchmark: '1.0', good: (v) => Math.abs(v - 1) < 0.2 },
    { name: 'Volatility (annualised)', raw: currentRisk.volatility, value: (v) => `${(v * 100).toFixed(1)}%`, benchmark: '15.0%', good: (v) => v < 0.2 },
    { name: 'Leverage', raw: currentRisk.leverage, value: (v) => `${v.toFixed(2)}x`, benchmark: '2.0x', good: (v) => v <= 1.5 },
  ].map((m) => ({
    ...m,
    shown: has(m.raw) ? m.value(m.raw) : '—',
    status: !has(m.raw) ? 'none' : (m.good(m.raw) ? 'good' : 'warning'),
  }));

  // 2. Drawdown: how far the account is below its highest value so far,
  // from the account-value curve ({timestamp, value} points).
  const drawdownAnalysis = [];
  let peak = 0;
  portfolioChart.forEach((point) => {
    const val = Number(point.value ?? point.portfolio_value ?? point.equity);
    if (!(val > 0)) return;
    if (val > peak) peak = val;
    drawdownAnalysis.push({ timestamp: point.timestamp, drawdown: +(((val - peak) / peak) * 100).toFixed(3) });
  });
  const worst = drawdownAnalysis.reduce((m, p) => Math.min(m, p.drawdown), 0);
  const nowDd = drawdownAnalysis.length ? drawdownAnalysis[drawdownAnalysis.length - 1].drawdown : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? '16px' : '30px' }}>
      {/* Control bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <h3 style={{ fontSize: mobile ? '16px' : '20px', fontWeight: '800', margin: 0, color: '#fff' }}>Risk and strategy analytics</h3>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: mobile ? 'minmax(0, 1fr)' : '1fr 1fr', gap: mobile ? '16px' : '24px' }}>
        {/* Risk Metrics */}
        <div style={{ ...theme.glass, padding: '24px' }}>
          <h4 style={{ fontSize: '15px', fontWeight: '800', marginBottom: '18px', color: '#fff', textTransform: 'uppercase' }}>
            Risk measures
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {riskMetrics.map((metric, index) => (
              <div key={index} style={{ borderBottom: `1px solid ${theme.colors.border}20`, paddingBottom: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '13px', color: theme.colors.textSecondary }}>{metric.name}</span>
                  <span style={{
                    fontSize: '14px',
                    fontWeight: '800',
                    fontVariantNumeric: 'tabular-nums',
                    color: metric.status === 'good' ? theme.colors.primary
                      : metric.status === 'warning' ? theme.colors.warning : theme.colors.textMuted,
                  }}>
                    {metric.shown}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: theme.colors.textMuted }}>
                  <span>Limit or reference</span>
                  <span>{metric.benchmark}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Drawdown Chart */}
        <div style={{ ...theme.glass, padding: '24px' }}>
          <h4 style={{ fontSize: '15px', fontWeight: '800', marginBottom: '18px', color: '#fff', textTransform: 'uppercase' }}>
            Drawdown
          </h4>
          <div style={{ fontSize: '12px', color: theme.colors.textMuted, margin: '-10px 0 12px' }}>
            How far the account is below its highest value so far. Now {nowDd.toFixed(2)}%; the deepest so far {worst.toFixed(2)}%.
          </div>
          <TradingViewDrawer
            data={drawdownAnalysis}
            height={210}
            timeKey="timestamp"
            formatValue={(v) => `${v.toFixed(2)}%`}
            series={DRAWDOWN_SERIES}
            removeWeekendGaps={false}
            emptyMessage="The drawdown appears once the account has been valued for a while."
          />
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
                <th style={{ padding: '10px' }}>REALISED P&L</th>
                <th style={{ padding: '10px' }}>WIN RATE</th>
                <th style={{ padding: '10px' }}>UNREALISED P&L</th>
                <th style={{ padding: '10px', textAlign: 'right' }}>CLOSED TRADES</th>
              </tr>
            </thead>
            <tbody>
              {strategies.length > 0 ? (
                strategies.map((strat, index) => (
                  <tr key={index} style={{ borderBottom: `1px solid ${theme.colors.border}20` }}>
                    <td style={{ padding: '12px 10px', fontWeight: 'bold' }}>{strat.name.toUpperCase()}</td>
                    <td style={{ padding: '12px 10px', color: strat.realized_pnl >= 0 ? theme.colors.primary : theme.colors.danger, fontWeight: '700' }}>
                      {(strat.realized_pnl || 0) < 0 ? '-' : ''}${Math.abs(strat.realized_pnl || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
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
                      padding: '12px 10px', fontVariantNumeric: 'tabular-nums',
                      color: (strat.unrealized_pnl || 0) >= 0 ? theme.colors.primary : theme.colors.danger,
                    }}>
                      {(strat.unrealized_pnl || 0) < 0 ? '-' : ''}${Math.abs(strat.unrealized_pnl || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      <span style={{ color: theme.colors.textMuted, marginLeft: '6px', fontSize: '11px' }}>
                        {(strat.open_positions || []).length} open
                      </span>
                    </td>
                    <td style={{ padding: '12px 10px', textAlign: 'right', color: theme.colors.textSecondary }}>
                      {strat.closed_trades ?? strat.trades_count ?? 0}
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