import { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, Heatmap, ScatterChart, Scatter } from 'recharts';

const AdvancedDashboard = ({ theme = 'dark' }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState({
    status: {}, performance: {}, positions: [], alerts: [], news: [], 
    orderBook: {}, riskMetrics: {}, modelPerf: {}, strategies: [], 
    correlations: [], heatmap: [], calendar: [], systemHealth: {}
  });
  const [isConnected, setIsConnected] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [backtestResults, setBacktestResults] = useState(null);

  const colors = {
    dark: {
      bg: '#0a0a0a', bgSecondary: '#1a1a1a', bgTertiary: '#2a2a2a',
      text: '#ffffff', textSecondary: '#cccccc', textMuted: '#888888',
      accent: '#00d4aa', success: '#00ff88', danger: '#ff4757', warning: '#ffa502',
      border: '#333333', hover: '#444444'
    }
  };
  const currentTheme = colors[theme];

  const fetchData = useCallback(async () => {
    try {
      const endpoints = [
        'status', 'performance', 'positions', 'alerts', 'news-feed', 
        'risk-metrics', 'model-performance', 'strategy-performance',
        'correlation-matrix', 'market-heatmap', 'economic-calendar', 'system-health'
      ];
      
      const responses = await Promise.all(
        endpoints.map(endpoint => 
          fetch(`http://localhost:5001/api/${endpoint}`).then(r => r.json())
        )
      );

      setData({
        status: responses[0], performance: responses[1], positions: responses[2],
        alerts: responses[3], news: responses[4], riskMetrics: responses[5],
        modelPerf: responses[6], strategies: responses[7], correlations: responses[8],
        heatmap: responses[9], calendar: responses[10], systemHealth: responses[11]
      });
      setIsConnected(true);
    } catch (error) {
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const runBacktest = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy: 'momentum',
          start_date: '2023-01-01',
          end_date: '2024-01-01'
        })
      });
      const result = await response.json();
      setBacktestResults(result);
    } catch (error) {
      console.error('Backtest failed:', error);
    }
  };

  const controlStrategy = async (strategy, action) => {
    try {
      await fetch('http://localhost:5001/api/strategy-control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy, action })
      });
      fetchData();
    } catch (error) {
      console.error('Strategy control failed:', error);
    }
  };

  const emergencyStop = async () => {
    try {
      await fetch('http://localhost:5001/api/emergency-stop', { method: 'POST' });
      fetchData();
    } catch (error) {
      console.error('Emergency stop failed:', error);
    }
  };

  const TabButton = ({ id, label, active, onClick }) => (
    <button
      onClick={() => onClick(id)}
      style={{
        padding: '8px 16px',
        backgroundColor: active ? currentTheme.accent : 'transparent',
        color: active ? '#000' : currentTheme.text,
        border: `1px solid ${currentTheme.border}`,
        borderRadius: '4px',
        cursor: 'pointer',
        fontSize: '14px',
        fontWeight: '500'
      }}
    >
      {label}
    </button>
  );

  const MetricCard = ({ title, value, change, color }) => (
    <div style={{
      backgroundColor: currentTheme.bgSecondary,
      border: `1px solid ${currentTheme.border}`,
      borderRadius: '8px',
      padding: '16px',
      minWidth: '200px'
    }}>
      <div style={{ fontSize: '12px', color: currentTheme.textMuted, marginBottom: '4px' }}>
        {title}
      </div>
      <div style={{ fontSize: '24px', fontWeight: 'bold', color: color || currentTheme.text }}>
        {value}
      </div>
      {change && (
        <div style={{ fontSize: '12px', color: change > 0 ? currentTheme.success : currentTheme.danger }}>
          {change > 0 ? '↗' : '↘'} {Math.abs(change)}%
        </div>
      )}
    </div>
  );

  const renderOverview = () => (
    <div style={{ display: 'grid', gap: '20px' }}>
      {/* Key Metrics */}
      <div style={{ display: 'flex', gap: '16px', overflowX: 'auto' }}>
        <MetricCard title="Portfolio Value" value={`$${data.performance.portfolio_value?.toLocaleString()}`} change={2.4} color={currentTheme.success} />
        <MetricCard title="Total P&L" value={`$${data.performance.total_pnl?.toFixed(2)}`} change={data.performance.total_pnl > 0 ? 1.2 : -1.2} />
        <MetricCard title="Win Rate" value={`${(data.performance.win_rate * 100)?.toFixed(1)}%`} />
        <MetricCard title="Sharpe Ratio" value={data.performance.sharpe_ratio?.toFixed(2)} />
        <MetricCard title="Max Drawdown" value={`${(data.performance.max_drawdown * 100)?.toFixed(1)}%`} color={currentTheme.danger} />
        <MetricCard title="VaR (95%)" value={`$${data.riskMetrics.portfolio_var?.toLocaleString()}`} color={currentTheme.warning} />
      </div>

      {/* Charts Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        {/* Performance Chart */}
        <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Portfolio Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={backtestResults?.chart_data || []}>
              <CartesianGrid strokeDasharray="3 3" stroke={currentTheme.border} />
              <XAxis dataKey="date" stroke={currentTheme.textMuted} />
              <YAxis stroke={currentTheme.textMuted} />
              <Tooltip contentStyle={{ backgroundColor: currentTheme.bgTertiary, border: `1px solid ${currentTheme.border}` }} />
              <Area type="monotone" dataKey="value" stroke={currentTheme.accent} fill={currentTheme.accent} fillOpacity={0.2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Strategy Allocation */}
        <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Strategy Performance</h3>
          {data.strategies.map((strategy, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', padding: '8px', backgroundColor: currentTheme.bgTertiary, borderRadius: '4px' }}>
              <div>
                <div style={{ fontWeight: 'bold', fontSize: '14px' }}>{strategy.name}</div>
                <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
                  {strategy.trades} trades • {(strategy.win_rate * 100).toFixed(1)}% win rate
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ color: strategy.pnl > 0 ? currentTheme.success : currentTheme.danger, fontWeight: 'bold' }}>
                  ${strategy.pnl}
                </div>
                <button
                  onClick={() => controlStrategy(strategy.name.toLowerCase().replace(' ', '_'), strategy.active ? 'stop' : 'start')}
                  style={{
                    padding: '4px 8px',
                    fontSize: '10px',
                    backgroundColor: strategy.active ? currentTheme.danger : currentTheme.success,
                    color: '#fff',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  {strategy.active ? 'STOP' : 'START'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Positions Table */}
      <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Current Positions</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${currentTheme.border}` }}>
                {['Symbol', 'Quantity', 'Avg Price', 'Current Price', 'P&L', 'P&L %', 'Market Value'].map(header => (
                  <th key={header} style={{ textAlign: 'left', padding: '8px', fontSize: '12px', color: currentTheme.textMuted }}>
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.positions.map((pos, i) => (
                <tr key={i} style={{ borderBottom: `1px solid ${currentTheme.border}` }}>
                  <td style={{ padding: '8px', fontWeight: 'bold' }}>{pos.symbol}</td>
                  <td style={{ padding: '8px' }}>{pos.quantity}</td>
                  <td style={{ padding: '8px' }}>${pos.avg_price?.toFixed(2)}</td>
                  <td style={{ padding: '8px' }}>${pos.current_price?.toFixed(2)}</td>
                  <td style={{ padding: '8px', color: pos.pnl > 0 ? currentTheme.success : currentTheme.danger }}>
                    ${pos.pnl?.toFixed(2)}
                  </td>
                  <td style={{ padding: '8px', color: pos.pnl_percent > 0 ? currentTheme.success : currentTheme.danger }}>
                    {pos.pnl_percent?.toFixed(2)}%
                  </td>
                  <td style={{ padding: '8px' }}>${pos.market_value?.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderRiskManagement = () => (
    <div style={{ display: 'grid', gap: '20px' }}>
      {/* Risk Metrics */}
      <div style={{ display: 'flex', gap: '16px', overflowX: 'auto' }}>
        <MetricCard title="Portfolio VaR" value={`$${data.riskMetrics.portfolio_var?.toLocaleString()}`} color={currentTheme.warning} />
        <MetricCard title="Expected Shortfall" value={`$${data.riskMetrics.expected_shortfall?.toLocaleString()}`} color={currentTheme.danger} />
        <MetricCard title="Beta" value={data.riskMetrics.beta?.toFixed(2)} />
        <MetricCard title="Volatility" value={`${(data.riskMetrics.volatility * 100)?.toFixed(1)}%`} />
        <MetricCard title="Leverage" value={`${data.riskMetrics.current_leverage?.toFixed(2)}x`} />
        <MetricCard title="Risk Score" value={`${data.riskMetrics.risk_score?.toFixed(1)}/10`} color={currentTheme.warning} />
      </div>

      {/* Correlation Matrix */}
      <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Correlation Matrix</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '4px', fontSize: '12px' }}>
          {data.correlations.flat?.()?.map((cell, i) => (
            <div key={i} style={{
              padding: '8px',
              backgroundColor: `rgba(${cell?.correlation > 0 ? '0,212,170' : '255,71,87'}, ${Math.abs(cell?.correlation || 0)})`,
              color: '#fff',
              textAlign: 'center',
              borderRadius: '4px'
            }}>
              {cell?.correlation?.toFixed(2)}
            </div>
          ))}
        </div>
      </div>

      {/* Alerts */}
      <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ margin: 0, fontSize: '16px' }}>Risk Alerts</h3>
          <button
            onClick={emergencyStop}
            style={{
              padding: '8px 16px',
              backgroundColor: currentTheme.danger,
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            🛑 EMERGENCY STOP
          </button>
        </div>
        {data.alerts.map((alert, i) => (
          <div key={i} style={{
            padding: '12px',
            marginBottom: '8px',
            backgroundColor: currentTheme.bgTertiary,
            borderLeft: `4px solid ${alert.severity === 'high' ? currentTheme.danger : alert.severity === 'medium' ? currentTheme.warning : currentTheme.accent}`,
            borderRadius: '4px'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{alert.message}</div>
            <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
              {alert.type.toUpperCase()} • {alert.time}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderMarketIntelligence = () => (
    <div style={{ display: 'grid', gap: '20px' }}>
      {/* Market Heatmap */}
      <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Sector Performance</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '8px' }}>
          {data.heatmap.map((sector, i) => (
            <div key={i} style={{
              padding: '16px',
              backgroundColor: sector.change > 0 ? currentTheme.success : currentTheme.danger,
              color: '#fff',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{sector.sector}</div>
              <div style={{ fontSize: '18px', fontWeight: 'bold' }}>{sector.change?.toFixed(2)}%</div>
              <div style={{ fontSize: '12px', opacity: 0.8 }}>Vol: {(sector.volume / 1000000).toFixed(1)}M</div>
            </div>
          ))}
        </div>
      </div>

      {/* News Feed */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Market News</h3>
          {data.news.map((item, i) => (
            <div key={i} style={{
              padding: '12px',
              marginBottom: '12px',
              backgroundColor: currentTheme.bgTertiary,
              borderRadius: '8px',
              borderLeft: `4px solid ${item.sentiment > 0.5 ? currentTheme.success : item.sentiment < -0.5 ? currentTheme.danger : currentTheme.warning}`
            }}>
              <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{item.title}</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: currentTheme.textMuted }}>
                <span>Sentiment: {(item.sentiment * 100).toFixed(0)}%</span>
                <span>{item.time}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Economic Calendar */}
        <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Economic Calendar</h3>
          {data.calendar.map((event, i) => (
            <div key={i} style={{
              padding: '8px',
              marginBottom: '8px',
              backgroundColor: currentTheme.bgTertiary,
              borderRadius: '4px'
            }}>
              <div style={{ fontWeight: 'bold', fontSize: '14px' }}>{event.event}</div>
              <div style={{ fontSize: '12px', color: currentTheme.textMuted, marginBottom: '4px' }}>
                {event.time} • Impact: {event.impact}
              </div>
              <div style={{ fontSize: '12px' }}>
                Forecast: {event.forecast} | Previous: {event.previous}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderAIAnalytics = () => (
    <div style={{ display: 'grid', gap: '20px' }}>
      {/* Model Performance */}
      <div style={{ display: 'flex', gap: '16px', overflowX: 'auto' }}>
        <MetricCard title="Model Accuracy" value={`${(data.modelPerf.accuracy * 100)?.toFixed(1)}%`} color={currentTheme.accent} />
        <MetricCard title="Precision" value={`${(data.modelPerf.precision * 100)?.toFixed(1)}%`} />
        <MetricCard title="Recall" value={`${(data.modelPerf.recall * 100)?.toFixed(1)}%`} />
        <MetricCard title="F1 Score" value={`${(data.modelPerf.f1_score * 100)?.toFixed(1)}%`} />
        <MetricCard title="Confidence" value={`${(data.modelPerf.prediction_confidence * 100)?.toFixed(1)}%`} color={currentTheme.success} />
      </div>

      {/* Feature Importance */}
      <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Feature Importance</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data.modelPerf.feature_importance || []}>
            <CartesianGrid strokeDasharray="3 3" stroke={currentTheme.border} />
            <XAxis dataKey="feature" stroke={currentTheme.textMuted} />
            <YAxis stroke={currentTheme.textMuted} />
            <Tooltip contentStyle={{ backgroundColor: currentTheme.bgTertiary, border: `1px solid ${currentTheme.border}` }} />
            <Bar dataKey="importance" fill={currentTheme.accent} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Backtesting */}
      <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ margin: 0, fontSize: '16px' }}>Backtesting Results</h3>
          <button
            onClick={runBacktest}
            style={{
              padding: '8px 16px',
              backgroundColor: currentTheme.accent,
              color: '#000',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            Run Backtest
          </button>
        </div>
        {backtestResults && (
          <div>
            <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
              <MetricCard title="Total Return" value={`${backtestResults.total_return}%`} color={backtestResults.total_return > 0 ? currentTheme.success : currentTheme.danger} />
              <MetricCard title="Sharpe Ratio" value={backtestResults.sharpe_ratio} />
              <MetricCard title="Max Drawdown" value={`${backtestResults.max_drawdown}%`} color={currentTheme.danger} />
              <MetricCard title="Win Rate" value={`${(backtestResults.win_rate * 100).toFixed(1)}%`} />
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={backtestResults.chart_data}>
                <CartesianGrid strokeDasharray="3 3" stroke={currentTheme.border} />
                <XAxis dataKey="date" stroke={currentTheme.textMuted} />
                <YAxis stroke={currentTheme.textMuted} />
                <Tooltip contentStyle={{ backgroundColor: currentTheme.bgTertiary, border: `1px solid ${currentTheme.border}` }} />
                <Line type="monotone" dataKey="value" stroke={currentTheme.accent} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );

  const renderSystemHealth = () => (
    <div style={{ display: 'grid', gap: '20px' }}>
      {/* System Metrics */}
      <div style={{ display: 'flex', gap: '16px', overflowX: 'auto' }}>
        <MetricCard title="Uptime" value={`${data.systemHealth.uptime?.toFixed(1)}%`} color={currentTheme.success} />
        <MetricCard title="CPU Usage" value={`${data.systemHealth.cpu_usage}%`} color={currentTheme.warning} />
        <MetricCard title="Memory Usage" value={`${data.systemHealth.memory_usage}%`} />
        <MetricCard title="Error Rate" value={`${(data.systemHealth.error_rate * 100)?.toFixed(2)}%`} color={currentTheme.danger} />
        <MetricCard title="Active Connections" value={data.systemHealth.active_connections} />
      </div>

      {/* API Latency */}
      <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>API Latency</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
          {Object.entries(data.systemHealth.api_latency || {}).map(([api, latency]) => (
            <div key={api} style={{
              padding: '12px',
              backgroundColor: currentTheme.bgTertiary,
              borderRadius: '8px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{ fontWeight: 'bold' }}>{api.toUpperCase()}</span>
              <span style={{ color: latency < 50 ? currentTheme.success : latency < 100 ? currentTheme.warning : currentTheme.danger }}>
                {latency}ms
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Component Status */}
      <div style={{ backgroundColor: currentTheme.bgSecondary, border: `1px solid ${currentTheme.border}`, borderRadius: '8px', padding: '16px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Component Status</h3>
        <div style={{ display: 'grid', gap: '8px' }}>
          {Object.entries(data.status.components || {}).map(([component, info]) => (
            <div key={component} style={{
              padding: '12px',
              backgroundColor: currentTheme.bgTertiary,
              borderRadius: '8px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div>
                <div style={{ fontWeight: 'bold' }}>{component.replace('_', ' ').toUpperCase()}</div>
                <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
                  {info.latency && `Latency: ${info.latency}`}
                  {info.feeds && ` • Feeds: ${info.feeds}`}
                  {info.active_strategies && ` • Strategies: ${info.active_strategies}`}
                  {info.alerts && ` • Alerts: ${info.alerts}`}
                </div>
              </div>
              <div style={{
                padding: '4px 8px',
                backgroundColor: info.status === 'connected' || info.status === 'active' || info.status === 'running' || info.status === 'monitoring' ? currentTheme.success : currentTheme.danger,
                color: '#fff',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 'bold'
              }}>
                {info.status?.toUpperCase()}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const tabs = [
    { id: 'overview', label: 'Overview', component: renderOverview },
    { id: 'risk', label: 'Risk Management', component: renderRiskManagement },
    { id: 'market', label: 'Market Intelligence', component: renderMarketIntelligence },
    { id: 'ai', label: 'AI Analytics', component: renderAIAnalytics },
    { id: 'system', label: 'System Health', component: renderSystemHealth }
  ];

  return (
    <div style={{ minHeight: '100vh', backgroundColor: currentTheme.bg, color: currentTheme.text, fontFamily: 'system-ui' }}>
      {/* Header */}
      <header style={{
        backgroundColor: currentTheme.bgSecondary,
        borderBottom: `1px solid ${currentTheme.border}`,
        padding: '16px 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: 0 }}>
            🤖 Advanced AI Trading Dashboard
          </h1>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '4px 12px',
            borderRadius: '20px',
            backgroundColor: isConnected ? currentTheme.success : currentTheme.danger,
            fontSize: '12px',
            fontWeight: 'bold',
            color: '#000'
          }}>
            <div style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: '#000',
              animation: isConnected ? 'pulse 2s infinite' : 'none'
            }} />
            {isConnected ? 'LIVE' : 'OFFLINE'}
          </div>
        </div>
        
        <div style={{ fontSize: '14px', color: currentTheme.textMuted }}>
          Last Update: {new Date().toLocaleTimeString()}
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav style={{
        backgroundColor: currentTheme.bgSecondary,
        borderBottom: `1px solid ${currentTheme.border}`,
        padding: '12px 24px',
        display: 'flex',
        gap: '8px',
        overflowX: 'auto'
      }}>
        {tabs.map(tab => (
          <TabButton
            key={tab.id}
            id={tab.id}
            label={tab.label}
            active={activeTab === tab.id}
            onClick={setActiveTab}
          />
        ))}
      </nav>

      {/* Main Content */}
      <main style={{ padding: '24px' }}>
        {tabs.find(tab => tab.id === activeTab)?.component()}
      </main>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default AdvancedDashboard;