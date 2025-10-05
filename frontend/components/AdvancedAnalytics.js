import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Heatmap } from 'recharts';

const AdvancedAnalytics = ({ theme, data }) => {
  const [selectedAnalysis, setSelectedAnalysis] = useState('risk');
  const [timeRange, setTimeRange] = useState('7d');
  const [analyticsData, setAnalyticsData] = useState({});
  const currentTheme = theme;
  
  useEffect(() => {
    // Update analytics data when timeRange changes
    const updateAnalytics = () => {
      const multiplier = timeRange === '1d' ? 1 : timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 7;
      setAnalyticsData({
        riskMultiplier: multiplier * 0.1,
        performanceMultiplier: multiplier * 0.05,
        correlationMultiplier: multiplier * 0.02
      });
    };
    updateAnalytics();
  }, [timeRange]);

  // Mock advanced analytics data
  const riskMetrics = [
    { name: 'VaR (95%)', value: (2.3 + (analyticsData.riskMultiplier || 0)).toFixed(2), benchmark: 2.5, status: 'good' },
    { name: 'Beta', value: (1.15 + (analyticsData.riskMultiplier || 0) * 0.1).toFixed(2), benchmark: 1.0, status: 'neutral' },
    { name: 'Volatility', value: (18.5 + (analyticsData.riskMultiplier || 0) * 2).toFixed(1), benchmark: 20.0, status: 'good' },
    { name: 'Correlation', value: (0.72 + (analyticsData.correlationMultiplier || 0)).toFixed(2), benchmark: 0.8, status: 'good' }
  ];

  const performanceAttribution = [
    { strategy: 'Momentum', contribution: 45.2, alpha: 2.1, sharpe: 1.8 },
    { strategy: 'Mean Reversion', contribution: 28.7, alpha: 1.5, sharpe: 1.4 },
    { strategy: 'Sentiment', contribution: 15.3, alpha: 0.8, sharpe: 0.9 },
    { strategy: 'ML Strategy', contribution: 10.8, alpha: 1.2, sharpe: 1.1 }
  ];

  const correlationMatrix = [
    { asset: 'AAPL', AAPL: 1.0, GOOGL: 0.65, MSFT: 0.72, TSLA: 0.45, NVDA: 0.58 },
    { asset: 'GOOGL', AAPL: 0.65, GOOGL: 1.0, MSFT: 0.78, TSLA: 0.42, NVDA: 0.61 },
    { asset: 'MSFT', AAPL: 0.72, GOOGL: 0.78, MSFT: 1.0, TSLA: 0.38, NVDA: 0.69 },
    { asset: 'TSLA', AAPL: 0.45, GOOGL: 0.42, MSFT: 0.38, TSLA: 1.0, NVDA: 0.51 },
    { asset: 'NVDA', AAPL: 0.58, GOOGL: 0.61, MSFT: 0.69, TSLA: 0.51, NVDA: 1.0 }
  ];

  const drawdownAnalysis = Array.from({ length: 30 }, (_, i) => ({
    date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toLocaleDateString(),
    drawdown: Math.random() * -10,
    underwater: Math.random() * -15,
    recovery: Math.random() * 5
  }));

  const renderRiskAnalysis = () => (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
      {/* Risk Metrics */}
      <div style={{
        backgroundColor: currentTheme.bgSecondary,
        border: `1px solid ${currentTheme.border}`,
        borderRadius: '12px',
        padding: '20px'
      }}>
        <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: currentTheme.text }}>
          Risk Metrics
        </h4>
        {riskMetrics.map((metric, index) => (
          <div key={index} style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
              <span style={{ fontSize: '14px', color: currentTheme.textSecondary }}>{metric.name}</span>
              <span style={{ 
                fontSize: '14px', 
                fontWeight: '600',
                color: metric.status === 'good' ? currentTheme.success : 
                       metric.status === 'neutral' ? currentTheme.warning : currentTheme.danger
              }}>
                {metric.value}
              </span>
            </div>
            <div style={{ 
              width: '100%', 
              height: '6px', 
              backgroundColor: currentTheme.bgTertiary, 
              borderRadius: '3px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${Math.min((parseFloat(metric.value) / metric.benchmark) * 100, 100)}%`,
                height: '100%',
                backgroundColor: metric.status === 'good' ? currentTheme.success : 
                                metric.status === 'neutral' ? currentTheme.warning : currentTheme.danger,
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
        ))}
      </div>

      {/* Drawdown Chart */}
      <div style={{
        backgroundColor: currentTheme.bgSecondary,
        border: `1px solid ${currentTheme.border}`,
        borderRadius: '12px',
        padding: '20px'
      }}>
        <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: currentTheme.text }}>
          Drawdown Analysis
        </h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={drawdownAnalysis}>
            <CartesianGrid strokeDasharray="3 3" stroke={currentTheme.border} />
            <XAxis dataKey="date" stroke={currentTheme.textMuted} fontSize={10} />
            <YAxis stroke={currentTheme.textMuted} fontSize={10} />
            <Tooltip 
              contentStyle={{
                backgroundColor: currentTheme.bgTertiary,
                border: `1px solid ${currentTheme.border}`,
                borderRadius: '8px',
                color: currentTheme.text
              }}
            />
            <Line type="monotone" dataKey="drawdown" stroke={currentTheme.danger} strokeWidth={2} />
            <Line type="monotone" dataKey="underwater" stroke={currentTheme.warning} strokeWidth={1} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderPerformanceAttribution = () => (
    <div style={{
      backgroundColor: currentTheme.bgSecondary,
      border: `1px solid ${currentTheme.border}`,
      borderRadius: '12px',
      padding: '20px'
    }}>
      <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: currentTheme.text }}>
        Strategy Performance Attribution
      </h4>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${currentTheme.border}` }}>
              {['Strategy', 'Contribution (%)', 'Alpha', 'Sharpe Ratio'].map(header => (
                <th key={header} style={{ 
                  textAlign: 'left', 
                  padding: '12px', 
                  fontSize: '12px', 
                  fontWeight: '600',
                  color: currentTheme.textSecondary,
                  textTransform: 'uppercase'
                }}>
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {performanceAttribution.map((item, index) => (
              <tr key={index} style={{ borderBottom: `1px solid ${currentTheme.border}` }}>
                <td style={{ padding: '12px', fontSize: '14px', fontWeight: '600', color: currentTheme.text }}>
                  {item.strategy}
                </td>
                <td style={{ padding: '12px', fontSize: '14px', color: currentTheme.text }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{
                      width: `${item.contribution}%`,
                      height: '6px',
                      backgroundColor: currentTheme.accent,
                      borderRadius: '3px',
                      minWidth: '20px'
                    }} />
                    {item.contribution.toFixed(1)}%
                  </div>
                </td>
                <td style={{ 
                  padding: '12px', 
                  fontSize: '14px', 
                  color: item.alpha > 1 ? currentTheme.success : currentTheme.warning
                }}>
                  {item.alpha.toFixed(1)}
                </td>
                <td style={{ 
                  padding: '12px', 
                  fontSize: '14px',
                  color: item.sharpe > 1.5 ? currentTheme.success : 
                         item.sharpe > 1 ? currentTheme.warning : currentTheme.danger
                }}>
                  {item.sharpe.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const renderCorrelationMatrix = () => (
    <div style={{
      backgroundColor: currentTheme.bgSecondary,
      border: `1px solid ${currentTheme.border}`,
      borderRadius: '12px',
      padding: '20px'
    }}>
      <h4 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: currentTheme.text }}>
        Asset Correlation Matrix
      </h4>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ padding: '8px', fontSize: '12px', color: currentTheme.textSecondary }}></th>
              {['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'].map(asset => (
                <th key={asset} style={{ 
                  padding: '8px', 
                  fontSize: '12px', 
                  fontWeight: '600',
                  color: currentTheme.textSecondary,
                  textAlign: 'center'
                }}>
                  {asset}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {correlationMatrix.map((row, i) => (
              <tr key={i}>
                <td style={{ 
                  padding: '8px', 
                  fontSize: '12px', 
                  fontWeight: '600',
                  color: currentTheme.textSecondary
                }}>
                  {row.asset}
                </td>
                {['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'].map(asset => (
                  <td key={asset} style={{ 
                    padding: '8px', 
                    textAlign: 'center',
                    backgroundColor: `rgba(59, 130, 246, ${row[asset] * 0.3})`,
                    color: currentTheme.text,
                    fontSize: '12px',
                    fontWeight: '600'
                  }}>
                    {row[asset].toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div style={{
      backgroundColor: currentTheme.bgSecondary,
      border: `1px solid ${currentTheme.border}`,
      borderRadius: '12px',
      padding: '24px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '18px', fontWeight: '600', margin: 0, color: currentTheme.text }}>
          Advanced Analytics
        </h3>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            style={{
              backgroundColor: currentTheme.bgTertiary,
              color: currentTheme.text,
              border: `1px solid ${currentTheme.border}`,
              borderRadius: '6px',
              padding: '6px 12px',
              fontSize: '12px',
              marginRight: '16px'
            }}
          >
            <option value="1d">1 Day</option>
            <option value="7d">7 Days</option>
            <option value="30d">30 Days</option>
          </select>
          {[
            { key: 'risk', label: 'Risk Analysis' },
            { key: 'performance', label: 'Performance' },
            { key: 'correlation', label: 'Correlation' }
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setSelectedAnalysis(tab.key)}
              style={{
                backgroundColor: selectedAnalysis === tab.key ? currentTheme.accent : 'transparent',
                color: selectedAnalysis === tab.key ? 'white' : currentTheme.textSecondary,
                border: `1px solid ${currentTheme.border}`,
                borderRadius: '6px',
                padding: '8px 16px',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: '500'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {selectedAnalysis === 'risk' && renderRiskAnalysis()}
      {selectedAnalysis === 'performance' && renderPerformanceAttribution()}
      {selectedAnalysis === 'correlation' && renderCorrelationMatrix()}
    </div>
  );
};

export default AdvancedAnalytics;