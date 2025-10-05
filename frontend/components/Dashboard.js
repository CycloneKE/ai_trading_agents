import { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const Dashboard = ({ theme, onThemeChange }) => {
  const [data, setData] = useState({
    status: { running: false, uptime: 0, last_trade: null },
    performance: { total_pnl: 0, win_rate: 0, sharpe_ratio: 0, max_drawdown: 0 },
    trades: [],
    positions: [],
    portfolio: { total_value: 100000, available_cash: 50000 }
  });
  
  const [timeframe, setTimeframe] = useState('1D');
  const [selectedMetric, setSelectedMetric] = useState('pnl');
  const [isConnected, setIsConnected] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [chartData, setChartData] = useState([]);

  // Theme colors
  const colors = {
    dark: {
      bg: '#0f0f23', bgSecondary: '#1a1a2e', bgTertiary: '#16213e',
      text: '#e2e8f0', textSecondary: '#94a3b8', textMuted: '#64748b',
      accent: '#3b82f6', success: '#10b981', danger: '#ef4444', warning: '#f59e0b',
      border: '#334155', hover: '#475569'
    },
    light: {
      bg: '#ffffff', bgSecondary: '#f8fafc', bgTertiary: '#f1f5f9',
      text: '#1e293b', textSecondary: '#475569', textMuted: '#64748b',
      accent: '#3b82f6', success: '#059669', danger: '#dc2626', warning: '#d97706',
      border: '#e2e8f0', hover: '#f1f5f9'
    }
  };

  const currentTheme = theme || colors['dark'];

  // Fetch data with error handling
  const fetchData = useCallback(async () => {
    try {
      const [statusRes, perfRes, tradesRes] = await Promise.all([
        fetch('http://localhost:5001/api/status'),
        fetch('http://localhost:5001/api/performance'),
        fetch('http://localhost:5001/api/trades')
      ]);

      if (statusRes.ok && perfRes.ok && tradesRes.ok) {
        const [status, performance, trades] = await Promise.all([
          statusRes.json(), perfRes.json(), tradesRes.json()
        ]);
        
        setData(prev => ({ ...prev, status, performance, trades }));
        setIsConnected(true);
        
        // Generate chart data based on selected metric
        const now = new Date();
        const newChartData = Array.from({ length: 24 }, (_, i) => {
          const baseValue = selectedMetric === 'pnl' ? performance.total_pnl : 
                           selectedMetric === 'volume' ? 5000 : 10;
          return {
            time: new Date(now - (23 - i) * 60 * 60 * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
            pnl: baseValue + (Math.random() - 0.5) * (baseValue * 0.2),
            volume: 5000 + Math.random() * 5000,
            trades: Math.floor(Math.random() * 20) + 5
          };
        });
        setChartData(newChartData);
      }
    } catch (error) {
      setIsConnected(false);
      addNotification('Connection lost', 'danger');
    }
  }, []);

  const addNotification = (message, type = 'info') => {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, message, type, timestamp: new Date() }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData, selectedMetric]);

  // Metric cards data
  const metrics = [
    { 
      title: 'Total P&L', 
      value: `$${data.performance.total_pnl?.toFixed(2) || '0.00'}`,
      change: '+2.4%',
      trend: 'up',
      color: data.performance.total_pnl >= 0 ? currentTheme.success : currentTheme.danger
    },
    { 
      title: 'Win Rate', 
      value: `${(data.performance.win_rate * 100)?.toFixed(1) || '0.0'}%`,
      change: '+0.8%',
      trend: 'up',
      color: currentTheme.accent
    },
    { 
      title: 'Sharpe Ratio', 
      value: data.performance.sharpe_ratio?.toFixed(2) || '0.00',
      change: '-0.1',
      trend: 'down',
      color: currentTheme.warning
    },
    { 
      title: 'Max Drawdown', 
      value: `${(data.performance.max_drawdown * 100)?.toFixed(1) || '0.0'}%`,
      change: '+0.2%',
      trend: 'down',
      color: currentTheme.danger
    }
  ];

  const pieData = [
    { name: 'Momentum', value: 35, color: '#3b82f6' },
    { name: 'Mean Reversion', value: 25, color: '#10b981' },
    { name: 'Sentiment', value: 20, color: '#f59e0b' },
    { name: 'ML Strategy', value: 20, color: '#ef4444' }
  ];

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: currentTheme.bg, 
      color: currentTheme.text,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
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
          <h1 style={{ fontSize: '24px', fontWeight: '700', margin: 0 }}>
            AI Trading Dashboard
          </h1>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px',
            padding: '4px 12px',
            borderRadius: '20px',
            backgroundColor: isConnected ? currentTheme.success : currentTheme.danger,
            fontSize: '12px',
            fontWeight: '600'
          }}>
            <div style={{ 
              width: '6px', 
              height: '6px', 
              borderRadius: '50%', 
              backgroundColor: 'white',
              animation: isConnected ? 'pulse 2s infinite' : 'none'
            }} />
            {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </div>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <select 
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            style={{
              backgroundColor: currentTheme.bgTertiary,
              color: currentTheme.text,
              border: `1px solid ${currentTheme.border}`,
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '14px'
            }}
          >
            <option value="1H">1 Hour</option>
            <option value="1D">1 Day</option>
            <option value="1W">1 Week</option>
            <option value="1M">1 Month</option>
          </select>
          
          <button
            onClick={() => onThemeChange && onThemeChange(currentTheme === colors.dark ? 'light' : 'dark')}
            style={{
              backgroundColor: currentTheme.bgTertiary,
              color: currentTheme.text,
              border: `1px solid ${currentTheme.border}`,
              borderRadius: '6px',
              padding: '8px 12px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      {/* Notifications */}
      <div style={{ position: 'fixed', top: '80px', right: '24px', zIndex: 1000 }}>
        {notifications.map(notification => (
          <div
            key={notification.id}
            style={{
              backgroundColor: currentTheme.bgSecondary,
              border: `1px solid ${currentTheme[notification.type] || currentTheme.border}`,
              borderRadius: '8px',
              padding: '12px 16px',
              marginBottom: '8px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              animation: 'slideIn 0.3s ease-out'
            }}
          >
            <div style={{ fontSize: '14px', fontWeight: '500' }}>
              {notification.message}
            </div>
            <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
              {notification.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>

      <div style={{ padding: '24px' }}>
        {/* Metrics Grid */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '20px', 
          marginBottom: '32px' 
        }}>
          {metrics.map((metric, index) => (
            <div
              key={index}
              style={{
                backgroundColor: currentTheme.bgSecondary,
                border: `1px solid ${currentTheme.border}`,
                borderRadius: '12px',
                padding: '24px',
                transition: 'all 0.2s ease',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = 'none';
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h3 style={{ 
                    fontSize: '14px', 
                    fontWeight: '500', 
                    color: currentTheme.textSecondary,
                    margin: '0 0 8px 0'
                  }}>
                    {metric.title}
                  </h3>
                  <div style={{ 
                    fontSize: '28px', 
                    fontWeight: '700', 
                    color: metric.color,
                    margin: '0 0 4px 0'
                  }}>
                    {metric.value}
                  </div>
                  <div style={{ 
                    fontSize: '12px', 
                    color: metric.trend === 'up' ? currentTheme.success : currentTheme.danger,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    <span>{metric.trend === 'up' ? '↗' : '↘'}</span>
                    {metric.change}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Charts Section */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '2fr 1fr', 
          gap: '24px', 
          marginBottom: '32px' 
        }}>
          {/* Main Chart */}
          <div style={{
            backgroundColor: currentTheme.bgSecondary,
            border: `1px solid ${currentTheme.border}`,
            borderRadius: '12px',
            padding: '24px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: '600', margin: 0 }}>
                Performance Chart
              </h3>
              <div style={{ display: 'flex', gap: '8px' }}>
                {['pnl', 'volume', 'trades'].map(metric => (
                  <button
                    key={metric}
                    onClick={() => setSelectedMetric(metric)}
                    style={{
                      backgroundColor: selectedMetric === metric ? currentTheme.accent : 'transparent',
                      color: selectedMetric === metric ? 'white' : currentTheme.textSecondary,
                      border: `1px solid ${currentTheme.border}`,
                      borderRadius: '6px',
                      padding: '6px 12px',
                      fontSize: '12px',
                      cursor: 'pointer',
                      textTransform: 'uppercase'
                    }}
                  >
                    {metric}
                  </button>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke={currentTheme.border} />
                <XAxis dataKey="time" stroke={currentTheme.textMuted} fontSize={12} />
                <YAxis stroke={currentTheme.textMuted} fontSize={12} />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: currentTheme.bgTertiary,
                    border: `1px solid ${currentTheme.border}`,
                    borderRadius: '8px',
                    color: currentTheme.text
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey={selectedMetric} 
                  stroke={currentTheme.accent} 
                  fill={currentTheme.accent}
                  fillOpacity={0.1}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Strategy Allocation */}
          <div style={{
            backgroundColor: currentTheme.bgSecondary,
            border: `1px solid ${currentTheme.border}`,
            borderRadius: '12px',
            padding: '24px'
          }}>
            <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '20px' }}>
              Strategy Allocation
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: currentTheme.bgTertiary,
                    border: `1px solid ${currentTheme.border}`,
                    borderRadius: '8px',
                    color: currentTheme.text
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ marginTop: '16px' }}>
              {pieData.map((item, index) => (
                <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <div style={{ width: '12px', height: '12px', backgroundColor: item.color, borderRadius: '2px' }} />
                  <span style={{ fontSize: '14px', color: currentTheme.textSecondary }}>{item.name}</span>
                  <span style={{ fontSize: '14px', fontWeight: '600', marginLeft: 'auto' }}>{item.value}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Trades */}
        <div style={{
          backgroundColor: currentTheme.bgSecondary,
          border: `1px solid ${currentTheme.border}`,
          borderRadius: '12px',
          padding: '24px'
        }}>
          <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '20px' }}>
            Recent Trades
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${currentTheme.border}` }}>
                  {['Time', 'Symbol', 'Side', 'Quantity', 'Price', 'P&L'].map(header => (
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
                {[...Array(5)].map((_, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${currentTheme.border}` }}>
                    <td style={{ padding: '12px', fontSize: '14px' }}>
                      {new Date(Date.now() - i * 300000).toLocaleTimeString()}
                    </td>
                    <td style={{ padding: '12px', fontSize: '14px', fontWeight: '600' }}>
                      {['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'][i]}
                    </td>
                    <td style={{ padding: '12px' }}>
                      <span style={{
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        fontWeight: '600',
                        backgroundColor: i % 2 === 0 ? currentTheme.success : currentTheme.danger,
                        color: 'white'
                      }}>
                        {i % 2 === 0 ? 'BUY' : 'SELL'}
                      </span>
                    </td>
                    <td style={{ padding: '12px', fontSize: '14px' }}>
                      {Math.floor(Math.random() * 100) + 1}
                    </td>
                    <td style={{ padding: '12px', fontSize: '14px' }}>
                      ${(Math.random() * 200 + 100).toFixed(2)}
                    </td>
                    <td style={{ 
                      padding: '12px', 
                      fontSize: '14px', 
                      fontWeight: '600',
                      color: i % 3 === 0 ? currentTheme.success : currentTheme.danger
                    }}>
                      {i % 3 === 0 ? '+' : '-'}${(Math.random() * 500).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes slideIn {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default Dashboard;