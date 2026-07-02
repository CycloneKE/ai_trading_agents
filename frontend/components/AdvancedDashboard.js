import { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell 
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Activity, Shield, Zap, AlertTriangle, 
  Clock, Layers, ChevronRight, Maximize2, Globe, Cpu, RefreshCw,
  Search, Info, ExternalLink, Play, Square, Pause, Terminal, Flag,
  Lock, Bell, BarChart as BarChartIcon, LogOut
} from 'lucide-react';
import AgentActivity from './AgentActivity';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

const StatCard = ({ label, value, icon: Icon, color }) => (
  <div style={{ ...glassCard, flex: 1, padding: '20px' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
      <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase' }}>{label}</span>
      <Icon size={18} color={color} />
    </div>
    <div style={{ fontSize: '24px', fontWeight: '800', color: '#fff' }}>{value}</div>
  </div>
);

const SectionHeader = ({ title, icon: Icon }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
    <Icon size={20} color={theme.colors.secondary} />
    <h3 style={{ fontSize: '18px', fontWeight: '700', margin: 0, color: '#fff' }}>{title}</h3>
    <div style={{ height: '1px', flex: 1, background: `linear-gradient(90deg, ${theme.colors.border}, transparent)` }} />
  </div>
);

const HUDCard = ({ title, value, subValue, icon: Icon, color }) => (
  <div style={{ ...glassCard, flex: 1, minWidth: '220px', position: 'relative', overflow: 'hidden' }}>
    <div style={{ position: 'absolute', right: '-10px', bottom: '-10px', opacity: 0.05 }}>
      <Icon size={100} color={color} />
    </div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
      <p style={{ fontSize: '12px', color: theme.colors.textSecondary, margin: 0, fontWeight: '700', textTransform: 'uppercase', letterSpacing: '1px' }}>{title}</p>
      <div style={{ backgroundColor: `${color}20`, padding: '6px', borderRadius: '8px', color }}>
        <Icon size={18} />
      </div>
    </div>
    <h2 style={{ fontSize: '28px', fontWeight: '800', margin: 0, color: '#fff' }}>{value}</h2>
    <p style={{ fontSize: '13px', margin: '4px 0 0 0', color: subValue?.includes('-') || subValue?.includes('↘') ? theme.colors.danger : theme.colors.primary }}>
      {subValue}
    </p>
  </div>
);

const AdvancedDashboard = ({ onLogout }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState({
    status: { components: {} }, 
    performance: { portfolio_value: 0, total_pnl: 0, win_rate: 0, sharpe_ratio: 0, max_drawdown: 0, portfolio_chart: [] }, 
    positions: [], 
    alerts: [], 
    news: [], 
    riskMetrics: { portfolio_var: 0, beta: 1.0, volatility: 0.15, current_leverage: 1.0, risk_score: 5.0 }, 
    modelPerf: { accuracy: 0.5, feature_importance: [] }, 
    strategies: [], 
    heatmap: [], 
    systemHealth: { uptime: 0, cpu_usage: 0, memory_usage: 'stable' }, 
    agentActivity: []
  });
  const [nseData, setNseData] = useState({ quotes: [], movers: { gainers: [], losers: [] }, sectors: [], status: {}, kes_usd_rate: 0.0077, market_open: false });
  const [isConnected, setIsConnected] = useState(false);

  const fetchData = async () => {
    const endpoints = [
      'status', 'performance', 'positions', 'alerts', 'news-feed', 
      'risk-metrics', 'model-performance', 'strategy-performance',
      'market-heatmap', 'system-health', 'agent-activity'
    ];
    
    const token = localStorage.getItem('trading_token');
    if (!token) { onLogout(); return; }
    
    const newResponses = [];
    for (const endpoint of endpoints) {
      try {
        const res = await fetch(`${getApiBase()}/api/${endpoint}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.status === 401) { onLogout(); return; }
        const json = await res.json();
        newResponses.push(json);
      } catch (err) {
        newResponses.push({ error: err.message });
      }
    }

    setData(prev => ({
      status: newResponses[0] || prev.status,
      performance: newResponses[1] || prev.performance,
      positions: Array.isArray(newResponses[2]) ? newResponses[2] : prev.positions,
      alerts: Array.isArray(newResponses[3]) ? newResponses[3] : prev.alerts,
      news: Array.isArray(newResponses[4]) ? newResponses[4] : prev.news,
      riskMetrics: newResponses[5] || prev.riskMetrics,
      modelPerf: newResponses[6] || prev.modelPerf,
      strategies: Object.entries(newResponses[7] || {}).map(([name, stats]) => ({ name, ...stats })),
      heatmap: Array.isArray(newResponses[8]) ? newResponses[8] : prev.heatmap,
      systemHealth: newResponses[9] || prev.systemHealth,
      agentActivity: Array.isArray(newResponses[10]) ? newResponses[10] : prev.agentActivity
    }));
    setIsConnected(true);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 20000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchNSE = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const resp = await fetch(`${getApiBase()}/api/nse-market`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (resp.status === 401) { onLogout(); return; }
        const nse = await resp.json();
        if (nse && !nse.error) setNseData(nse);
      } catch (e) {}
    };
    fetchNSE();
    const nseInterval = setInterval(fetchNSE, 30000);
    return () => clearInterval(nseInterval);
  }, []);

  const renderOverview = () => (
    <div style={{ display: 'grid', gap: '30px' }}>
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <HUDCard title="Consolidated Equity" value={`$${(data.performance.portfolio_value || 0).toLocaleString()}`} subValue={data.performance.total_pnl > 0 ? `↗ $${data.performance.total_pnl.toFixed(2)}` : `↘ $${(data.performance.total_pnl || 0).toFixed(2)}`} icon={TrendingUp} color={theme.colors.primary} />
        <HUDCard title="Exposure (VaR)" value={`$${(data.riskMetrics.portfolio_var || 0).toLocaleString()}`} subValue={`Risk Score: ${data.riskMetrics.risk_score?.toFixed(1) || '0.0'}/10`} icon={Shield} color={theme.colors.warning} />
        <HUDCard title="Win Rate" value={`${((data.performance.win_rate || 0) * 100).toFixed(1)}%`} subValue={`${data.performance.total_trades || 0} Trades`} icon={Zap} color={theme.colors.secondary} />
        <HUDCard title="System Health" value={isConnected ? 'OPTIMAL' : 'OFFLINE'} subValue={`${Object.keys(data.status.components || {}).length} Services Active`} icon={Activity} color={theme.colors.accent} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={glassCard}>
            <SectionHeader title="Performance Curve" icon={Activity} />
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={data.performance.portfolio_chart || []}>
                <defs><linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={theme.colors.primary} stopOpacity={0.3}/><stop offset="95%" stopColor={theme.colors.primary} stopOpacity={0}/></linearGradient></defs>
                <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border} vertical={false} />
                <XAxis dataKey="timestamp" stroke={theme.colors.textMuted} fontSize={10} tickFormatter={(t) => new Date(t).toLocaleTimeString()} />
                <YAxis stroke={theme.colors.textMuted} fontSize={10} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}` }} />
                <Area type="monotone" dataKey="value" stroke={theme.colors.primary} fill="url(#colorVal)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <AgentActivity activities={data.agentActivity} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={glassCard}>
            <SectionHeader title="Top Positions" icon={RefreshCw} />
            {data.positions.length > 0 ? (
              data.positions.slice(0, 5).map((pos, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
                  <span style={{ fontWeight: '700' }}>{pos.symbol}</span>
                  <span style={{ color: pos.unrealized_pl >= 0 ? theme.colors.primary : theme.colors.danger }}>{pos.unrealized_pl >= 0 ? '+' : ''}{pos.unrealized_pl_pct?.toFixed(2)}%</span>
                </div>
              ))
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No active positions</div>
            )}
          </div>
          <div style={glassCard}>
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

  const renderKenyaNSE = () => (
    <div style={{ display: 'grid', gap: '30px' }}>
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <HUDCard title="KES / USD" value={`${(1 / (nseData.kes_usd_rate || 0.0077)).toFixed(2)}`} subValue="Central Bank Rate" icon={Globe} color={theme.colors.secondary} />
        <HUDCard title="NSE Status" value={nseData.market_open ? 'OPEN' : 'CLOSED'} subValue={`${nseData.quotes?.length || 0} Symbols`} icon={Activity} color={nseData.market_open ? theme.colors.primary : theme.colors.warning} />
        <HUDCard title="Top NSE Gainer" value={nseData.movers?.gainers?.[0]?.symbol || '—'} subValue={`+${nseData.movers?.gainers?.[0]?.change_pct?.toFixed(2) || 0}%`} icon={TrendingUp} color={theme.colors.primary} />
        <HUDCard title="Top NSE Loser" value={nseData.movers?.losers?.[0]?.symbol || '—'} subValue={`${nseData.movers?.losers?.[0]?.change_pct?.toFixed(2) || 0}%`} icon={TrendingDown} color={theme.colors.danger} />
      </div>
      <div style={glassCard}>
        <SectionHeader title="NSE Market Watch" icon={Flag} />
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: theme.colors.textMuted, fontSize: '12px', borderBottom: `1px solid ${theme.colors.border}` }}>
                <th style={{ padding: '12px' }}>SYMBOL</th><th style={{ padding: '12px' }}>PRICE (KES)</th><th style={{ padding: '12px' }}>CHANGE</th><th style={{ padding: '12px' }}>VOLUME</th>
              </tr>
            </thead>
            <tbody>
              {nseData.quotes.length > 0 ? (
                nseData.quotes.map((q, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${theme.colors.border}` }}>
                    <td style={{ padding: '12px', fontWeight: '700' }}>{q.symbol}</td>
                    <td style={{ padding: '12px' }}>{q.price_kes?.toFixed(2)}</td>
                    <td style={{ padding: '12px', color: q.change_pct >= 0 ? theme.colors.primary : theme.colors.danger }}>{q.change_pct >= 0 ? '+' : ''}{q.change_pct?.toFixed(2)}%</td>
                    <td style={{ padding: '12px' }}>{q.volume?.toLocaleString()}</td>
                  </tr>
                ))
              ) : (
                <tr><td colSpan="4" style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No NSE data available</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderRiskView = () => {
    const rm = data.riskMetrics || {};
    const metrics = rm.current_metrics || {};
    return (
      <div style={{ display: 'grid', gap: '30px' }}>
        <div style={{ display: 'flex', gap: '20px' }}>
          <StatCard label="Portfolio VaR" value={`${((metrics.portfolio_var || 0) * 100).toFixed(4)}%`} icon={Shield} color={theme.colors.warning} />
          <StatCard label="Max Drawdown" value={`${((metrics.max_drawdown || 0) * 100).toFixed(2)}%`} icon={TrendingDown} color={theme.colors.danger} />
          <StatCard label="Leverage" value={`${(metrics.leverage || 1.0).toFixed(2)}x`} icon={Zap} color={theme.colors.primary} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          <div style={glassCard}>
            <SectionHeader title="Risk Limits" icon={Lock} />
            {(rm.risk_limits || []).map((l, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
                <span>{l.name}</span>
                <span style={{ color: l.status === 'ok' ? theme.colors.primary : theme.colors.danger }}>{(l.current_value || 0).toFixed(4)} / {l.threshold}</span>
              </div>
            ))}
          </div>
          <div style={glassCard}>
            <SectionHeader title="Recent Alerts" icon={Bell} />
            {(rm.alerts?.recent || []).length > 0 ? (
              (rm.alerts?.recent || []).map((a, i) => (
                <div key={i} style={{ padding: '10px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', marginBottom: '10px', borderLeft: `4px solid ${theme.colors.danger}` }}>
                  <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>{new Date(a.timestamp).toLocaleTimeString()}</div>
                  <div style={{ fontWeight: '700' }}>{a.limit_name} Breach</div>
                </div>
              ))
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No active risk alerts</div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderMarketView = () => (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
      <div style={glassCard}>
        <SectionHeader title="Intelligence Feed" icon={Globe} />
        {data.news.map((n, i) => (
          <div key={i} style={{ padding: '15px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
              <span style={{ fontSize: '10px', fontWeight: '800', color: n.sentiment_label === 'positive' ? theme.colors.primary : (n.sentiment_label === 'negative' ? theme.colors.danger : theme.colors.textMuted) }}>{n.sentiment_label?.toUpperCase() || 'NEUTRAL'}</span>
              <span style={{ fontSize: '10px', color: theme.colors.textMuted }}>{new Date(n.time).toLocaleTimeString()}</span>
            </div>
            <div style={{ fontWeight: '700', marginBottom: '5px' }}>{n.title}</div>
            <div style={{ fontSize: '12px', color: theme.colors.textSecondary }}>{n.summary}</div>
          </div>
        ))}
      </div>
      <div style={glassCard}>
        <SectionHeader title="Market Heatmap" icon={Layers} />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
          {data.heatmap.map((s, i) => (
            <div key={i} style={{ aspectRatio: '1', borderRadius: '8px', background: s.performance >= 0 ? theme.colors.primary : theme.colors.danger, opacity: 0.1 + Math.abs(s.performance || 0) * 5, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', fontSize: '9px', fontWeight: '800' }}>
              {s.sector}<br/>{((s.performance || 0) * 100).toFixed(1)}%
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderPulseView = () => (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div style={glassCard}>
          <SectionHeader title="System Status" icon={Cpu} />
          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Uptime</span><span>{Math.floor((data.systemHealth?.uptime || 0) / 3600)}h</span></div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px' }}><span>Memory</span><span style={{ color: theme.colors.primary }}>{(data.systemHealth?.memory_usage || 'stable').toUpperCase()}</span></div>
        </div>
        <div style={glassCard}>
          <SectionHeader title="Service Health" icon={Activity} />
          {['Database', 'Scraper', 'RiskMgr', 'Model'].map(s => (
            <div key={s} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span>{s}</span><span style={{ color: theme.colors.primary }}>ONLINE</span>
            </div>
          ))}
        </div>
      </div>
      <div style={glassCard}>
        <SectionHeader title="Agent Log" icon={Terminal} />
        <div style={{ backgroundColor: '#000', padding: '15px', borderRadius: '8px', height: '400px', overflowY: 'auto', fontFamily: 'monospace', fontSize: '11px' }}>
          {data.agentActivity.length > 0 ? (
            data.agentActivity.map((log, i) => (
              <div key={i} style={{ marginBottom: '8px' }}><span style={{ color: theme.colors.textMuted }}>[{new Date(log.timestamp).toLocaleTimeString()}]</span> <span style={{ color: theme.colors.primary }}>{log.component}</span>: {log.message}</div>
            ))
          ) : (
            <div style={{ color: theme.colors.textMuted }}>Waiting for logs...</div>
          )}
        </div>
      </div>
    </div>
  );

  const tabs = [
    { id: 'overview', label: 'DASHBOARD', icon: Activity, component: renderOverview },
    { id: 'nse', label: 'NSE KENYA', icon: Flag, component: renderKenyaNSE },
    { id: 'risk', label: 'RISK', icon: Shield, component: renderRiskView },
    { id: 'market', label: 'MARKET', icon: Globe, component: renderMarketView },
    { id: 'system', label: 'SYSTEM', icon: Cpu, component: renderPulseView }
  ];

  return (
    <div style={{ minHeight: '100vh', backgroundColor: theme.colors.bg, color: theme.colors.text, fontFamily: 'Outfit, sans-serif' }}>
      <header style={{ ...theme.glass, borderRadius: 0, padding: '20px 40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ backgroundColor: theme.colors.primary, padding: '10px', borderRadius: '10px' }}><Zap color="#000" size={20} /></div>
          <h1 style={{ fontSize: '18px', fontWeight: '800', margin: 0 }}>AEGIS AI</h1>
        </div>
        <nav style={{ display: 'flex', gap: '30px' }}>
          {tabs.map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{ background: 'transparent', border: 'none', color: activeTab === tab.id ? theme.colors.primary : theme.colors.textSecondary, fontSize: '12px', fontWeight: '800', cursor: 'pointer', borderBottom: activeTab === tab.id ? `2px solid ${theme.colors.primary}` : '2px solid transparent', padding: '5px 0' }}>{tab.label}</button>
          ))}
        </nav>
        <button onClick={onLogout} style={{ background: 'transparent', border: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, padding: '8px 15px', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}><LogOut size={16} /> SIGN OUT</button>
      </header>
      <main style={{ padding: '40px' }}>{tabs.find(t => t.id === activeTab)?.component()}</main>
    </div>
  );
};

export default AdvancedDashboard;