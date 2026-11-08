// The dashboard shell: header, navigation, the shared data it polls, and
// the section pages under components/views/. On a phone the navigation moves
// to a bar along the bottom of the screen and every page drops to one column.
import { useState, useEffect, useCallback, useRef } from 'react';
import {
  LayoutDashboard, Wallet, Flag, FileText, ChartLine, Shield, LogOut, Search, Bell,
  X, AlertTriangle, HelpCircle, CheckCircle2, ChevronRight, Activity, Zap
} from 'lucide-react';
import MarketClock from './MarketClock';
import HelpPanel from './HelpPanel';
import SymbolDrilldown from './SymbolDrilldown';
import SectorDrilldown from './SectorDrilldown';
import PortfolioSection from './views/PortfolioSection';
import OverviewView from './views/OverviewView';
import ResearchView from './views/ResearchView';
import NseSection from './views/NseSection';
import InsightsSection from './views/InsightsSection';
import SystemSection from './views/SystemSection';
import NotificationsView, { notificationItems } from './views/NotificationsView';
import { SEEN_KEY, newestAlertTime } from './agentAlerts';
import { theme } from './DashboardStyles';
import { useIsMobile } from './ui';
import { getApiBase } from '../utils/apiBase';

const SECTIONS = [
  { id: 'overview', label: 'Overview', short: 'Home', icon: LayoutDashboard },
  { id: 'portfolio', label: 'Portfolio', short: 'Portfolio', icon: Wallet },
  { id: 'nse', label: 'NSE Kenya', short: 'NSE', icon: Flag },
  { id: 'research', label: 'Research', short: 'Research', icon: FileText },
  { id: 'insights', label: 'Insights', short: 'Insights', icon: ChartLine },
  { id: 'system', label: 'Risk & System', short: 'System', icon: Shield },
];
const DEFAULT_SUB = { portfolio: 'all', nse: 'watch', insights: 'analytics', system: 'risk' };

// Pages reached from the header rather than the navigation bar.
const HEADER_PAGES = ['notifications'];

// The dashboard's own source fingerprint, set at build time (next.config.js).
const BUILT_FROM = process.env.NEXT_PUBLIC_SOURCE_HASH || '';

// The page lives in the address (#nse/paper), so a reload keeps it and the
// phone's back button steps back through the pages visited.
function readRoute() {
  const [section, sub] = (typeof window !== 'undefined' ? window.location.hash.slice(1) : '').split('/');
  if (!SECTIONS.some((s) => s.id === section) && !HEADER_PAGES.includes(section)) return { section: 'overview', sub: undefined };
  return { section, sub: sub || DEFAULT_SUB[section] };
}

const Heartbeat = ({ heartbeat }) => {
  const bad = heartbeat.triggered;
  const ok = heartbeat.healthy && !bad;
  const color = bad ? '#f43f5e' : (heartbeat.healthy ? theme.colors.primary : '#fbbf24');
  return (
    <div title={bad ? 'Heartbeat timeout: the kill switch has activated' : `Agent loop alive (${heartbeat.elapsed_seconds}s ago)`} style={{
      display: 'flex', alignItems: 'center', gap: '6px', padding: '5px 10px', borderRadius: '10px', fontSize: '10px', fontWeight: '800',
      background: bad ? 'rgba(244,63,94,0.15)' : (heartbeat.healthy ? 'rgba(16,185,129,0.1)' : 'rgba(251,191,36,0.15)'),
      border: `1px solid ${color}40`, color, whiteSpace: 'nowrap',
      fontFamily: 'JetBrains Mono, monospace',
    }}>
      <span style={{
        width: '7px', height: '7px', borderRadius: '50%', display: 'inline-block', background: color,
        animation: ok ? 'pulse 2s infinite' : 'none', boxShadow: ok ? `0 0 6px ${color}` : 'none',
      }} />
      {bad ? 'TIMEOUT' : (heartbeat.healthy ? `${Math.round(heartbeat.elapsed_seconds)}s` : 'STALE')}
    </div>
  );
};

const iconButtonStyle = {
  background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(255, 255, 255, 0.08)', color: theme.colors.textSecondary,
  height: '34px', minWidth: '34px', borderRadius: '10px', cursor: 'pointer', fontSize: '12px', fontWeight: 800,
  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', padding: '0 10px',
  transition: 'all 0.15s ease',
};

// Global Command / Symbol Quick-Search Palette
const SymbolSearchPalette = ({ isOpen, onClose, onSelect, nseQuotes = [] }) => {
  const [query, setQuery] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    } else {
      setQuery('');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const popularNse = [
    { symbol: 'SCOM', name: 'Safaricom Plc' },
    { symbol: 'EQTY', name: 'Equity Group' },
    { symbol: 'KCB', name: 'KCB Group' },
    { symbol: 'EABL', name: 'East African Breweries' },
    { symbol: 'BAT', name: 'British American Tobacco' },
    { symbol: 'ABSA', name: 'Absa Bank Kenya' },
  ];

  const popularGlobal = [
    { symbol: 'NVDA', name: 'NVIDIA Corp' },
    { symbol: 'AAPL', name: 'Apple Inc' },
    { symbol: 'MSFT', name: 'Microsoft' },
    { symbol: 'TSLA', name: 'Tesla Inc' },
    { symbol: 'BTC/USD', name: 'Bitcoin / USD' },
    { symbol: 'EUR/USD', name: 'Euro / US Dollar' },
  ];

  const cleanQ = query.trim().toUpperCase().replace(/[\/\s]/g, '');

  const filteredQuotes = cleanQ
    ? (nseQuotes || []).filter((q) =>
        q.symbol.toUpperCase().includes(cleanQ) || (q.name && q.name.toUpperCase().includes(cleanQ))
      ).slice(0, 6)
    : [];

  const handleSelect = (sym) => {
    if (!sym) return;
    onSelect(sym.trim().toUpperCase().replace(/[\/\s]/g, ''));
    onClose();
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1200,
        backgroundColor: 'rgba(2, 6, 23, 0.75)',
        backdropFilter: 'blur(8px)',
        display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
        padding: '12vh 16px 20px',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: '540px',
          background: 'rgba(11, 17, 32, 0.95)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: '16px',
          boxShadow: '0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08)',
          overflow: 'hidden',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', padding: '14px 16px', borderBottom: '1px solid rgba(255, 255, 255, 0.08)', gap: '10px' }}>
          <Search size={18} color={theme.colors.secondary} />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search symbol (e.g. SCOM, NVDA, BTC/USD)..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && query.trim()) {
                handleSelect(query);
              } else if (e.key === 'Escape') {
                onClose();
              }
            }}
            style={{
              flex: 1, background: 'transparent', border: 'none', color: '#fff',
              fontSize: '15px', outline: 'none', fontWeight: 600,
            }}
          />
          {query ? (
            <button onClick={() => setQuery('')} style={{ background: 'transparent', border: 'none', color: theme.colors.textMuted, cursor: 'pointer' }}>
              <X size={16} />
            </button>
          ) : (
            <span style={{ fontSize: '10px', background: 'rgba(255,255,255,0.06)', color: theme.colors.textMuted, padding: '2px 6px', borderRadius: '4px', fontWeight: 700 }}>
              ESC
            </span>
          )}
        </div>

        {filteredQuotes.length > 0 && (
          <div style={{ padding: '8px 12px', borderBottom: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <div style={{ fontSize: '10px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase', padding: '4px 6px' }}>Market Matches</div>
            {filteredQuotes.map((q) => (
              <div
                key={q.symbol}
                onClick={() => handleSelect(q.symbol)}
                style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '8px 10px', borderRadius: '8px', cursor: 'pointer',
                  transition: 'background 0.15s ease',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)')}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <div>
                  <span style={{ fontWeight: 800, color: '#fff', marginRight: '8px' }}>{q.symbol}</span>
                  <span style={{ fontSize: '12px', color: theme.colors.textSecondary }}>{q.name}</span>
                </div>
                <div style={{ fontSize: '12px', fontWeight: 700, color: (q.change_pct || 0) >= 0 ? theme.colors.primary : theme.colors.danger }}>
                  {q.price ? `KES ${q.price.toFixed(2)}` : ''}
                </div>
              </div>
            ))}
          </div>
        )}

        <div style={{ padding: '12px 16px', display: 'grid', gap: '12px' }}>
          <div>
            <div style={{ fontSize: '10px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.6px' }}>
              NSE Kenya Equities
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {popularNse.map((item) => (
                <button
                  key={item.symbol}
                  onClick={() => handleSelect(item.symbol)}
                  style={{
                    background: 'rgba(255, 255, 255, 0.04)', border: '1px solid rgba(255, 255, 255, 0.08)',
                    color: '#fff', borderRadius: '8px', padding: '5px 10px', fontSize: '11px',
                    fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = theme.colors.primary;
                    e.currentTarget.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                  }}
                >
                  <span style={{ color: theme.colors.primary }}>●</span>
                  {item.symbol}
                  <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>{item.name}</span>
                </button>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '10px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.6px' }}>
              US Equities & Crypto/FX
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {popularGlobal.map((item) => (
                <button
                  key={item.symbol}
                  onClick={() => handleSelect(item.symbol)}
                  style={{
                    background: 'rgba(255, 255, 255, 0.04)', border: '1px solid rgba(255, 255, 255, 0.08)',
                    color: '#fff', borderRadius: '8px', padding: '5px 10px', fontSize: '11px',
                    fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = theme.colors.secondary;
                    e.currentTarget.style.backgroundColor = 'rgba(14, 165, 233, 0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                  }}
                >
                  <span style={{ color: theme.colors.secondary }}>●</span>
                  {item.symbol}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div style={{ padding: '8px 16px', background: 'rgba(0, 0, 0, 0.25)', borderTop: '1px solid rgba(255, 255, 255, 0.06)', fontSize: '11px', color: theme.colors.textMuted, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Press <strong style={{ color: '#fff' }}>Enter</strong> to view live chart</span>
          <span><strong style={{ color: '#fff' }}>Esc</strong> to close</span>
        </div>
      </div>
    </div>
  );
};

const AdvancedDashboard = ({ onLogout }) => {
  const mobile = useIsMobile();
  const [route, setRoute] = useState(readRoute);
  const [data, setData] = useState({
    loading: true,
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
    agentActivity: [],
    allocation: [],
  });
  const [nseData, setNseData] = useState({ quotes: [], movers: { gainers: [], losers: [] }, sectors: [], status: {}, kes_usd_rate: 0.0077, market_open: false });
  const [isConnected, setIsConnected] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [drilldownSymbol, setDrilldownSymbol] = useState(null);
  const [drilldownSector, setDrilldownSector] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [pendingApprovals, setPendingApprovals] = useState(0);
  const [aiHealth, setAiHealth] = useState(null);
  const [agentAlerts, setAgentAlerts] = useState(null);
  const [alertsSeenAt, setAlertsSeenAt] = useState(null);
  const [heartbeat, setHeartbeat] = useState({ healthy: true, elapsed_seconds: 0, triggered: false });
  const [searchOpen, setSearchOpen] = useState(false);

  useEffect(() => {
    const onHash = () => { setRoute(readRoute()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  // Keyboard shortcut for quick search: Ctrl+K or Cmd+K
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setSearchOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const go = useCallback((section, sub) => {
    const next = sub || DEFAULT_SUB[section];
    window.location.hash = next ? `${section}/${next}` : section;
  }, []);

  // First login: open the getting-started guide automatically.
  useEffect(() => {
    if (!localStorage.getItem('aegis_help_seen')) {
      setShowHelp(true);
      localStorage.setItem('aegis_help_seen', '1');
    }
  }, []);

  const isHalted = !!data.status.trading_halted;
  const isOperator = data.status.role === 'operator';

  // Operator-only anomaly scan: what's unusual today. 403 for viewers.
  useEffect(() => {
    if (!isOperator) { setAnomalies([]); return undefined; }
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/anomalies`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setAnomalies(json.anomalies || []);
      } catch (e) { /* transient */ }
    };
    load();
    const id = setInterval(load, 30000);
    return () => { alive = false; clearInterval(id); };
  }, [isOperator]);

  // Research ratings waiting for the operator's decision, for the bell.
  useEffect(() => {
    if (!isOperator) { setPendingApprovals(0); return undefined; }
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/operator/escalations`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setPendingApprovals((json.escalations || []).length);
      } catch (e) { /* transient */ }
    };
    load();
    const id = setInterval(load, 60000);
    return () => { alive = false; clearInterval(id); };
  }, [isOperator]);

  // Whether the AI services are answering (llm_orchestrator.provider_health).
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/ai/health`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setAiHealth(json);
      } catch (e) { /* transient */ }
    };
    load();
    const id = setInterval(load, 60000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // The agent's own alerts and what is wrong right now (src/agent/alerts.py).
  useEffect(() => {
    try { setAlertsSeenAt(localStorage.getItem(SEEN_KEY)); } catch (e) { /* storage blocked */ }
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/agent-alerts`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setAgentAlerts(json);
      } catch (e) { /* transient */ }
    };
    load();
    const id = setInterval(load, 30000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const markAlertsRead = useCallback(() => {
    const seen = newestAlertTime(agentAlerts && agentAlerts.alerts);
    try { localStorage.setItem(SEEN_KEY, seen); } catch (e) { /* storage blocked: this session only */ }
    setAlertsSeenAt(seen);
  }, [agentAlerts]);

  // Built from different code than the server
  const serverBuilt = data.status.dashboard_source;
  const staleDashboard = !!(serverBuilt && BUILT_FROM && serverBuilt !== BUILT_FROM);
  const notices = notificationItems({ anomalies, pendingApprovals, status: data.status, staleDashboard, ai: aiHealth,
    agentAlerts, alertsSeenAt });
  const needsYou = notices.filter((n) => n.attention).length;

  // Heartbeat monitor polling
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/heartbeat`, { headers: { Authorization: `Bearer ${token}` } });
        if (res.ok) { const json = await res.json(); if (alive) setHeartbeat(json); }
      } catch (e) { /* transient */ }
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const fetchData = async () => {
    const endpoints = [
      'status', 'performance', 'positions', 'alerts', 'news-feed',
      'risk-metrics', 'model-performance', 'strategy-performance',
      'market-heatmap', 'system-health', 'agent-activity', 'portfolio-allocation',
    ];

    const token = localStorage.getItem('trading_token');
    if (!token) { onLogout(); return; }

    let unauthorized = false;
    const newResponses = await Promise.all(endpoints.map(async (endpoint) => {
      try {
        const res = await fetch(`${getApiBase()}/api/${endpoint}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.status === 401) { unauthorized = true; return null; }
        if (!res.ok) return null;
        return await res.json();
      } catch (err) {
        return null;
      }
    }));
    if (unauthorized) { onLogout(); return; }

    const ok = (r) => r != null && !(typeof r === 'object' && !Array.isArray(r) && r.error);

    setData((prev) => ({
      loading: false,
      status: ok(newResponses[0]) ? newResponses[0] : prev.status,
      performance: ok(newResponses[1]) ? newResponses[1] : prev.performance,
      positions: Array.isArray(newResponses[2]) ? newResponses[2] : prev.positions,
      alerts: Array.isArray(newResponses[3]) ? newResponses[3] : prev.alerts,
      news: Array.isArray(newResponses[4]) ? newResponses[4] : prev.news,
      riskMetrics: ok(newResponses[5]) ? newResponses[5] : prev.riskMetrics,
      modelPerf: ok(newResponses[6]) ? newResponses[6] : prev.modelPerf,
      strategies: ok(newResponses[7])
        ? Object.entries(newResponses[7])
          .filter(([, stats]) => stats && typeof stats === 'object' && 'realized_pnl' in stats)
          .map(([name, stats]) => ({ name, ...stats }))
        : prev.strategies,
      heatmap: Array.isArray(newResponses[8]) ? newResponses[8] : prev.heatmap,
      systemHealth: ok(newResponses[9]) ? newResponses[9] : prev.systemHealth,
      agentActivity: Array.isArray(newResponses[10]) ? newResponses[10] : prev.agentActivity,
      allocation: Array.isArray(newResponses[11]) ? newResponses[11] : prev.allocation,
    }));
    setIsConnected(ok(newResponses[0]));
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 20000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const fetchNSE = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const resp = await fetch(`${getApiBase()}/api/nse-market`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (resp.status === 401) { onLogout(); return; }
        const nse = await resp.json();
        if (nse && !nse.error) setNseData(nse);
      } catch (e) { /* transient */ }
    };
    fetchNSE();
    const nseInterval = setInterval(fetchNSE, 30000);
    return () => clearInterval(nseInterval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleHalt = async () => {
    const message = isHalted
      ? 'Resume automated trading?'
      : 'HALT TRADING?\n\nThe agent stops submitting new orders immediately. Open positions stay open (protective stop-losses keep working).';
    if (!window.confirm(message)) return;
    try {
      const token = localStorage.getItem('trading_token');
      const res = await fetch(`${getApiBase()}/api/trading/${isHalted ? 'resume' : 'halt'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ confirm: true }),
      });
      if (res.status === 401) { onLogout(); return; }
      await fetchData();
    } catch (e) { /* transient */ }
  };

  const drill = (symbol) => setDrilldownSymbol(symbol);
  const { section, sub } = route;
  const onSub = (s) => go(section, s);

  const page = () => {
    switch (section) {
      case 'portfolio': return <PortfolioSection sub={sub} onSub={onSub} onGo={go} mobile={mobile} onDrill={drill} />;
      case 'research': return <ResearchView active onChanged={fetchData} onDrill={drill} mobile={mobile} />;
      case 'nse': return <NseSection active sub={sub} onSub={onSub} nseData={nseData} isOperator={isOperator} mobile={mobile} onDrill={drill} />;
      case 'insights': return <InsightsSection sub={sub} onSub={onSub} data={data} mobile={mobile} onSector={setDrilldownSector} />;
      case 'system': return <SystemSection sub={sub} onSub={onSub} data={data} mobile={mobile} />;
      case 'notifications': return <NotificationsView items={notices} isOperator={isOperator} mobile={mobile} onDrill={drill} onGo={(id) => go(id)}
        agentAlerts={agentAlerts} alertsSeenAt={alertsSeenAt} onMarkRead={markAlertsRead} />;
      default: return <OverviewView data={data} isConnected={isConnected} onDrill={drill} mobile={mobile} />;
    }
  };

  const haltButton = isOperator && (
    <button
      onClick={toggleHalt}
      title={isHalted ? 'Resume automated trading' : 'Stop all new orders immediately'}
      style={{
        background: isHalted ? theme.colors.warning : 'rgba(244, 63, 94, 0.12)',
        border: `1px solid ${isHalted ? theme.colors.warning : theme.colors.danger}`,
        color: isHalted ? '#000' : theme.colors.danger,
        height: '34px',
        padding: '0 14px',
        borderRadius: '10px',
        cursor: 'pointer',
        fontSize: '11px',
        fontWeight: 800,
        letterSpacing: '0.5px',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        transition: 'all 0.15s ease',
      }}
    >
      {isHalted ? <AlertTriangle size={14} color="#000" /> : <Shield size={14} color={theme.colors.danger} />}
      {isHalted ? 'RESUME' : 'HALT'}
    </button>
  );

  const bell = (
    <button
      onClick={() => go('notifications')}
      aria-label={`Notifications: ${needsYou} need your attention`}
      title={needsYou ? `${needsYou} notification${needsYou > 1 ? 's' : ''} need your attention` : 'Notifications'}
      style={{
        ...iconButtonStyle,
        position: 'relative',
        borderColor: section === 'notifications' ? theme.colors.primary : 'rgba(255, 255, 255, 0.08)',
        background: section === 'notifications' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(15, 23, 42, 0.65)',
      }}
    >
      <Bell size={15} color={needsYou ? theme.colors.warning : theme.colors.textSecondary} />
      {needsYou > 0 && (
        <span style={{
          position: 'absolute', top: '-5px', right: '-5px', minWidth: '18px', height: '18px', borderRadius: '9px',
          background: theme.colors.danger, color: '#fff', fontSize: '10px', fontWeight: 800,
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0 4px',
          boxShadow: '0 0 8px rgba(244, 63, 94, 0.6)',
        }}>{needsYou > 99 ? '99+' : needsYou}</span>
      )}
    </button>
  );

  const brand = (
    <div style={{ display: 'flex', alignItems: 'center', gap: mobile ? '10px' : '14px', minWidth: 0 }}>
      <div style={{
        background: 'linear-gradient(135deg, #10b981 0%, #0ea5e9 100%)',
        padding: mobile ? '7px' : '8px 10px',
        borderRadius: '12px',
        boxShadow: '0 0 16px rgba(16, 185, 129, 0.25)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}>
        <Shield color="#020617" size={mobile ? 18 : 22} strokeWidth={2.5} />
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <h1 style={{
            fontSize: mobile ? '15px' : '18px', fontWeight: '900', margin: 0,
            letterSpacing: '0.8px', color: '#fff', whiteSpace: 'nowrap',
          }}>
            EMPIRE SLY VAULT
          </h1>
          {!mobile && (
            <span style={{
              fontSize: '10px', backgroundColor: 'rgba(16,185,129,0.12)', color: theme.colors.primary,
              padding: '2px 8px', borderRadius: '6px', border: `1px solid ${theme.colors.primary}35`, fontWeight: '800',
              letterSpacing: '0.5px',
            }}>
              QUANTUM AI
            </span>
          )}
        </div>
        {!mobile && (
          <div style={{ fontSize: '11px', color: theme.colors.textMuted, fontWeight: '700', letterSpacing: '0.4px', marginTop: '1px' }}>
            MULTI-REGION WEALTH & AUTONOMOUS DESK
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div style={{ minHeight: '100vh', backgroundColor: theme.colors.bg, color: theme.colors.text, fontFamily: 'Outfit, sans-serif', overflowX: 'hidden' }}>
      <header style={{
        borderRadius: 0, padding: mobile ? '10px 12px' : '14px 28px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
        background: 'rgba(3, 7, 18, 0.88)',
        backdropFilter: 'blur(16px)',
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        {mobile ? (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
              {brand}
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
                {bell}
                <button onClick={() => setSearchOpen(true)} aria-label="Find a symbol" style={iconButtonStyle}><Search size={15} /></button>
                <button onClick={() => setShowHelp(true)} aria-label="Getting started guide" style={iconButtonStyle}>?</button>
                <button onClick={onLogout} aria-label="Log out" style={iconButtonStyle}><LogOut size={15} /></button>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px', marginTop: '8px', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Heartbeat heartbeat={heartbeat} />
                <MarketClock />
              </div>
              {haltButton}
            </div>
          </>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '16px' }}>
            {brand}

            {/* Streamlined Desktop Navigation Pill Bar */}
            <nav style={{
              display: 'flex', gap: '4px', backgroundColor: 'rgba(15, 23, 42, 0.65)',
              padding: '4px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)',
            }}>
              {SECTIONS.map(({ id, label, icon: Icon }) => {
                const on = section === id;
                return (
                  <button key={id} onClick={() => go(id)} style={{
                    background: on ? 'linear-gradient(135deg, rgba(16,185,129,0.22) 0%, rgba(14,165,233,0.15) 100%)' : 'transparent',
                    border: `1px solid ${on ? theme.colors.primary + '65' : 'transparent'}`,
                    color: on ? '#fff' : theme.colors.textSecondary, fontSize: '11px', fontWeight: '800', cursor: 'pointer',
                    borderRadius: '9px', padding: '7px 13px', display: 'flex', alignItems: 'center', gap: '6px',
                    transition: 'all 0.15s ease', letterSpacing: '0.4px', textTransform: 'uppercase',
                  }}>
                    <Icon size={14} color={on ? theme.colors.primary : theme.colors.textMuted} />
                    {label}
                  </button>
                );
              })}
            </nav>

            {/* Quick Actions & System Telemetry */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Heartbeat heartbeat={heartbeat} />
              <MarketClock />
              
              {/* Quick Search trigger */}
              <button
                onClick={() => setSearchOpen(true)}
                title="Search symbols (Ctrl+K)"
                style={{
                  ...iconButtonStyle,
                  padding: '0 12px',
                  display: 'flex', alignItems: 'center', gap: '8px',
                  color: theme.colors.textSecondary,
                }}
              >
                <Search size={14} color={theme.colors.textMuted} />
                <span style={{ fontSize: '11px', fontWeight: 600 }}>Search...</span>
                <span style={{
                  fontSize: '9px', fontWeight: 800, padding: '1px 5px', borderRadius: '4px',
                  background: 'rgba(255,255,255,0.08)', color: theme.colors.textMuted,
                  fontFamily: 'JetBrains Mono, monospace',
                }}>⌘K</span>
              </button>

              {bell}
              {haltButton}
              <button onClick={() => setShowHelp(true)} title="Getting started guide" style={iconButtonStyle}>?</button>
              <button onClick={onLogout} style={{ ...iconButtonStyle, fontSize: '11px' }} title="Log out">
                <LogOut size={14} /> EXIT
              </button>
            </div>
          </div>
        )}
      </header>

      {isHalted && (
        <div style={{
          backgroundColor: 'rgba(245, 158, 11, 0.18)',
          borderBottom: `1px solid ${theme.colors.warning}`,
          color: '#fbbf24', textAlign: 'center', padding: '10px 16px',
          fontSize: mobile ? '12px' : '13px', fontWeight: 800, letterSpacing: '0.5px',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', flexWrap: 'wrap',
        }}>
          <AlertTriangle size={16} color="#fbbf24" />
          <span>TRADING HALTED{data.status.halt_reason ? ` (${data.status.halt_reason})` : ''}: agent order submission is paused. Protective stop-losses remain active.</span>
          {isOperator && <span style={{ color: '#fff', textDecoration: 'underline', cursor: 'pointer' }} onClick={toggleHalt}>Click to RESUME</span>}
          {data.status.halted_at && (
            <span style={{ fontWeight: 600, color: theme.colors.textMuted }}>
              (Halted: {new Date(data.status.halted_at.endsWith('Z') ? data.status.halted_at : `${data.status.halted_at}Z`).toLocaleTimeString()})
            </span>
          )}
        </div>
      )}

      <main style={{ padding: mobile ? '14px 12px calc(84px + env(safe-area-inset-bottom))' : '28px 32px' }}>
        {page()}
      </main>

      {mobile && (
        <nav style={{
          position: 'fixed', left: 0, right: 0, bottom: 0, zIndex: 900,
          display: 'grid', gridTemplateColumns: `repeat(${SECTIONS.length}, 1fr)`,
          background: 'rgba(3, 7, 18, 0.95)', borderTop: '1px solid rgba(255, 255, 255, 0.08)',
          backdropFilter: 'blur(16px)', paddingBottom: 'env(safe-area-inset-bottom)',
        }}>
          {SECTIONS.map(({ id, short, icon: Icon }) => {
            const on = section === id;
            return (
              <button key={id} onClick={() => go(id)} aria-current={on ? 'page' : undefined} style={{
                background: 'transparent', border: 'none', cursor: 'pointer', padding: '9px 0 8px',
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '3px',
                color: on ? theme.colors.primary : theme.colors.textMuted, fontSize: '10px', fontWeight: 800,
                borderTop: `2px solid ${on ? theme.colors.primary : 'transparent'}`,
              }}>
                <Icon size={19} />
                {short}
              </button>
            );
          })}
        </nav>
      )}

      {/* Global Command Palette / Symbol Quick Search Modal */}
      <SymbolSearchPalette
        isOpen={searchOpen}
        onClose={() => setSearchOpen(false)}
        onSelect={drill}
        nseQuotes={nseData.quotes}
      />

      {showHelp && <HelpPanel onClose={() => setShowHelp(false)} />}
      {drilldownSymbol && <SymbolDrilldown symbol={drilldownSymbol} onClose={() => setDrilldownSymbol(null)} />}
      {drilldownSector && <SectorDrilldown sector={drilldownSector} onClose={() => setDrilldownSector(null)} />}
    </div>
  );
};

export default AdvancedDashboard;
