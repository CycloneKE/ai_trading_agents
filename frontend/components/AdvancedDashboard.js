// The dashboard shell: header, navigation, the shared data it polls, and
// the section pages under components/views/. On a phone the navigation moves
// to a bar along the bottom of the screen and every page drops to one column.
import { useState, useEffect, useCallback } from 'react';
import {
  LayoutDashboard, Wallet, Flag, FileText, ChartLine, Shield, LogOut, AlertTriangle, Search,
} from 'lucide-react';
import MarketClock from './MarketClock';
import HelpPanel from './HelpPanel';
import SymbolDrilldown from './SymbolDrilldown';
import SectorDrilldown from './SectorDrilldown';
import UnifiedPortfolio from './UnifiedPortfolio';
import OverviewView from './views/OverviewView';
import ResearchView from './views/ResearchView';
import NseSection from './views/NseSection';
import InsightsSection from './views/InsightsSection';
import SystemSection from './views/SystemSection';
import { theme } from './DashboardStyles';
import { card, useIsMobile } from './ui';
import { getApiBase } from '../utils/apiBase';

const SECTIONS = [
  { id: 'overview', label: 'Overview', short: 'Home', icon: LayoutDashboard },
  { id: 'portfolio', label: 'Portfolio', short: 'Portfolio', icon: Wallet },
  { id: 'nse', label: 'NSE Kenya', short: 'NSE', icon: Flag },
  { id: 'research', label: 'Research', short: 'Research', icon: FileText },
  { id: 'insights', label: 'Insights', short: 'Insights', icon: ChartLine },
  { id: 'system', label: 'Risk & System', short: 'System', icon: Shield },
];
const DEFAULT_SUB = { nse: 'watch', insights: 'analytics', system: 'risk' };

// The page lives in the address (#nse/paper), so a reload keeps it and the
// phone's back button steps back through the pages visited.
function readRoute() {
  const [section, sub] = (typeof window !== 'undefined' ? window.location.hash.slice(1) : '').split('/');
  if (!SECTIONS.some((s) => s.id === section)) return { section: 'overview', sub: undefined };
  return { section, sub: sub || DEFAULT_SUB[section] };
}

const Heartbeat = ({ heartbeat }) => {
  const bad = heartbeat.triggered;
  const ok = heartbeat.healthy && !bad;
  const color = bad ? '#f43f5e' : (heartbeat.healthy ? theme.colors.primary : '#fbbf24');
  return (
    <div title={bad ? 'Heartbeat timeout: the kill switch has activated' : `Agent loop alive (${heartbeat.elapsed_seconds}s ago)`} style={{
      display: 'flex', alignItems: 'center', gap: '6px', padding: '5px 10px', borderRadius: '8px', fontSize: '10px', fontWeight: '800',
      background: bad ? 'rgba(244,63,94,0.15)' : (heartbeat.healthy ? 'rgba(16,185,129,0.1)' : 'rgba(251,191,36,0.15)'),
      border: `1px solid ${color}40`, color, whiteSpace: 'nowrap',
    }}>
      <span style={{
        width: '8px', height: '8px', borderRadius: '50%', display: 'inline-block', background: color,
        animation: ok ? 'pulse 2s infinite' : 'none', boxShadow: ok ? `0 0 6px ${color}` : 'none',
      }} />
      {bad ? 'TIMEOUT' : (heartbeat.healthy ? `${Math.round(heartbeat.elapsed_seconds)}s` : 'STALE')}
    </div>
  );
};

const iconButton = {
  background: 'transparent', border: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary,
  height: '34px', minWidth: '34px', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: 800,
  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', padding: '0 10px',
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
  const [heartbeat, setHeartbeat] = useState({ healthy: true, elapsed_seconds: 0, triggered: false });
  const [searchOpen, setSearchOpen] = useState(false);

  useEffect(() => {
    const onHash = () => { setRoute(readRoute()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
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

    // A response only replaces cached state when it parsed and isn't an error envelope.
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
      case 'portfolio': return <UnifiedPortfolio onDrill={drill} />;
      case 'research': return <ResearchView active onChanged={fetchData} onDrill={drill} mobile={mobile} />;
      case 'nse': return <NseSection active sub={sub} onSub={onSub} nseData={nseData} isOperator={isOperator} mobile={mobile} onDrill={drill} />;
      case 'insights': return <InsightsSection sub={sub} onSub={onSub} data={data} mobile={mobile} onSector={setDrilldownSector} />;
      case 'system': return <SystemSection sub={sub} onSub={onSub} data={data} mobile={mobile} />;
      default: return <OverviewView data={data} isConnected={isConnected} onDrill={drill} mobile={mobile} />;
    }
  };

  const searchBox = (
    <input
      autoFocus={mobile && searchOpen}
      placeholder="Symbol…"
      aria-label="Open a symbol's chart"
      title="Type a ticker and press Enter to open its chart"
      onKeyDown={(e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
          setDrilldownSymbol(e.target.value.trim().toUpperCase());
          e.target.value = '';
          setSearchOpen(false);
        }
      }}
      style={{
        background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border}`, color: theme.colors.text,
        padding: '7px 12px', borderRadius: '8px', outline: 'none',
        // 16px stops phones zooming the page when the box is focused.
        fontSize: mobile ? '16px' : '11px', width: mobile ? '100%' : '110px',
      }}
    />
  );

  const haltButton = isOperator && (
    <button onClick={toggleHalt} title={isHalted ? 'Resume automated trading' : 'Stop all new orders immediately'} style={{
      background: isHalted ? theme.colors.warning : 'rgba(244, 63, 94, 0.12)',
      border: `1px solid ${isHalted ? theme.colors.warning : theme.colors.danger}`,
      color: isHalted ? '#000' : theme.colors.danger, height: '34px',
      padding: '0 14px', borderRadius: '8px', cursor: 'pointer', fontSize: '11px', fontWeight: 800, letterSpacing: '0.5px',
    }}>
      {isHalted ? 'RESUME' : 'HALT'}
    </button>
  );

  const brand = (
    <div style={{ display: 'flex', alignItems: 'center', gap: mobile ? '10px' : '14px', minWidth: 0 }}>
      <div style={{
        background: 'linear-gradient(135deg, #10b981 0%, #3b82f6 100%)', padding: mobile ? '7px' : '10px 12px',
        borderRadius: '12px', boxShadow: '0 0 15px rgba(16, 185, 129, 0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <Shield color="#000" size={mobile ? 18 : 22} strokeWidth={2.5} />
      </div>
      <div style={{ minWidth: 0 }}>
        <h1 style={{ fontSize: mobile ? '15px' : '20px', fontWeight: '900', margin: 0, letterSpacing: mobile ? '1px' : '1.5px', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px', whiteSpace: 'nowrap' }}>
          EMPIRE SLY VAULT
          {!mobile && (
            <span style={{ fontSize: '10px', backgroundColor: 'rgba(16,185,129,0.15)', color: theme.colors.primary, padding: '2px 8px', borderRadius: '4px', border: `1px solid ${theme.colors.primary}40`, fontWeight: '800' }}>
              QUANTUM AI
            </span>
          )}
        </h1>
        {!mobile && (
          <div style={{ fontSize: '11px', color: theme.colors.textMuted, fontWeight: '700', letterSpacing: '0.5px' }}>
            MULTI-REGION WEALTH & AUTONOMOUS DESK
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div style={{ minHeight: '100vh', backgroundColor: theme.colors.bg, color: theme.colors.text, fontFamily: 'Outfit, sans-serif', overflowX: 'hidden' }}>
      <header style={{
        ...theme.glass, borderRadius: 0, padding: mobile ? '10px 12px' : '16px 32px',
        borderBottom: `1px solid ${theme.colors.border}`, background: 'rgba(2, 6, 23, 0.95)', backdropFilter: 'blur(16px)',
      }}>
        {mobile ? (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
              {brand}
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
                <button onClick={() => setSearchOpen((o) => !o)} aria-label="Find a symbol" style={iconButton}><Search size={16} /></button>
                <button onClick={() => setShowHelp(true)} aria-label="Getting started guide" style={iconButton}>?</button>
                <button onClick={onLogout} aria-label="Log out" style={iconButton}><LogOut size={16} /></button>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px', marginTop: '10px', alignItems: 'center' }}>
              <Heartbeat heartbeat={heartbeat} />
              <div style={{ flex: 1, minWidth: 0 }}>{searchOpen && searchBox}</div>
              {haltButton}
            </div>
          </>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            {brand}
            <nav style={{ display: 'flex', gap: '4px', backgroundColor: 'rgba(0, 0, 0, 0.4)', padding: '5px', borderRadius: '12px', border: `1px solid ${theme.colors.border}60`, flexWrap: 'wrap' }}>
              {SECTIONS.map(({ id, label, icon: Icon }) => {
                const on = section === id;
                return (
                  <button key={id} onClick={() => go(id)} style={{
                    background: on ? 'linear-gradient(135deg, rgba(16,185,129,0.2) 0%, rgba(59,130,246,0.2) 100%)' : 'transparent',
                    border: on ? `1px solid ${theme.colors.primary}60` : '1px solid transparent',
                    color: on ? '#fff' : theme.colors.textSecondary, fontSize: '11px', fontWeight: '800', cursor: 'pointer',
                    borderRadius: '8px', padding: '8px 14px', display: 'flex', alignItems: 'center', gap: '6px',
                    transition: 'all 0.2s ease', letterSpacing: '0.5px', textTransform: 'uppercase',
                  }}>
                    <Icon size={14} color={on ? theme.colors.primary : theme.colors.textMuted} />
                    {label}
                  </button>
                );
              })}
            </nav>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Heartbeat heartbeat={heartbeat} />
              <MarketClock />
              {searchBox}
              {haltButton}
              <button onClick={() => setShowHelp(true)} title="Getting started guide" style={iconButton}>?</button>
              <button onClick={onLogout} style={{ ...iconButton, fontSize: '11px' }}><LogOut size={14} /> EXIT</button>
            </div>
          </div>
        )}
      </header>

      {isHalted && (
        <div style={{ backgroundColor: theme.colors.warning, color: '#000', textAlign: 'center', padding: '8px 12px', fontSize: mobile ? '12px' : '13px', fontWeight: 800, letterSpacing: '0.5px' }}>
          ⚠ TRADING HALTED: the agent is not submitting new orders. Protective stop-losses remain active.
        </div>
      )}

      {isOperator && anomalies.length > 0 && (
        <div style={{ padding: mobile ? '12px 12px 0' : '16px 40px 0' }}>
          <div style={card(mobile, { padding: '14px 18px', borderLeft: `3px solid ${theme.colors.warning}` })}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
              <AlertTriangle size={16} color={theme.colors.warning} />
              <span style={{ fontSize: '12px', fontWeight: 800, textTransform: 'uppercase', color: theme.colors.textSecondary }}>
                {anomalies.length} thing{anomalies.length > 1 ? 's' : ''} worth a look
              </span>
            </div>
            {anomalies.slice(0, 5).map((a, i) => {
              const c = a.severity === 'high' ? theme.colors.danger : a.severity === 'medium' ? theme.colors.warning : theme.colors.textMuted;
              return (
                <div key={i} onClick={() => a.symbol && setDrilldownSymbol(a.symbol)}
                     style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '6px 0', fontSize: '13px', cursor: a.symbol ? 'pointer' : 'default' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: c, flexShrink: 0 }} />
                  <span style={{ color: theme.colors.text }}>{a.message}</span>
                  {a.symbol && <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span>}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <main style={{ padding: mobile ? '14px 12px calc(84px + env(safe-area-inset-bottom))' : '32px 40px' }}>
        {page()}
      </main>

      {mobile && (
        <nav style={{
          position: 'fixed', left: 0, right: 0, bottom: 0, zIndex: 900,
          display: 'grid', gridTemplateColumns: `repeat(${SECTIONS.length}, 1fr)`,
          background: 'rgba(2, 6, 23, 0.97)', borderTop: `1px solid ${theme.colors.border}`,
          backdropFilter: 'blur(16px)', paddingBottom: 'env(safe-area-inset-bottom)',
        }}>
          {SECTIONS.map(({ id, short, icon: Icon }) => {
            const on = section === id;
            return (
              <button key={id} onClick={() => go(id)} aria-current={on ? 'page' : undefined} style={{
                background: 'transparent', border: 'none', cursor: 'pointer', padding: '9px 0 8px',
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '3px',
                color: on ? theme.colors.primary : theme.colors.textMuted, fontSize: '10px', fontWeight: 700,
                borderTop: `2px solid ${on ? theme.colors.primary : 'transparent'}`,
              }}>
                <Icon size={20} />
                {short}
              </button>
            );
          })}
        </nav>
      )}

      {showHelp && <HelpPanel onClose={() => setShowHelp(false)} />}
      {drilldownSymbol && <SymbolDrilldown symbol={drilldownSymbol} onClose={() => setDrilldownSymbol(null)} />}
      {drilldownSector && <SectorDrilldown sector={drilldownSector} onClose={() => setDrilldownSector(null)} />}
    </div>
  );
};

export default AdvancedDashboard;
