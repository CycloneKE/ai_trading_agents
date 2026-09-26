import { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  AreaChart, Area, XAxis, YAxis, CartesianGrid, BarChart, Bar
} from 'recharts';
import { 
  Globe, Shield, TrendingUp, TrendingDown, DollarSign, Activity, 
  Layers, RefreshCw, CheckCircle, Clock, Filter, ArrowUpRight, ArrowDownRight,
  Zap, Lightbulb, Compass, Award, Percent, Layers3, Cpu
} from 'lucide-react';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';
import { fxSourceNote } from './ui';

const REGION_COLORS = {
  'US': '#3b82f6',
  'Crypto': '#a855f7',
  'Kenya/Africa': '#10b981'
};

// Shown only until the server answers with its market rate
// (src/connectors/fx_rate.py); this used to be the rate itself.
const FALLBACK_KES_PER_USD = 130;

// Below this many closed trades, Sharpe and drawdown are noise, not evidence.
// The card shows how far along the agent is instead of a number.
const MIN_TRADES_FOR_RATIOS = 20;

export default function UnifiedPortfolio({ onDrill }) {
  const [loading, setLoading] = useState(true);
  const [portfolioData, setPortfolioData] = useState({ account: {}, positions: [], summary: {} });
  const [regionFilter, setRegionFilter] = useState('ALL');
  const [tradeHistory, setTradeHistory] = useState([]);
  // Real metrics and equity history from /api/performance. Null until loaded,
  // so nothing is shown as a result before one exists.
  const [perf, setPerf] = useState(null);

  // Shillings per dollar from the server's live rate, with where it came from.
  const fx = portfolioData.fx || null;
  const KES_FX_RATE = fx?.kes_per_usd || FALLBACK_KES_PER_USD;

  const token = typeof window !== 'undefined' ? localStorage.getItem('trading_token') : null;

  const fetchPortfolio = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/api/portfolio`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setPortfolioData(data || { account: {}, positions: [], summary: {} });
      }
    } catch (err) {
      console.error("Failed to load portfolio data:", err);
    }

    try {
      const perfRes = await fetch(`${getApiBase()}/api/performance`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (perfRes.ok) {
        const perfJson = await perfRes.json();
        if (perfJson && !perfJson.error) setPerf(perfJson);
      }
    } catch (err) {
      console.debug("Failed to load performance:", err);
    }

    try {
      const fillsRes = await fetch(`${getApiBase()}/api/operator/recent-fills`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (fillsRes.ok) {
        const fillsJson = await fillsRes.json();
        setTradeHistory(fillsJson.fills || []);
      }
    } catch (err) {
      console.debug("Failed to load recent fills:", err);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    fetchPortfolio();
    const timer = setInterval(fetchPortfolio, 15000);
    return () => clearInterval(timer);
  }, [fetchPortfolio]);

  const positions = portfolioData.positions || [];
  
  const filteredPositions = useMemo(() => {
    if (regionFilter === 'ALL') return positions;
    return positions.filter(p => p.region === regionFilter);
  }, [positions, regionFilter]);

  const pieChartData = useMemo(() => {
    const summary = portfolioData.summary?.regional_allocation_usd || {};
    return Object.keys(summary).map(reg => ({
      name: reg === 'Kenya/Africa' ? '🇰🇪 Kenya & Africa' : reg === 'US' ? '🇺🇸 US Equities' : '🪙 Crypto',
      value: Math.round(summary[reg] || 0),
      rawRegion: reg
    })).filter(d => d.value > 0);
  }, [portfolioData]);

  // Cash held by the NSE paper account, in KES (null when it is off).
  const nseCashKES = portfolioData.nse_account?.cash_kes ?? null;
  // The core-satellite split of the US account ({enabled: false} when off).
  const core = portfolioData.core || null;

  // Aggregate total net worth in USD: US cash, NSE paper cash and every
  // holding, KES converted at KES_FX_RATE.
  const totalNetWorthUSD = useMemo(() => {
    const usdCash = portfolioData.account?.cash || 0;
    const positionsUSD = positions.reduce((sum, p) => {
      const val = p.market_value || 0;
      return sum + (p.currency === 'KES' ? val / KES_FX_RATE : val);
    }, 0);
    return usdCash + (nseCashKES || 0) / KES_FX_RATE + positionsUSD;
  }, [portfolioData, positions, KES_FX_RATE, nseCashKES]);

  const totalNetWorthKES = useMemo(() => totalNetWorthUSD * KES_FX_RATE, [totalNetWorthUSD, KES_FX_RATE]);

  const totalUnrealizedPLUSD = useMemo(() => {
    return positions.reduce((acc, p) => {
      const pl = p.unrealized_pl || 0;
      const val = p.currency === 'USD' ? pl : pl / KES_FX_RATE;
      return acc + val;
    }, 0);
  }, [positions, KES_FX_RATE]);

  // Sector Exposure Breakdown
  const sectorBreakdown = useMemo(() => {
    const sectors = {};
    let totalVal = 0;
    positions.forEach(p => {
      const sec = p.region === 'Crypto' ? 'Cryptocurrency' : p.region === 'Kenya/Africa' ? 'Kenyan Equities' : 'US Technology & ETFs';
      const val = p.currency === 'KES' ? (p.market_value || 0) / KES_FX_RATE : (p.market_value || 0);
      sectors[sec] = (sectors[sec] || 0) + val;
      totalVal += val;
    });
    return Object.entries(sectors).map(([name, val]) => ({
      name,
      val: Math.round(val),
      pct: totalVal > 0 ? Math.round((val / totalVal) * 100) : 0
    }));
  }, [positions, KES_FX_RATE]);

  // Real account equity over time, from the analytics engine. This used to be
  // generated by multiplying today's balance by fixed factors (0.96, 0.98,
  // 1.01 ...) chosen to show the agent beating a benchmark, whatever the
  // agent had actually done. There is no benchmark series in the backend, so
  // none is drawn.
  const equityCurve = useMemo(() => (perf?.portfolio_chart || [])
    .filter(pt => pt && typeof pt.value === 'number')
    .map(pt => ({ time: (pt.timestamp || '').slice(0, 10), value: Math.round(pt.value) })),
  [perf]);

  const equityReturnPct = useMemo(() => {
    if (equityCurve.length < 2 || !equityCurve[0].value) return null;
    const first = equityCurve[0].value, last = equityCurve[equityCurve.length - 1].value;
    return ((last - first) / first) * 100;
  }, [equityCurve]);

  const metrics = perf?.metrics || {};
  const closedTrades = metrics.total_trades || 0;
  const ratiosMeaningful = closedTrades >= MIN_TRADES_FOR_RATIOS;
  const fmtPct = (x, digits = 2) => `${x >= 0 ? '+' : ''}${x.toFixed(digits)}%`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Dashboard Top Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '24px', fontWeight: '800', margin: 0, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Globe size={24} color={theme.colors.primary} /> Unified Multi-Region Portfolio & Market Intelligence
          </h2>
          <p style={{ fontSize: '13px', color: theme.colors.textSecondary, margin: '4px 0 0 0' }}>
            Real-time multi-asset wealth aggregation across US Equities, Crypto, and Kenyan / African Markets
          </p>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <span style={{ backgroundColor: 'rgba(59, 130, 246, 0.12)', color: theme.colors.accent, border: `1px solid ${theme.colors.accent}40`, padding: '6px 12px', borderRadius: '8px', fontSize: '11px', fontWeight: 800 }}>
            FX: 1 USD = {KES_FX_RATE.toFixed(2)} KES
            <span style={{ fontWeight: 600, color: theme.colors.textMuted, marginLeft: '6px' }}>
              {fxSourceNote(fx)}
            </span>
          </span>
          <button 
            onClick={fetchPortfolio}
            style={{
              ...glassCard,
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '10px 16px', borderRadius: '10px',
              color: '#fff', fontSize: '13px', fontWeight: '700',
              cursor: 'pointer', border: `1px solid ${theme.colors.border}`
            }}
          >
            <RefreshCw size={14} className={loading ? "spin" : ""} /> Refresh
          </button>
        </div>
      </div>

      {/* Top Wealth & Risk HUD Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
        <div style={{ ...glassCard, padding: '20px', borderLeft: `4px solid ${theme.colors.primary}` }}>
          <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase' }}>Net Portfolio Equity</span>
          <h3 style={{ fontSize: '24px', fontWeight: '800', color: '#fff', margin: '6px 0 0 0' }}>
            ${totalNetWorthUSD.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </h3>
          <span style={{ fontSize: '11px', color: theme.colors.textSecondary, display: 'block', marginTop: '4px' }}>
            ≈ KES {totalNetWorthKES.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>

        <div style={{ ...glassCard, padding: '20px', borderLeft: `4px solid ${theme.colors.secondary}` }}>
          <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase' }}>Liquid Cash Reserves</span>
          <h3 style={{ fontSize: '24px', fontWeight: '800', color: '#fff', margin: '6px 0 0 0' }}>
            ${(portfolioData.account?.cash || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </h3>
          <span style={{ fontSize: '11px', color: theme.colors.primary, display: 'block', marginTop: '4px' }}>
            {nseCashKES === null ? 'USD Primary Account'
              : `US account · plus KES ${nseCashKES.toLocaleString(undefined, { maximumFractionDigits: 0 })} in the NSE paper account`}
          </span>
        </div>

        {core?.enabled && (
          <div style={{ ...glassCard, padding: '20px', borderLeft: `4px solid ${theme.colors.accent}` }}>
            {/* The core-satellite split (src/agent/core_portfolio.py): index
                funds held for the long run, and the share the strategies trade. */}
            <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase' }}>Core / Active Split</span>
            <h3 style={{ fontSize: '24px', fontWeight: '800', color: '#fff', margin: '6px 0 0 0' }}>
              {Math.round((1 - core.active_share) * 100)}% / {Math.round(core.active_share * 100)}%
            </h3>
            <span style={{ fontSize: '11px', color: theme.colors.textSecondary, display: 'block', marginTop: '4px' }}>
              Core {(core.symbols || []).join(', ')}: ${(core.value_usd || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })} of ${(core.target_usd || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })} target
              {core.last_rebalance ? ` · rebalanced ${core.last_rebalance}` : ' · first rebalance at the next US session'}
            </span>
          </div>
        )}

        <div style={{ ...glassCard, padding: '20px', borderLeft: `4px solid ${totalUnrealizedPLUSD >= 0 ? theme.colors.primary : theme.colors.danger}` }}>
          <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase' }}>Unrealized Floating P&L</span>
          <h3 style={{ fontSize: '24px', fontWeight: '800', color: totalUnrealizedPLUSD >= 0 ? theme.colors.primary : theme.colors.danger, margin: '6px 0 0 0' }}>
            {totalUnrealizedPLUSD >= 0 ? '+' : ''}${totalUnrealizedPLUSD.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </h3>
          <span style={{ fontSize: '11px', color: theme.colors.textMuted, display: 'block', marginTop: '4px' }}>Across {positions.length} Active Positions</span>
        </div>

        <div style={{ ...glassCard, padding: '20px', borderLeft: `4px solid ${theme.colors.warning}` }}>
          {/* Was hardcoded "VaR 1.82%, Sharpe 1.94", shown identically with zero
              trades. Now the analytics engine's real figures, and only once
              there are enough closed trades for them to mean anything. */}
          <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase' }}>Risk-Adjusted Performance</span>
          {ratiosMeaningful ? (
            <>
              <h3 style={{ fontSize: '24px', fontWeight: '800', color: theme.colors.warning, margin: '6px 0 0 0' }}>
                Sharpe {Number(metrics.sharpe_ratio || 0).toFixed(2)}
              </h3>
              <span style={{ fontSize: '11px', color: theme.colors.textMuted, display: 'block', marginTop: '4px' }}>
                Max drawdown {fmtPct(-Math.abs((metrics.max_drawdown || 0) * 100))} · {closedTrades} closed trades
              </span>
            </>
          ) : (
            <>
              <h3 style={{ fontSize: '18px', fontWeight: '800', color: theme.colors.textMuted, margin: '6px 0 0 0' }}>
                Not enough data yet
              </h3>
              <span style={{ fontSize: '11px', color: theme.colors.textMuted, display: 'block', marginTop: '4px' }}>
                {closedTrades} of {MIN_TRADES_FOR_RATIOS} closed trades needed
              </span>
            </>
          )}
        </div>
      </div>

      {/* Agent track record. This panel used to be three paragraphs of fixed
          text and a "BULLISH EXPANSION" badge: an 8.4% dividend yield
          projecting +14.2% growth, and a claim that the strategy was
          "outperforming Buy & Hold by +2.8% this month", displayed whether
          or not the agent had ever traded. Nothing in the backend computes a
          market regime or those projections, so they are gone; what remains
          is what the agent has actually done. */}
      <div style={{ ...glassCard, padding: '24px', border: `1px solid ${theme.colors.border}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <Lightbulb size={22} color={theme.colors.primary} />
          <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '800', color: '#fff', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Agent Track Record
          </h3>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          {[
            ['Closed trades', perf ? String(closedTrades) : '—'],
            // win_rate is a 0-1 fraction; the Command Desk renders it the same way.
            ['Win rate', closedTrades > 0 ? `${((metrics.win_rate || 0) * 100).toFixed(1)}%` : '—'],
            ['Equity change (period)', equityReturnPct === null ? '—' : fmtPct(equityReturnPct)],
          ].map(([label, value]) => (
            <div key={label} style={{ backgroundColor: 'rgba(0,0,0,0.25)', padding: '14px', borderRadius: '10px', border: `1px solid ${theme.colors.border}40` }}>
              <div style={{ fontSize: '11px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase', marginBottom: '6px' }}>{label}</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#fff' }}>{value}</div>
            </div>
          ))}
        </div>
        {!ratiosMeaningful && (
          <p style={{ margin: '14px 0 0 0', fontSize: '12px', color: theme.colors.textMuted, lineHeight: '1.5' }}>
            Too few closed trades to judge whether the strategy has an edge. Performance ratios appear here after {MIN_TRADES_FOR_RATIOS} closed trades.
          </p>
        )}
      </div>

      {/* Visual Analytics: Allocation Donut & Capital Growth Curve */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Capital Growth Curve */}
        <div style={{ ...glassCard, padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '800', color: theme.colors.textSecondary, textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={16} color={theme.colors.primary} /> Account Equity
            </h3>
            <span style={{ fontSize: '11px', color: equityReturnPct === null ? theme.colors.textMuted : (equityReturnPct >= 0 ? theme.colors.primary : theme.colors.danger), fontWeight: 700 }}>
              {equityReturnPct === null ? 'Not enough history yet' : `${fmtPct(equityReturnPct)} over period`}
            </span>
          </div>

          <div style={{ height: '220px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equityCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke={`${theme.colors.border}40`} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: theme.colors.textMuted }} />
                <YAxis tick={{ fontSize: 10, fill: theme.colors.textMuted }} width={55} />
                <Tooltip contentStyle={{ background: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, borderRadius: '8px', fontSize: '12px' }} />
                <Area type="monotone" dataKey="value" stroke={theme.colors.primary} fill="rgba(16,185,129,0.15)" strokeWidth={2} name="Account equity ($)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Regional Allocation Donut */}
        <div style={{ ...glassCard, padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '800', color: theme.colors.textSecondary, textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <PieChart size={16} color={theme.colors.accent} /> Multi-Region Capital Allocation
            </h3>
          </div>

          {pieChartData.length > 0 ? (
            <div style={{ height: '220px', display: 'flex', alignItems: 'center' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieChartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {pieChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={REGION_COLORS[entry.rawRegion] || theme.colors.primary} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ background: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, borderRadius: '8px', fontSize: '12px' }}
                    formatter={(v) => `$${v.toLocaleString()}`}
                  />
                  <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ padding: '60px 0', textAlign: 'center', color: theme.colors.textMuted, fontSize: '13px' }}>
              No active holdings found across US, Crypto, or Kenya regions.
            </div>
          )}
        </div>
      </div>

      {/* Sector Exposure Progress Bars */}
      <div style={{ ...glassCard, padding: '20px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '800', color: theme.colors.textSecondary, textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Layers3 size={16} color={theme.colors.primary} /> Sector Risk Exposure & Concentration
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {sectorBreakdown.map((s, i) => (
            <div key={i}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: '700', marginBottom: '4px' }}>
                <span style={{ color: '#fff' }}>{s.name}</span>
                <span style={{ color: theme.colors.textSecondary }}>${s.val.toLocaleString()} ({s.pct}%)</span>
              </div>
              <div style={{ height: '8px', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${s.pct}%`, height: '100%', backgroundColor: i === 0 ? theme.colors.primary : i === 1 ? theme.colors.accent : theme.colors.warning, borderRadius: '4px' }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Filterable High-Density Holdings Table */}
      <div style={{ ...glassCard, padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Layers size={18} color={theme.colors.primary} />
            <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '800', color: '#fff' }}>
              Multi-Region Active Holdings ({filteredPositions.length})
            </h3>
          </div>

          <div style={{ display: 'flex', gap: '8px', backgroundColor: 'rgba(0,0,0,0.3)', padding: '4px', borderRadius: '8px', border: `1px solid ${theme.colors.border}` }}>
            {['ALL', 'US', 'Crypto', 'Kenya/Africa'].map(reg => (
              <button
                key={reg}
                onClick={() => setRegionFilter(reg)}
                style={{
                  backgroundColor: regionFilter === reg ? theme.colors.primary : 'transparent',
                  color: regionFilter === reg ? '#000' : theme.colors.textSecondary,
                  border: 'none', padding: '6px 12px', borderRadius: '6px',
                  fontSize: '11px', fontWeight: '800', cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                {reg === 'ALL' ? 'ALL REGIONS' : reg === 'Kenya/Africa' ? '🇰🇪 KENYA' : reg === 'US' ? '🇺🇸 US' : '🪙 CRYPTO'}
              </button>
            ))}
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textMuted, fontSize: '11px', textTransform: 'uppercase' }}>
                <th style={{ padding: '12px 0' }}>Asset Symbol</th>
                <th style={{ padding: '12px 0' }}>Market Region</th>
                <th style={{ padding: '12px 0' }}>Quantity</th>
                <th style={{ padding: '12px 0' }}>Avg Entry</th>
                <th style={{ padding: '12px 0' }}>Current Price</th>
                <th style={{ padding: '12px 0' }}>Market Value</th>
                <th style={{ padding: '12px 0' }}>Unrealized P&L</th>
                <th style={{ padding: '12px 0', textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredPositions.length > 0 ? (
                filteredPositions.map((pos, idx) => {
                  const isPositive = (pos.unrealized_pl || 0) >= 0;
                  const isKES = pos.currency === 'KES';
                  return (
                    <tr key={idx} style={{ borderBottom: `1px solid ${theme.colors.border}30` }}>
                      <td style={{ padding: '14px 0', fontWeight: '800', color: '#fff' }}>
                        <span style={{ marginRight: '6px' }}>{pos.flag}</span>
                        {pos.symbol}
                      </td>
                      <td style={{ padding: '14px 0' }}>
                        <span style={{ 
                          backgroundColor: `${REGION_COLORS[pos.region] || theme.colors.primary}15`, 
                          color: REGION_COLORS[pos.region] || theme.colors.primary, 
                          padding: '3px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: '800' 
                        }}>
                          {pos.market_name}
                        </span>
                      </td>
                      <td style={{ padding: '14px 0', fontWeight: '600' }}>{pos.quantity}</td>
                      <td style={{ padding: '14px 0', color: theme.colors.textSecondary }}>
                        {isKES ? `${(pos.avg_entry_price || 0).toLocaleString()} KES` : `$${(pos.avg_entry_price || 0).toLocaleString()}`}
                      </td>
                      <td style={{ padding: '14px 0', fontWeight: '700', color: '#fff' }}>
                        {isKES ? `${(pos.current_price || 0).toLocaleString()} KES` : `$${(pos.current_price || 0).toLocaleString()}`}
                      </td>
                      <td style={{ padding: '14px 0', fontWeight: '700' }}>
                        {isKES ? `${(pos.market_value || 0).toLocaleString()} KES` : `$${(pos.market_value || 0).toLocaleString()}`}
                        <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>
                          ≈ ${(isKES ? (pos.market_value || 0) / KES_FX_RATE : (pos.market_value || 0)).toFixed(2)}
                        </div>
                      </td>
                      <td style={{ padding: '14px 0', color: isPositive ? theme.colors.primary : theme.colors.danger, fontWeight: '800' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                          {isPositive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                          {isKES ? `${isPositive ? '+' : ''}${(pos.unrealized_pl || 0).toLocaleString()} KES` : `${isPositive ? '+' : ''}$${(pos.unrealized_pl || 0).toLocaleString()}`}
                          <span style={{ fontSize: '11px', opacity: 0.8 }}>({(pos.unrealized_pl_pct || 0).toFixed(1)}%)</span>
                        </div>
                      </td>
                      <td style={{ padding: '14px 0', textAlign: 'right' }}>
                        <button
                          onClick={() => onDrill && onDrill(pos.symbol)}
                          style={{
                            backgroundColor: 'rgba(255,255,255,0.05)',
                            border: `1px solid ${theme.colors.border}`,
                            color: theme.colors.textSecondary,
                            padding: '4px 10px', borderRadius: '6px',
                            fontSize: '11px', fontWeight: 800, cursor: 'pointer'
                          }}
                        >
                          INSPECT
                        </button>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan="8" style={{ padding: '40px 0', textAlign: 'center', color: theme.colors.textMuted }}>
                    No positions found for the selected region filter ({regionFilter}).
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Live Trade History Feed */}
      <div style={{ ...glassCard, padding: '24px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '800', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={18} color={theme.colors.primary} /> Live Multi-Region Trade Execution Feed
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {tradeHistory.length > 0 ? (
            tradeHistory.slice(0, 5).map((fill, idx) => (
              <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: `1px solid ${theme.colors.border}30` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ 
                    backgroundColor: fill.side === 'buy' ? 'rgba(16,185,129,0.15)' : 'rgba(244,63,94,0.15)', 
                    color: fill.side === 'buy' ? theme.colors.primary : theme.colors.danger, 
                    padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '800' 
                  }}>
                    {fill.side?.toUpperCase()}
                  </span>
                  <div>
                    <span style={{ fontWeight: '800', color: '#fff', fontSize: '13px' }}>{fill.symbol}</span>
                    <div style={{ fontSize: '11px', color: theme.colors.textMuted }}>
                      Qty: {fill.fill_quantity || fill.quantity} @ {fill.fill_price ? `$${fill.fill_price}` : 'Market'} · {fill.resolved_by || 'auto_paper_trader'}
                    </div>
                  </div>
                </div>
                <span style={{ fontSize: '11px', color: theme.colors.textMuted }}>{fill.created_at ? new Date(fill.created_at).toLocaleTimeString() : 'Recent'}</span>
              </div>
            ))
          ) : (
            <div style={{ color: theme.colors.textMuted, fontSize: '13px', textAlign: 'center', padding: '20px 0' }}>
              No recent trade fills recorded yet. Systems operating autonomously.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
