import React, { useState, useEffect, useCallback } from 'react';
import { 
  Trophy, Award, TrendingUp, TrendingDown, Shield, CheckCircle, XCircle, 
  AlertTriangle, RefreshCw, Cpu, Layers, GitBranch, ArrowUpRight, ArrowDownRight, Zap, Target
} from 'lucide-react';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

const SectionHeader = ({ title, icon: Icon, subtitle }) => (
  <div style={{ marginBottom: '20px' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      <Icon size={20} color={theme.colors.primary} />
      <h3 style={{ fontSize: '18px', fontWeight: '700', margin: 0, color: '#fff' }}>{title}</h3>
      <div style={{ height: '1px', flex: 1, background: `linear-gradient(90deg, ${theme.colors.border}, transparent)` }} />
    </div>
    {subtitle && (
      <p style={{ margin: '6px 0 0 30px', fontSize: '12px', color: theme.colors.textMuted }}>{subtitle}</p>
    )}
  </div>
);

const HUDCard = ({ title, value, subValue, icon: Icon, color, badge }) => (
  <div style={{ ...glassCard, flex: 1, minWidth: '200px', position: 'relative', overflow: 'hidden' }}>
    <div style={{ position: 'absolute', right: '-10px', bottom: '-10px', opacity: 0.05 }}>
      <Icon size={90} color={color} />
    </div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
      <p style={{ fontSize: '11px', color: theme.colors.textSecondary, margin: 0, fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.8px' }}>{title}</p>
      {badge ? (
        <span style={{ fontSize: '10px', fontWeight: 800, padding: '2px 6px', borderRadius: '4px', backgroundColor: `${color}20`, color }}>{badge}</span>
      ) : (
        <div style={{ backgroundColor: `${color}15`, padding: '5px', borderRadius: '6px', color }}>
          <Icon size={16} />
        </div>
      )}
    </div>
    <div style={{ fontSize: '22px', fontWeight: '800', color: '#fff', marginBottom: '4px' }}>{value}</div>
    {subValue && <div style={{ fontSize: '11px', color: theme.colors.textMuted }}>{subValue}</div>}
  </div>
);

const GrowthView = ({ onDrill }) => {
  const [growthData, setGrowthData] = useState(null);
  const [rawReport, setRawReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const token = typeof window !== 'undefined' ? localStorage.getItem('trading_token') : null;

  const loadGrowthData = useCallback(async () => {
    try {
      const headers = { Authorization: `Bearer ${token}` };
      const [shadowRes, reportRes] = await Promise.all([
        fetch(`${getApiBase()}/api/shadow-strategies`, { headers }),
        fetch(`${getApiBase()}/api/growth-report`, { headers })
      ]);

      if (shadowRes.ok) {
        const sJson = await shadowRes.json();
        setGrowthData(sJson);
      }
      if (reportRes.ok) {
        const rJson = await reportRes.json();
        setRawReport(rJson);
      }
      setError(null);
    } catch (e) {
      console.error('Failed to load growth & learning data:', e);
      setError(e.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [token]);

  useEffect(() => {
    loadGrowthData();
    const id = setInterval(loadGrowthData, 30000);
    return () => clearInterval(id);
  }, [loadGrowthData]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadGrowthData();
  };

  const champion = growthData?.champion || { ensemble_method: 'adaptive_confidence', weights: {} };
  const challengers = growthData?.challengers || [];
  const primaryChallenger = challengers[0] || null;
  const learning = growthData?.learning_cycle || {};
  const attributions = learning.active_attributions || rawReport?.attribution || {};
  const strategyWeights = learning.strategy_weights || champion.weights || {};
  const removals = learning.removal_proposals || rawReport?.removals || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Top Controls & Cycle Timestamp */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h2 style={{ margin: 0, fontSize: '24px', fontWeight: 800, color: '#fff', letterSpacing: '0.5px' }}>
              CONTINUOUS GROWTH & REINFORCEMENT ARENA
            </h2>
            <span style={{ backgroundColor: 'rgba(16, 185, 129, 0.15)', color: theme.colors.primary, padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 800 }}>
              CLOSED-LOOP ACTIVE
            </span>
          </div>
          <p style={{ margin: '6px 0 0 0', fontSize: '13px', color: theme.colors.textMuted }}>
            Autonomous self-assessment, Champion vs Challenger shadow tournaments, and dynamic weight drift adaptation.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              backgroundColor: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border}`,
              color: theme.colors.textSecondary, padding: '8px 16px', borderRadius: '8px',
              fontSize: '12px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s'
            }}
          >
            <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
            {refreshing ? 'Refreshing...' : 'Refresh Arena'}
          </button>
        </div>
      </div>

      {/* Overview Stat HUDs */}
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
        <HUDCard
          title="Active Champion"
          value={champion.ensemble_method?.replace(/_/g, ' ').toUpperCase() || 'ADAPTIVE'}
          subValue="Live production routing"
          icon={Trophy}
          color={theme.colors.primary}
          badge="PRODUCTION"
        />
        <HUDCard
          title="Top Shadow Challenger"
          value={primaryChallenger?.ensemble_method?.replace(/_/g, ' ').toUpperCase() || 'REGIME ROUTED'}
          subValue={primaryChallenger?.status === 'cleared_for_promotion' ? 'Cleared for promotion' : 'Under evaluation'}
          icon={GitBranch}
          color={primaryChallenger?.status === 'cleared_for_promotion' ? theme.colors.primary : theme.colors.accent}
          badge={primaryChallenger?.status === 'cleared_for_promotion' ? 'CANARY READY' : 'SHADOW'}
        />
        <HUDCard
          title="Learning Cycles"
          value={learning.fills_analyzed ? `${learning.fills_analyzed} Fills` : 'Initialized'}
          subValue={learning.last_run ? `Last run: ${new Date(learning.last_run).toLocaleDateString()}` : 'Nightly cycle scheduled'}
          icon={Cpu}
          color={theme.colors.secondary}
        />
        <HUDCard
          title="Tracked Universe"
          value={`${learning.tracked_symbols_count || 12} Assets`}
          subValue={removals.length > 0 ? `${removals.length} flagged for pruning` : 'Zero pruning alerts'}
          icon={Target}
          color={removals.length > 0 ? theme.colors.warning : theme.colors.primary}
          badge={removals.length > 0 ? 'PRUNE REVIEW' : 'HEALTHY'}
        />
      </div>

      {/* Champion vs Challenger Arena Section */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Champion Card */}
        <div style={{ ...glassCard, border: `1px solid ${theme.colors.primary}40`, position: 'relative' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ backgroundColor: 'rgba(16, 185, 129, 0.2)', padding: '8px', borderRadius: '8px' }}>
                <Trophy size={20} color={theme.colors.primary} />
              </div>
              <div>
                <h4 style={{ margin: 0, fontSize: '16px', fontWeight: 800, color: '#fff' }}>
                  CHAMPION: {champion.ensemble_method?.replace(/_/g, ' ').toUpperCase()}
                </h4>
                <span style={{ fontSize: '11px', color: theme.colors.primary, fontWeight: 700 }}>
                  Active Live Router
                </span>
              </div>
            </div>
            <span style={{ backgroundColor: 'rgba(16, 185, 129, 0.15)', color: theme.colors.primary, border: `1px solid ${theme.colors.primary}40`, padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 800 }}>
              100% TRAFFIC
            </span>
          </div>

          <p style={{ fontSize: '12px', color: theme.colors.textSecondary, lineHeight: '1.5', margin: '0 0 16px 0' }}>
            The reigning decision architecture executes multi-strategy weighted consensus with continuous Bayes dynamic weight adjustments.
          </p>

          <div style={{ backgroundColor: 'rgba(0,0,0,0.25)', borderRadius: '10px', padding: '14px', border: `1px solid ${theme.colors.border}40` }}>
            <div style={{ fontSize: '11px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase', marginBottom: '10px' }}>
              Allocated Strategy Weights in Champion
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {Object.keys(strategyWeights).length === 0 && (
                <div style={{ fontSize: '12px', color: theme.colors.textMuted }}>Baseline strategy weights initialized.</div>
              )}
              {Object.entries(strategyWeights).map(([strat, wt]) => (
                <div key={strat} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px' }}>
                  <span style={{ color: theme.colors.textSecondary, textTransform: 'capitalize' }}>
                    {strat.replace(/_/g, ' ')}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ width: '80px', height: '6px', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ width: `${Math.min(100, (Number(wt) / 2.0) * 100)}%`, height: '100%', backgroundColor: theme.colors.primary }} />
                    </div>
                    <span style={{ fontWeight: 800, color: '#fff', minWidth: '35px', textAlign: 'right' }}>
                      {Number(wt).toFixed(2)}x
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Challenger Card */}
        <div style={{ ...glassCard, border: `1px solid ${primaryChallenger?.status === 'cleared_for_promotion' ? theme.colors.primary : theme.colors.accent}40`, position: 'relative' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ backgroundColor: 'rgba(139, 92, 246, 0.2)', padding: '8px', borderRadius: '8px' }}>
                <GitBranch size={20} color={theme.colors.accent} />
              </div>
              <div>
                <h4 style={{ margin: 0, fontSize: '16px', fontWeight: 800, color: '#fff' }}>
                  CHALLENGER: {primaryChallenger?.ensemble_method?.replace(/_/g, ' ').toUpperCase() || 'REGIME ROUTED'}
                </h4>
                <span style={{ fontSize: '11px', color: theme.colors.accent, fontWeight: 700 }}>
                  Shadow Paper Evaluation
                </span>
              </div>
            </div>
            <span style={{
              backgroundColor: primaryChallenger?.status === 'cleared_for_promotion' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
              color: primaryChallenger?.status === 'cleared_for_promotion' ? theme.colors.primary : theme.colors.warning,
              border: `1px solid ${primaryChallenger?.status === 'cleared_for_promotion' ? theme.colors.primary : theme.colors.warning}40`,
              padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 800
            }}>
              {primaryChallenger?.status === 'cleared_for_promotion' ? 'CLEARED FOR PROMOTION' : 'IN SHADOW TRIAL'}
            </span>
          </div>

          <p style={{ fontSize: '12px', color: theme.colors.textSecondary, lineHeight: '1.5', margin: '0 0 16px 0' }}>
            Routes market signals dynamically based on detected Hidden Markov regimes (Bull Volatile, Bull Trend, Mean-Reverting, Bear).
          </p>

          {/* Challenger Performance Metrics */}
          {primaryChallenger?.metrics ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '16px' }}>
              <div style={{ backgroundColor: 'rgba(0,0,0,0.25)', padding: '10px', borderRadius: '8px', border: `1px solid ${theme.colors.border}30` }}>
                <div style={{ fontSize: '10px', color: theme.colors.textMuted, fontWeight: 700 }}>OOS SHARPE</div>
                <div style={{ fontSize: '16px', fontWeight: 800, color: (primaryChallenger.metrics.sharpe || 0) >= 0 ? theme.colors.primary : theme.colors.danger, margin: '2px 0' }}>
                  {(primaryChallenger.metrics.sharpe || 0).toFixed(3)}
                </div>
                <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>Baseline: -0.098</div>
              </div>

              <div style={{ backgroundColor: 'rgba(0,0,0,0.25)', padding: '10px', borderRadius: '8px', border: `1px solid ${theme.colors.border}30` }}>
                <div style={{ fontSize: '10px', color: theme.colors.textMuted, fontWeight: 700 }}>OOS RETURN</div>
                <div style={{ fontSize: '16px', fontWeight: 800, color: theme.colors.primary, margin: '2px 0' }}>
                  +{((primaryChallenger.metrics.total_return || 0) * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>Drawdown: {((primaryChallenger.metrics.max_drawdown || 0) * 100).toFixed(1)}%</div>
              </div>

              <div style={{ backgroundColor: 'rgba(0,0,0,0.25)', padding: '10px', borderRadius: '8px', border: `1px solid ${theme.colors.border}30` }}>
                <div style={{ fontSize: '10px', color: theme.colors.textMuted, fontWeight: 700 }}>SAMPLE SIZE</div>
                <div style={{ fontSize: '16px', fontWeight: 800, color: '#fff', margin: '2px 0' }}>
                  {primaryChallenger.metrics.trades || 0}
                </div>
                <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>Win rate: {((primaryChallenger.metrics.win_rate || 0) * 100).toFixed(0)}%</div>
              </div>
            </div>
          ) : (
            <div style={{ padding: '14px', color: theme.colors.textMuted, fontSize: '12px' }}>Challenger backtest metrics pending cycle execution.</div>
          )}

          {/* Acceptance Gate Checklist */}
          {primaryChallenger?.acceptance_gate?.checks && (
            <div style={{ backgroundColor: 'rgba(0,0,0,0.25)', borderRadius: '10px', padding: '12px 14px', border: `1px solid ${theme.colors.border}40` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '11px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase' }}>
                  Promotion Acceptance Gate
                </span>
                <span style={{ fontSize: '10px', fontWeight: 800, color: primaryChallenger.acceptance_gate.passed ? theme.colors.primary : theme.colors.warning }}>
                  {primaryChallenger.acceptance_gate.passed ? 'ALL CHECKS PASSED' : 'GATED - IN TRIAL'}
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {Object.entries(primaryChallenger.acceptance_gate.checks).map(([gate, passed]) => (
                  <div key={gate} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px' }}>
                    <span style={{ color: theme.colors.textSecondary }}>{gate}</span>
                    <span style={{
                      display: 'flex', alignItems: 'center', gap: '4px',
                      color: passed ? theme.colors.primary : theme.colors.danger, fontWeight: 700
                    }}>
                      {passed ? <CheckCircle size={12} /> : <XCircle size={12} />}
                      {passed ? 'PASS' : 'FAIL'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Dynamic Weight Drift Adaptation Table */}
      <div style={glassCard}>
        <SectionHeader
          title="Dynamic Strategy Weight Adaptation"
          icon={Layers}
          subtitle="Closed-loop performance attribution rewards profitable sub-strategies (+10%) and throttles underperformers (-10%) into config overlay."
        />
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, textAlign: 'left' }}>
                <th style={{ padding: '10px 0' }}>SUB-STRATEGY</th>
                <th style={{ padding: '10px 0' }}>CURRENT WEIGHT</th>
                <th style={{ padding: '10px 0' }}>STATUS / DRIFT</th>
                <th style={{ padding: '10px 0' }}>REALIZED P&L</th>
                <th style={{ padding: '10px 0' }}>WIN RATE</th>
                <th style={{ padding: '10px 0' }}>CLOSED TRADES</th>
                <th style={{ padding: '10px 0', textAlign: 'right' }}>LAST EVALUATED</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(strategyWeights).length > 0 ? (
                Object.entries(strategyWeights).map(([stratName, weight]) => {
                  const stat = attributions[stratName] || {};
                  const pnl = stat.realized_pnl || 0;
                  const winRate = stat.win_rate || 0;
                  const closed = stat.closed_trades || 0;
                  const numWeight = Number(weight);
                  const isBoosted = numWeight > 1.0;
                  const isThrottled = numWeight < 1.0;

                  return (
                    <tr key={stratName} style={{ borderBottom: `1px solid ${theme.colors.border}30` }}>
                      <td style={{ padding: '12px 0', fontWeight: 'bold', textTransform: 'capitalize' }}>
                        {stratName.replace(/_/g, ' ')}
                      </td>
                      <td style={{ padding: '12px 0' }}>
                        <span style={{ fontSize: '14px', fontWeight: 800, color: '#fff' }}>
                          {numWeight.toFixed(2)}x
                        </span>
                      </td>
                      <td style={{ padding: '12px 0' }}>
                        {isBoosted && (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', backgroundColor: 'rgba(16,185,129,0.15)', color: theme.colors.primary, padding: '3px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 800 }}>
                            <ArrowUpRight size={13} /> +10% REWARD
                          </span>
                        )}
                        {isThrottled && (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', backgroundColor: 'rgba(244,63,94,0.15)', color: theme.colors.danger, padding: '3px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 800 }}>
                            <ArrowDownRight size={13} /> -10% THROTTLE
                          </span>
                        )}
                        {!isBoosted && !isThrottled && (
                          <span style={{ backgroundColor: 'rgba(255,255,255,0.05)', color: theme.colors.textMuted, padding: '3px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 700 }}>
                            1.00x BASELINE
                          </span>
                        )}
                      </td>
                      <td style={{ padding: '12px 0', fontWeight: 700, color: pnl >= 0 ? theme.colors.primary : theme.colors.danger }}>
                        {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                      </td>
                      <td style={{ padding: '12px 0', color: theme.colors.textSecondary }}>
                        {(winRate * 100).toFixed(0)}%
                      </td>
                      <td style={{ padding: '12px 0', color: theme.colors.textSecondary }}>
                        {closed} positions
                      </td>
                      <td style={{ padding: '12px 0', textAlign: 'right', color: theme.colors.textMuted, fontSize: '12px' }}>
                        {learning.last_run ? new Date(learning.last_run).toLocaleTimeString() : 'Scheduled'}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan="7" style={{ padding: '30px 0', textAlign: 'center', color: theme.colors.textMuted }}>
                    No strategy weights recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Systematic Universe Pruning Diagnostics */}
      <div style={glassCard}>
        <SectionHeader
          title="Systematic Portfolio Pruning & Universe Health"
          icon={AlertTriangle}
          subtitle="Flags assets experiencing persistent consecutive losses, negative returns, or stale volatility regimes for automated removal."
        />
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, textAlign: 'left' }}>
                <th style={{ padding: '10px 0' }}>FLAGGED ASSET</th>
                <th style={{ padding: '10px 0' }}>REASON & EVIDENCE</th>
                <th style={{ padding: '10px 0' }}>CONSECUTIVE LOSSES</th>
                <th style={{ padding: '10px 0' }}>STATUS</th>
                <th style={{ padding: '10px 0', textAlign: 'right' }}>ACTION</th>
              </tr>
            </thead>
            <tbody>
              {removals.length > 0 ? (
                removals.map(r => (
                  <tr key={r.symbol} style={{ borderBottom: `1px solid ${theme.colors.border}30` }}>
                    <td style={{ padding: '12px 0' }}>
                      <span
                        onClick={() => onDrill && onDrill(r.symbol)}
                        title={`Open ${r.symbol} drill-down`}
                        style={{ fontWeight: 800, color: '#fff', cursor: 'pointer', textDecoration: 'underline' }}
                      >
                        {r.symbol}
                      </span>
                    </td>
                    <td style={{ padding: '12px 0', color: theme.colors.warning, fontWeight: 600 }}>
                      {r.reason}
                    </td>
                    <td style={{ padding: '12px 0', color: theme.colors.danger, fontWeight: 700 }}>
                      {r.consecutive_losses ? `${r.consecutive_losses} losses` : 'Evaluated'}
                    </td>
                    <td style={{ padding: '12px 0' }}>
                      <span style={{ backgroundColor: 'rgba(244,63,94,0.15)', color: theme.colors.danger, padding: '3px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 800 }}>
                        ESCALATED TO OPERATOR
                      </span>
                    </td>
                    <td style={{ padding: '12px 0', textAlign: 'right' }}>
                      <span style={{ fontSize: '12px', color: theme.colors.textMuted }}>
                        Review in Operator Queue
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" style={{ padding: '30px 0', textAlign: 'center', color: theme.colors.primary }}>
                    ✓ All tracked universe assets meet risk and profitability criteria. Zero pruning alerts.
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

export default GrowthView;
