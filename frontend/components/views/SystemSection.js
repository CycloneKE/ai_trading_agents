// Risk & System: risk limits and alerts, and the agent's service health
// and log, as two tabs of one section.
import { useEffect, useState } from 'react';
import { Shield, TrendingDown, Zap, Lock, Bell, Cpu, Activity, Terminal } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { apiGet, card, columns, SectionHeader, StatCard, HudRow, SubTabs, whenVisible } from '../ui';
import Scorecard from './Scorecard';

// One limit as a bar: how much of it has been used.
const LimitBar = ({ label, used, limit, note }) => {
  const share = limit > 0 ? Math.min(Math.max(used / limit, 0), 1) : 0;
  const color = share >= 1 ? theme.colors.danger : share >= 0.6 ? theme.colors.warning : theme.colors.primary;
  return (
    <div style={{ display: 'grid', gap: '6px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', fontSize: '13px' }}>
        <span style={{ color: theme.colors.text, fontWeight: 600 }}>{label}</span>
        <span style={{ color, fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>
          {(used * 100).toFixed(2)}% <span style={{ color: theme.colors.textMuted, fontWeight: 600 }}>of {(limit * 100).toFixed(0)}%</span>
        </span>
      </div>
      <div style={{ height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
        <div style={{ width: `${share * 100}%`, height: '100%', background: color, borderRadius: '3px', transition: 'width 0.3s ease' }} />
      </div>
      {note && <div style={{ fontSize: '11px', color: theme.colors.textMuted }}>{note}</div>}
    </div>
  );
};

// The account's daily loss and drawdown limits (src/agent/loss_limits.py):
// while either is reached the agent buys nothing and can still sell.
const LossLimitsCard = ({ mobile }) => {
  const [limits, setLimits] = useState(null);
  useEffect(() => {
    let alive = true;
    const load = () => apiGet('/api/risk/loss-limits')
      .then(({ ok, json }) => { if (alive) setLimits(ok ? json : { enabled: false }); })
      .catch(() => {});
    load();
    const id = setInterval(whenVisible(load), 30000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  const money = (v) => (typeof v === 'number' ? `$${Math.round(v).toLocaleString()}` : '—');
  return (
    <div style={card(mobile)}>
      <SectionHeader title="Account loss limits" icon={Lock}
        badge={limits?.enabled ? (limits.buys_blocked ? 'BUYING PAUSED' : 'WITHIN LIMITS') : undefined} />
      {!limits && <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>Loading…</div>}
      {limits && !limits.enabled && (
        <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>The loss limits are not running on this agent.</div>
      )}
      {limits?.enabled && (
        <div style={{ display: 'grid', gap: '16px' }}>
          {limits.buys_blocked && (
            <div style={{
              padding: '10px 12px', borderRadius: '10px', fontSize: '12px', fontWeight: 600,
              background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.3)', color: '#fda4af',
            }}>
              A limit has been reached, so the agent is not opening new positions. It can still sell.
            </div>
          )}
          <LimitBar label="Lost today" used={Math.max(limits.daily_loss || 0, 0)} limit={limits.max_daily_loss || 0}
            note={`From ${money(limits.day_start_equity)} at the start of the day (UTC); it resets each day.`} />
          <LimitBar label="Below the high" used={Math.max(limits.drawdown || 0, 0)} limit={limits.max_drawdown || 0}
            note={`From the highest value, ${money(limits.peak_equity)}${limits.peak_at ? ` on ${String(limits.peak_at).slice(0, 10)}` : ''}. An operator resets it.`} />
        </div>
      )}
    </div>
  );
};

const RiskView = ({ data, mobile }) => {
  const rm = data.riskMetrics || {};
  const metrics = rm.current_metrics || {};
  const shown = (v, f) => (typeof v === 'number' ? f(v) : '—');
  return (
    <div style={{ display: 'grid', gap: mobile ? '14px' : '20px' }}>
      <HudRow mobile={mobile}>
        <StatCard label="Value at risk (1 day)" value={shown(metrics.portfolio_var, (v) => `${(v * 100).toFixed(2)}%`)} icon={Shield} color={theme.colors.warning} />
        <StatCard label="Deepest fall" value={shown(metrics.max_drawdown ?? data.performance?.max_drawdown, (v) => `${(Math.abs(v) * 100).toFixed(2)}%`)} icon={TrendingDown} color={theme.colors.danger} />
        <StatCard label="Leverage" value={shown(metrics.leverage, (v) => `${v.toFixed(2)}x`)} icon={Zap} color={theme.colors.primary} />
      </HudRow>
      <div style={columns('1fr 1fr', mobile)}>
        <LossLimitsCard mobile={mobile} />
        <div style={card(mobile)}>
          <SectionHeader title="Recent alerts" icon={Bell} />
          {(rm.alerts?.recent || []).length > 0 ? (
            <div style={{ display: 'grid', gap: '8px' }}>
              {(rm.alerts?.recent || []).map((a, i) => (
                <div key={i} style={{
                  padding: '12px 14px', borderRadius: '10px',
                  background: 'rgba(244, 63, 94, 0.05)',
                  border: '1px solid rgba(244, 63, 94, 0.2)',
                  borderLeft: `4px solid ${theme.colors.danger}`,
                }}>
                  <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>{new Date(a.timestamp).toLocaleString()}</div>
                  <div style={{ fontWeight: 800, fontSize: '13px', color: '#fff', marginTop: '2px' }}>{a.limit_name} limit reached</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ padding: '32px 16px', textAlign: 'center', color: theme.colors.textMuted, fontSize: '13px' }}>
              No risk alerts
            </div>
          )}
        </div>
      </div>
      {(rm.risk_limits || []).length > 0 && (
        <div style={card(mobile)}>
          <SectionHeader title="Position risk limits" icon={Lock} />
          <div style={{ display: 'grid', gap: '8px' }}>
            {(rm.risk_limits || []).map((l, i) => (
              <div key={i} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '10px 12px', borderRadius: '10px',
                background: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.04)',
              }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: theme.colors.text }}>{l.name}</span>
                <span style={{
                  fontSize: '11px', fontWeight: 800, padding: '2px 8px', borderRadius: '6px',
                  background: l.status === 'ok' ? `${theme.colors.primary}15` : `${theme.colors.danger}15`,
                  color: l.status === 'ok' ? theme.colors.primary : theme.colors.danger,
                  fontVariantNumeric: 'tabular-nums',
                }}>
                  {(l.current_value || 0).toFixed(4)} / {l.threshold}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const SystemPulse = ({ data, mobile }) => (
  <div style={columns('1fr 2fr', mobile)}>
    <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? '14px' : '20px', minWidth: 0 }}>
      <div style={card(mobile)}>
        <SectionHeader title="System Status" icon={Cpu} />
        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <span style={{ color: theme.colors.textSecondary, fontSize: '13px' }}>Uptime</span>
          <span style={{ fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>{Math.floor((data.systemHealth?.uptime || 0) / 3600)}h</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: '8px' }}>
          <span style={{ color: theme.colors.textSecondary, fontSize: '13px' }}>Memory</span>
          <span style={{ color: theme.colors.primary, fontWeight: 800 }}>{(data.systemHealth?.memory_usage || 'stable').toUpperCase()}</span>
        </div>
      </div>
      <div style={card(mobile)}>
        <SectionHeader title="Service Health" icon={Activity} />
        {(() => {
          const comps = data.status?.components || {};
          const names = Object.keys(comps);
          if (names.length === 0) {
            return <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>Awaiting status…</div>;
          }
          return (
            <div style={{ display: 'grid', gap: '8px' }}>
              {names.map(name => {
                const c = comps[name] || {};
                const up = c.is_running ?? c.is_connected ?? c.initialized ?? true;
                return (
                  <div key={name} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '8px 10px', borderRadius: '8px',
                    background: 'rgba(255, 255, 255, 0.02)',
                  }}>
                    <span style={{ textTransform: 'capitalize', fontSize: '12px', fontWeight: 600 }}>{name.replace(/_/g, ' ')}</span>
                    <span style={{
                      padding: '2px 8px', borderRadius: '6px', fontSize: '10px', fontWeight: 800,
                      background: up ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                      color: up ? theme.colors.primary : theme.colors.danger,
                    }}>
                      {up ? 'ONLINE' : 'OFFLINE'}
                    </span>
                  </div>
                );
              })}
            </div>
          );
        })()}
      </div>
    </div>
    <div style={card(mobile)}>
      <SectionHeader title="Agent Log" icon={Terminal} />
      <div style={{
        backgroundColor: '#020617', padding: '16px', borderRadius: '12px',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        height: mobile ? '320px' : '400px', overflowY: 'auto', overflowWrap: 'anywhere',
        fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', lineHeight: 1.6,
      }}>
        {data.agentActivity.length > 0 ? (
          data.agentActivity.map((log, i) => (
            <div key={i} style={{ marginBottom: '6px' }}>
              <span style={{ color: theme.colors.textMuted }}>[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
              <span style={{ color: theme.colors.secondary, fontWeight: 700 }}>{log.component}</span>: {log.message}
            </div>
          ))
        ) : (
          <div style={{ color: theme.colors.textMuted }}>Waiting for logs...</div>
        )}
      </div>
    </div>
  </div>
);

export default function SystemSection({ sub, onSub, data, mobile }) {
  const tabs = [
    { id: 'risk', label: 'Risk & Controls' },
    { id: 'pulse', label: 'System Pulse' },
    { id: 'scorecard', label: 'Scorecard' }
  ];
  const current = tabs.some((t) => t.id === sub) ? sub : 'risk';
  return (
    <div>
      <SubTabs tabs={tabs} active={current} onChange={onSub} />
      {current === 'risk' && <RiskView data={data} mobile={mobile} />}
      {current === 'pulse' && <SystemPulse data={data} mobile={mobile} />}
      {current === 'scorecard' && <Scorecard mobile={mobile} />}
    </div>
  );
}
