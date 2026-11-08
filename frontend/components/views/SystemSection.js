// Risk & System: risk limits and alerts, and the agent's service health
// and log, as two tabs of one section.
import { Shield, TrendingDown, Zap, Lock, Bell, Cpu, Activity, Terminal } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { card, columns, SectionHeader, StatCard, HudRow, SubTabs } from '../ui';
import Scorecard from './Scorecard';

const RiskView = ({ data, mobile }) => {
  const rm = data.riskMetrics || {};
  const metrics = rm.current_metrics || {};
  return (
    <div style={{ display: 'grid', gap: mobile ? '14px' : '20px' }}>
      <HudRow mobile={mobile}>
        <StatCard label="Portfolio VaR" value={`${((metrics.portfolio_var || 0) * 100).toFixed(4)}%`} icon={Shield} color={theme.colors.warning} />
        <StatCard label="Max Drawdown" value={`${((metrics.max_drawdown || 0) * 100).toFixed(2)}%`} icon={TrendingDown} color={theme.colors.danger} />
        <StatCard label="Leverage" value={`${(metrics.leverage || 1.0).toFixed(2)}x`} icon={Zap} color={theme.colors.primary} />
      </HudRow>
      <div style={columns('1fr 1fr', mobile)}>
        <div style={card(mobile)}>
          <SectionHeader title="Risk Limits" icon={Lock} />
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
        <div style={card(mobile)}>
          <SectionHeader title="Recent Alerts" icon={Bell} />
          {(rm.alerts?.recent || []).length > 0 ? (
            <div style={{ display: 'grid', gap: '8px' }}>
              {(rm.alerts?.recent || []).map((a, i) => (
                <div key={i} style={{
                  padding: '12px 14px', borderRadius: '10px',
                  background: 'rgba(244, 63, 94, 0.05)',
                  border: '1px solid rgba(244, 63, 94, 0.2)',
                  borderLeft: `4px solid ${theme.colors.danger}`,
                }}>
                  <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>{new Date(a.timestamp).toLocaleTimeString()}</div>
                  <div style={{ fontWeight: 800, fontSize: '13px', color: '#fff', marginTop: '2px' }}>{a.limit_name} Breach</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ padding: '32px 16px', textAlign: 'center', color: theme.colors.textMuted, fontSize: '13px' }}>
              No active risk alerts
            </div>
          )}
        </div>
      </div>
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
