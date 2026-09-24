// Risk & System: risk limits and alerts, and the agent's service health
// and log, as two tabs of one section.
import { Shield, TrendingDown, Zap, Lock, Bell, Cpu, Activity, Terminal } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { card, columns, SectionHeader, StatCard, HudRow, SubTabs } from '../ui';

const RiskView = ({ data, mobile }) => {
  const rm = data.riskMetrics || {};
  const metrics = rm.current_metrics || {};
  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      <HudRow mobile={mobile}>
        <StatCard label="Portfolio VaR" value={`${((metrics.portfolio_var || 0) * 100).toFixed(4)}%`} icon={Shield} color={theme.colors.warning} />
        <StatCard label="Max Drawdown" value={`${((metrics.max_drawdown || 0) * 100).toFixed(2)}%`} icon={TrendingDown} color={theme.colors.danger} />
        <StatCard label="Leverage" value={`${(metrics.leverage || 1.0).toFixed(2)}x`} icon={Zap} color={theme.colors.primary} />
      </HudRow>
      <div style={columns('1fr 1fr', mobile)}>
        <div style={card(mobile)}>
          <SectionHeader title="Risk Limits" icon={Lock} />
          {(rm.risk_limits || []).map((l, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
              <span>{l.name}</span>
              <span style={{ color: l.status === 'ok' ? theme.colors.primary : theme.colors.danger }}>{(l.current_value || 0).toFixed(4)} / {l.threshold}</span>
            </div>
          ))}
        </div>
        <div style={card(mobile)}>
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

const SystemPulse = ({ data, mobile }) => (
  <div style={columns('1fr 2fr', mobile)}>
    <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? '16px' : '24px', minWidth: 0 }}>
      <div style={card(mobile)}>
        <SectionHeader title="System Status" icon={Cpu} />
        <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Uptime</span><span>{Math.floor((data.systemHealth?.uptime || 0) / 3600)}h</span></div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px' }}><span>Memory</span><span style={{ color: theme.colors.primary }}>{(data.systemHealth?.memory_usage || 'stable').toUpperCase()}</span></div>
      </div>
      <div style={card(mobile)}>
        <SectionHeader title="Service Health" icon={Activity} />
        {(() => {
          const comps = data.status?.components || {};
          const names = Object.keys(comps);
          if (names.length === 0) {
            return <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>Awaiting status…</div>;
          }
          return names.map(name => {
            const c = comps[name] || {};
            // A component is healthy if it reports running/connected/initialized truthy.
            const up = c.is_running ?? c.is_connected ?? c.initialized ?? true;
            return (
              <div key={name} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ textTransform: 'capitalize' }}>{name.replace(/_/g, ' ')}</span>
                <span style={{ color: up ? theme.colors.primary : theme.colors.danger }}>{up ? 'ONLINE' : 'OFFLINE'}</span>
              </div>
            );
          });
        })()}
      </div>
    </div>
    <div style={card(mobile)}>
      <SectionHeader title="Agent Log" icon={Terminal} />
      <div style={{ backgroundColor: '#000', padding: '15px', borderRadius: '8px', height: mobile ? '320px' : '400px', overflowY: 'auto', overflowWrap: 'anywhere', fontFamily: 'monospace', fontSize: '11px' }}>
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

export default function SystemSection({ sub, onSub, data, mobile }) {
  const tabs = [{ id: 'risk', label: 'Risk & Controls' }, { id: 'pulse', label: 'System Pulse' }];
  const current = tabs.some((t) => t.id === sub) ? sub : 'risk';
  return (
    <div>
      <SubTabs tabs={tabs} active={current} onChange={onSub} />
      {current === 'risk' ? <RiskView data={data} mobile={mobile} /> : <SystemPulse data={data} mobile={mobile} />}
    </div>
  );
}
