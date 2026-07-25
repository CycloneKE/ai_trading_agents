import { useState, useEffect } from 'react';
import { theme } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

const Lane = ({ title, color, children }) => (
  <div style={{ flex: 1, minWidth: '180px' }}>
    <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '1px', color, marginBottom: '10px' }}>{title}</div>
    {children}
  </div>
);

const AgentFocus = ({ onDrill }) => {
  const [focus, setFocus] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/agent-focus`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setFocus(json);
      } catch (e) {}
    };
    load();
    const id = setInterval(load, 20000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!focus) return null;
  const row = { display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '13px', cursor: 'pointer' };
  const empty = (msg) => <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>{msg}</div>;

  return (
    <div style={{ ...theme.glass, padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 800, letterSpacing: '1px', color: '#fff' }}>AGENT OPERATIONS — LIVE</h3>
        <span style={{ fontSize: '12px', color: theme.colors.textMuted }}>
          Deployed: <span style={{ color: theme.colors.primary, fontWeight: 800 }}>{focus.cash?.deployed_pct ?? 0}%</span>
          {focus.cash?.policy_enabled && <> / target {focus.cash?.target_deployment_pct}%</>}
        </span>
      </div>
      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        <Lane title={`HOLDING (${focus.holding.length})`} color={theme.colors.primary}>
          {focus.holding.length === 0 && empty('No open positions')}
          {focus.holding.map(h => (
            <div key={h.symbol} style={row} onClick={() => onDrill(h.symbol)}>
              <span style={{ fontWeight: 700 }}>{h.symbol}</span>
              <span style={{ color: (h.unrealized_pl_pct ?? 0) >= 0 ? theme.colors.primary : theme.colors.danger }}>
                {h.quantity} · {(h.unrealized_pl_pct ?? 0) >= 0 ? '+' : ''}{(h.unrealized_pl_pct ?? 0).toFixed(2)}%
              </span>
            </div>
          ))}
        </Lane>
        <Lane title={`REVIEWING (${focus.reviewing.length})`} color={theme.colors.warning}>
          {focus.reviewing.length === 0 && empty('Nothing under review this cycle')}
          {focus.reviewing.slice(0, 8).map(r => (
            <div key={r.symbol} style={row} onClick={() => onDrill(r.symbol)}>
              <span style={{ fontWeight: 700 }}>{r.symbol}</span>
              <span style={{ color: theme.colors.textMuted, fontSize: '11px' }}>{r.action} · {r.reason}</span>
            </div>
          ))}
        </Lane>
        <Lane title={`TRADED (${focus.traded.length})`} color={theme.colors.accent}>
          {focus.traded.length === 0 && empty('No executions yet today')}
          {focus.traded.slice(0, 8).map(t => (
            <div key={`${t.symbol}-${t.ts}`} style={row} onClick={() => onDrill(t.symbol)}>
              <span style={{ fontWeight: 700 }}>{t.symbol}</span>
              <span style={{ fontSize: '11px' }}>{t.action} @ {t.price}</span>
            </div>
          ))}
        </Lane>
      </div>
    </div>
  );
};

export default AgentFocus;
