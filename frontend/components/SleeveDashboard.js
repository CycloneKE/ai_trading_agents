import { useState, useEffect } from 'react';
import { theme } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

const SleeveDashboard = () => {
  const [view, setView] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/sleeve`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setView(json);
      } catch (e) {}
    };
    load();
    const id = setInterval(load, 30000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!view) return null;
  const row = { display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '13px' };
  const empty = (msg) => <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>{msg}</div>;

  return (
    <div style={{ ...theme.glass, padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
        <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 800, letterSpacing: '1px', color: '#fff' }}>LONG-TERM SLEEVE — NSE DIVIDENDS</h3>
        <span style={{ fontSize: '12px', color: theme.colors.textMuted }}>
          Target: <span style={{ color: theme.colors.primary, fontWeight: 800 }}>KES {(view.capital_target_kes ?? 0).toLocaleString()}</span>
          {' · '}Dividends received: <span style={{ color: theme.colors.primary, fontWeight: 800 }}>KES {(view.cumulative_dividends_kes ?? 0).toLocaleString()}</span>
        </span>
      </div>
      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: '200px' }}>
          <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '1px', color: theme.colors.primary, marginBottom: '10px' }}>
            HOLDINGS ({view.holdings.length})
          </div>
          {view.holdings.length === 0 && empty('No sleeve holdings yet')}
          {view.holdings.map(h => (
            <div key={h.symbol} style={row}>
              <span style={{ fontWeight: 700 }}>{h.symbol}</span>
              <span>{h.quantity} @ avg {h.avg_entry_price_kes}</span>
            </div>
          ))}
        </div>
        <div style={{ flex: 1, minWidth: '200px' }}>
          <div style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '1px', color: theme.colors.warning, marginBottom: '10px' }}>
            PENDING ACCUMULATION ({view.pending_tickets.length})
          </div>
          {view.pending_tickets.length === 0 && empty('Nothing pending this cycle')}
          {view.pending_tickets.map(t => (
            <div key={t.symbol} style={row}>
              <span style={{ fontWeight: 700 }}>
                {t.symbol}
                {t.veto_flagged && (
                  <span style={{ color: theme.colors.danger, marginLeft: '6px', fontSize: '11px' }} title={t.llm_reasoning}>
                    ⚑ FLAGGED
                  </span>
                )}
              </span>
              <span style={{ fontSize: '11px' }}>{t.quantity} @ {t.suggested_limit_price}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SleeveDashboard;
