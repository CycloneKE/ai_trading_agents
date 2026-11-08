// Portfolio: every holding together (the original page), and one page for
// each paper account the agent trades, US and crypto and forex, with a strip
// at the top that says how each account, the NSE one included, is doing.
import { useCallback, useEffect, useState } from 'react';
import UnifiedPortfolio from '../UnifiedPortfolio';
import PaperBook from './PaperBook';
import { theme } from '../DashboardStyles';
import { apiGet, card, money, pct, gainColor, SubTabs } from '../ui';

const TABS = [
  { id: 'all', label: 'All holdings' },
  { id: 'us', label: 'US and crypto' },
  { id: 'forex', label: 'Forex' },
];

// Where each account's own page lives.
const OPEN = { us: ['portfolio', 'us'], forex: ['portfolio', 'forex'], nse: ['nse', 'paper'] };

function usePolled(path, active, seconds = 30) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const load = useCallback(async () => {
    if (!path) return;
    try {
      const { ok, status, json } = await apiGet(path);
      if (ok && json) { setData(json); setError(null); } else setError((json && json.error) || `HTTP ${status}`);
    } catch (e) { setError(e.message); }
  }, [path]);
  useEffect(() => {
    setData(null);
    if (!active || !path) return undefined;
    load();
    const id = setInterval(load, seconds * 1000);
    return () => clearInterval(id);
  }, [active, path, load, seconds]);
  return { data, error };
}

// One card per paper account: what it is worth, against what it started with.
function AccountsStrip({ accounts, onOpen, mobile }) {
  if (!accounts.length) return null;
  return (
    <div style={{
      display: 'grid', gap: mobile ? '10px' : '14px', marginBottom: mobile ? '16px' : '20px',
      gridTemplateColumns: mobile ? 'minmax(0, 1fr)' : 'repeat(auto-fit, minmax(220px, 1fr))',
    }}>
      {accounts.map((a) => {
        const cur = a.currency === 'KES' ? 'KES' : 'USD';
        return (
          <button
            key={a.id}
            onClick={() => onOpen(a.id)}
            disabled={!a.enabled}
            style={{
              ...card(mobile, { padding: '16px 18px' }),
              textAlign: 'left',
              cursor: a.enabled ? 'pointer' : 'default',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              color: 'inherit',
              opacity: a.enabled ? 1 : 0.6,
              borderRadius: '14px',
              transition: 'all 0.15s ease',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '10px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase', letterSpacing: '0.6px' }}>
                {a.title} Book
              </span>
              <span style={{ fontSize: '10px', fontWeight: 700, padding: '2px 6px', borderRadius: '4px', background: 'rgba(255, 255, 255, 0.06)', color: theme.colors.textSecondary }}>
                {cur}
              </span>
            </div>
            {a.enabled ? (
              <>
                <div style={{ fontSize: '22px', fontWeight: 900, color: '#fff', marginTop: '6px', fontVariantNumeric: 'tabular-nums' }}>
                  {money(a.equity, cur, 0)}
                </div>
                <div style={{ fontSize: '12px', marginTop: '4px' }}>
                  <span style={{ color: gainColor(a.return_pct), fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>
                    {a.return_pct == null ? '—' : pct(a.return_pct)}
                  </span>
                  <span style={{ color: theme.colors.textMuted, fontSize: '11px' }}>
                    {' '}from {money(a.starting_capital, cur, 0)} · {a.holdings} pos
                  </span>
                </div>
              </>
            ) : (
              <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginTop: '8px' }}>{a.reason || 'Off'}</div>
            )}
          </button>
        );
      })}
    </div>
  );
}

export default function PortfolioSection({ sub, onSub, onGo, mobile, onDrill }) {
  const current = TABS.some((t) => t.id === sub) ? sub : 'all';
  const paper = current !== 'all';
  const strip = usePolled('/api/paper-accounts', true);
  const book = usePolled(paper ? `/api/paper/${current}` : null, paper);

  const open = (id) => {
    const [section, page] = OPEN[id] || ['portfolio', 'all'];
    if (section === 'portfolio') onSub(page); else onGo(section, page);
  };

  return (
    <div>
      <AccountsStrip accounts={(strip.data && strip.data.accounts) || []} onOpen={open} mobile={mobile} />
      <SubTabs tabs={TABS} active={current} onChange={onSub} />
      {current === 'all'
        ? <UnifiedPortfolio onDrill={onDrill} />
        : <PaperBook kind={current} view={book.data} error={book.error} mobile={mobile} onDrill={onDrill} />}
    </div>
  );
}
