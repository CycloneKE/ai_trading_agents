// Shared building blocks for the dashboard views: layout that adapts to a
// phone, the card and header styles, API access and number formatting.
import { useEffect, useState } from 'react';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

// Screens this wide or narrower get the phone layout: one column, tighter
// padding and the bottom navigation bar.
export const PHONE_MAX_WIDTH = 760;

const phoneQuery = () => `(max-width: ${PHONE_MAX_WIDTH}px)`;

export function useIsMobile() {
  const [mobile, setMobile] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(phoneQuery()).matches);
  useEffect(() => {
    const mq = window.matchMedia(phoneQuery());
    const update = () => setMobile(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, []);
  return mobile;
}

// A card, with less padding on a phone.
export const card = (mobile, extra) => ({ ...glassCard, padding: mobile ? '16px' : '24px', ...extra });

// Columns on a wide screen, one column on a phone.
export const columns = (template, mobile, gap) => ({
  display: 'grid',
  gridTemplateColumns: mobile ? 'minmax(0, 1fr)' : template,
  gap: gap ?? (mobile ? '16px' : '24px'),
});

// A GET against the trading API with the saved login.
export async function apiGet(path) {
  const token = typeof window !== 'undefined' ? localStorage.getItem('trading_token') : null;
  const res = await fetch(`${getApiBase()}${path}`, { headers: { Authorization: `Bearer ${token}` } });
  let json = null;
  try { json = await res.json(); } catch (e) { /* empty or non-JSON reply */ }
  return { ok: res.ok, status: res.status, json };
}

// ------------------------------------------------------------- formatting

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export function money(value, currency = 'USD', digits = 2) {
  const v = num(value);
  if (v === null) return '—';
  const s = Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const sign = v < 0 ? '-' : '';
  return currency === 'KES' ? `${sign}KES ${s}` : `${sign}$${s}`;
}

export function signedMoney(value, currency = 'USD', digits = 2) {
  const v = num(value);
  if (v === null) return '—';
  return `${v > 0 ? '+' : ''}${money(v, currency, digits)}`;
}

export function pct(value, digits = 2) {
  const v = num(value);
  if (v === null) return '—';
  return `${v > 0 ? '+' : ''}${v.toFixed(digits)}%`;
}

// Where the KES/USD rate came from, in a few words: "market, 24 Sep, 13:05",
// "daily reference, ...", or that it is the fixed fallback.
export function fxSourceNote(fx) {
  if (!fx) return 'loading rate';
  const at = fx.as_of ? new Date(fx.as_of) : null;
  const when = at && !Number.isNaN(at.getTime())
    ? at.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '';
  if (fx.source === 'fixed') return 'fixed fallback, no live rate yet';
  if (fx.source === 'exchangerate_api') return `daily reference${when ? `, ${when}` : ''}`;
  return `${fx.live ? 'market' : 'last known'}${when ? `, ${when}` : ''}`;
}

export const gainColor = (v) => ((num(v) ?? 0) >= 0 ? theme.colors.primary : theme.colors.danger);

// ------------------------------------------------------------- components

export const SectionHeader = ({ title, icon: Icon, right }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', flexWrap: 'wrap' }}>
    {Icon && <Icon size={18} color={theme.colors.secondary} />}
    <h3 style={{ fontSize: '16px', fontWeight: '700', margin: 0, color: '#fff' }}>{title}</h3>
    <div style={{ height: '1px', flex: 1, minWidth: '20px', background: `linear-gradient(90deg, ${theme.colors.border}, transparent)` }} />
    {right}
  </div>
);

export const StatCard = ({ label, value, icon: Icon, color }) => (
  <div style={{ ...glassCard, padding: '16px', minWidth: 0 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
      <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase' }}>{label}</span>
      {Icon && <Icon size={16} color={color} />}
    </div>
    <div style={{ fontSize: '22px', fontWeight: '800', color: '#fff', overflowWrap: 'anywhere' }}>{value}</div>
  </div>
);

// A headline figure. `tone` colours the second line; by default a leading
// minus or a falling arrow reads as bad news.
export const HUDCard = ({ title, value, subValue, icon: Icon, color, footer, tone, onClick }) => {
  const subColor = tone || (String(subValue || '').match(/^-|↘/) ? theme.colors.danger : theme.colors.primary);
  return (
    <div onClick={onClick} style={{ ...glassCard, padding: '16px 18px', position: 'relative', overflow: 'hidden', minWidth: 0, cursor: onClick ? 'pointer' : 'default' }}>
      <div style={{ position: 'absolute', right: '-10px', bottom: '-10px', opacity: 0.05 }}>
        <Icon size={90} color={color} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px', gap: '8px' }}>
        <p style={{ fontSize: '11px', color: theme.colors.textSecondary, margin: 0, fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.8px' }}>{title}</p>
        <div style={{ backgroundColor: `${color}20`, padding: '5px', borderRadius: '8px', color, display: 'flex' }}>
          <Icon size={16} />
        </div>
      </div>
      <h2 style={{ fontSize: 'clamp(18px, 4.5vw, 26px)', fontWeight: '800', margin: 0, color: '#fff', overflowWrap: 'anywhere' }}>{value}</h2>
      {subValue != null && (
        <p style={{ fontSize: '12px', margin: '4px 0 0 0', color: subColor }}>{subValue}</p>
      )}
      {footer && (
        <p style={{ fontSize: '11px', margin: '6px 0 0 0', color: theme.colors.textMuted, position: 'relative', zIndex: 1 }}>{footer}</p>
      )}
    </div>
  );
};

// A row of headline cards: two across on a phone, as many as fit otherwise.
export const HudRow = ({ mobile, children }) => (
  <div style={{
    display: 'grid', gap: mobile ? '10px' : '16px',
    gridTemplateColumns: mobile ? 'repeat(2, minmax(0, 1fr))' : 'repeat(auto-fit, minmax(200px, 1fr))',
  }}>
    {children}
  </div>
);

// Section tabs inside a page (for example NSE: Market Watch, Paper Account,
// Order Tickets). Scrolls sideways on a narrow screen instead of wrapping.
export const SubTabs = ({ tabs, active, onChange }) => (
  <div style={{ display: 'flex', gap: '6px', overflowX: 'auto', paddingBottom: '2px', marginBottom: '18px', WebkitOverflowScrolling: 'touch' }}>
    {tabs.map(({ id, label, badge }) => {
      const on = id === active;
      return (
        <button key={id} onClick={() => onChange(id)} style={{
          flexShrink: 0, background: on ? `${theme.colors.primary}22` : 'rgba(255,255,255,0.04)',
          border: `1px solid ${on ? theme.colors.primary + '80' : theme.colors.border}`,
          color: on ? '#fff' : theme.colors.textSecondary, borderRadius: '999px',
          padding: '8px 14px', fontSize: '12px', fontWeight: 800, cursor: 'pointer', letterSpacing: '0.3px',
        }}>
          {label}
          {badge ? (
            <span style={{ marginLeft: '6px', background: theme.colors.warning, color: '#000', borderRadius: '999px', padding: '0 6px', fontSize: '10px' }}>{badge}</span>
          ) : null}
        </button>
      );
    })}
  </div>
);

export const Empty = ({ children }) => (
  <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted, fontSize: '13px' }}>{children}</div>
);

export const Note = ({ children, tone }) => (
  <div style={{
    padding: '10px 12px', borderRadius: '8px', fontSize: '12px', lineHeight: 1.5,
    background: tone ? `${tone}15` : 'rgba(255,255,255,0.03)',
    border: `1px solid ${tone ? tone + '55' : theme.colors.border}`,
    color: tone || theme.colors.textSecondary,
  }}>
    {children}
  </div>
);

export const th = { padding: '10px 8px', fontWeight: 700, whiteSpace: 'nowrap' };
export const td = { padding: '10px 8px' };
export const tableHeadRow = { textAlign: 'left', color: theme.colors.textMuted, fontSize: '11px', borderBottom: `1px solid ${theme.colors.border}` };
