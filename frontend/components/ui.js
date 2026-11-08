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
export const card = (mobile, extra) => ({
  ...glassCard,
  padding: mobile ? '16px' : '22px',
  borderRadius: '16px',
  border: '1px solid rgba(255, 255, 255, 0.08)',
  background: 'rgba(11, 17, 32, 0.75)',
  backdropFilter: 'blur(16px)',
  ...extra,
});

// Columns on a wide screen, one column on a phone.
export const columns = (template, mobile, gap) => ({
  display: 'grid',
  gridTemplateColumns: mobile ? 'minmax(0, 1fr)' : template,
  gap: gap ?? (mobile ? '14px' : '20px'),
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

// The server stores times in UTC without saying so ("2026-09-26T19:41:00");
// read such a time as UTC so it shows in the viewer's own time zone.
export function localTime(ts) {
  if (!ts) return '';
  const iso = typeof ts === 'string' && !/(Z|[+-]\d\d:?\d\d)$/.test(ts) ? `${ts}Z` : ts;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(ts) : d.toLocaleString();
}

export const gainColor = (v) => ((num(v) ?? 0) >= 0 ? theme.colors.primary : theme.colors.danger);

// ------------------------------------------------------------- components

export const SectionHeader = ({ title, icon: Icon, badge, desc, right }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '16px', flexWrap: 'wrap' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      {Icon && (
        <div style={{
          width: '32px', height: '32px', borderRadius: '10px',
          background: 'rgba(14, 165, 233, 0.1)', border: '1px solid rgba(14, 165, 233, 0.25)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon size={16} color={theme.colors.secondary} />
        </div>
      )}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <h3 style={{ fontSize: '15px', fontWeight: '800', margin: 0, color: '#fff', letterSpacing: '0.2px' }}>{title}</h3>
          {badge && (
            <span style={{ fontSize: '10px', fontWeight: 800, padding: '2px 7px', borderRadius: '6px', background: 'rgba(255,255,255,0.06)', color: theme.colors.textSecondary, border: `1px solid ${theme.colors.border}` }}>
              {badge}
            </span>
          )}
        </div>
        {desc && <p style={{ margin: '2px 0 0', fontSize: '11px', color: theme.colors.textMuted }}>{desc}</p>}
      </div>
    </div>
    {right && <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>{right}</div>}
  </div>
);

export const StatCard = ({ label, value, icon: Icon, color }) => (
  <div style={{
    ...glassCard,
    padding: '18px 20px',
    borderRadius: '14px',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    background: 'rgba(11, 17, 32, 0.75)',
    position: 'relative',
    overflow: 'hidden',
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
      <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase', letterSpacing: '0.6px' }}>{label}</span>
      {Icon && (
        <div style={{ width: '28px', height: '28px', borderRadius: '8px', background: `${color}15`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Icon size={15} color={color} />
        </div>
      )}
    </div>
    <div style={{ fontSize: '24px', fontWeight: '800', color: '#fff', fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.5px' }}>{value}</div>
  </div>
);

// A headline figure. `tone` colours the second line; by default a leading
// minus or a falling arrow reads as bad news.
export const HUDCard = ({ title, value, subValue, icon: Icon, color, footer, tone, onClick }) => {
  const isLoss = String(subValue || '').match(/^-|↘/);
  const subColor = tone || (isLoss ? theme.colors.danger : theme.colors.primary);
  return (
    <div onClick={onClick} style={{
      ...glassCard,
      padding: '18px 20px',
      position: 'relative',
      overflow: 'hidden',
      borderRadius: '16px',
      background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.8) 0%, rgba(11, 17, 32, 0.9) 100%)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      minWidth: 0,
      cursor: onClick ? 'pointer' : 'default',
      transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    }}>
      <div style={{ position: 'absolute', right: '-8px', bottom: '-8px', opacity: 0.04, pointerEvents: 'none' }}>
        <Icon size={96} color={color} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px', gap: '8px' }}>
        <span style={{ fontSize: '11px', color: theme.colors.textMuted, margin: 0, fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.8px' }}>{title}</span>
        <div style={{
          backgroundColor: `${color}18`,
          border: `1px solid ${color}35`,
          width: '30px', height: '30px',
          borderRadius: '10px',
          color,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon size={16} />
        </div>
      </div>
      <h2 style={{
        fontSize: 'clamp(20px, 4.5vw, 26px)',
        fontWeight: '900',
        margin: '2px 0 0 0',
        color: '#fff',
        fontVariantNumeric: 'tabular-nums',
        letterSpacing: '-0.5px',
        overflowWrap: 'anywhere',
      }}>{value}</h2>
      {subValue != null && (
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '4px',
          fontSize: '12px',
          fontWeight: '700',
          marginTop: '6px',
          padding: '2px 8px',
          borderRadius: '6px',
          background: `${subColor}15`,
          color: subColor,
          fontVariantNumeric: 'tabular-nums',
        }}>
          {subValue}
        </div>
      )}
      {footer && (
        <p style={{ fontSize: '11px', margin: '8px 0 0 0', color: theme.colors.textMuted, position: 'relative', zIndex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{footer}</p>
      )}
    </div>
  );
};

// A row of headline cards: two across on a phone, as many as fit otherwise.
export const HudRow = ({ mobile, children }) => (
  <div style={{
    display: 'grid', gap: mobile ? '10px' : '16px',
    gridTemplateColumns: mobile ? 'repeat(2, minmax(0, 1fr))' : 'repeat(auto-fit, minmax(210px, 1fr))',
  }}>
    {children}
  </div>
);

// Section tabs inside a page (for example NSE: Market Watch, Paper Account,
// Order Tickets). Scrolls sideways on a narrow screen instead of wrapping.
export const SubTabs = ({ tabs, active, onChange }) => (
  <div style={{
    display: 'flex', gap: '6px', overflowX: 'auto', padding: '4px', marginBottom: '18px',
    WebkitOverflowScrolling: 'touch', background: 'rgba(15, 23, 42, 0.65)',
    borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.07)', width: 'fit-content',
    maxWidth: '100%',
  }}>
    {tabs.map(({ id, label, badge }) => {
      const on = id === active;
      return (
        <button key={id} onClick={() => onChange(id)} style={{
          flexShrink: 0,
          background: on ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(14, 165, 233, 0.15) 100%)' : 'transparent',
          border: `1px solid ${on ? theme.colors.primary + '70' : 'transparent'}`,
          color: on ? '#fff' : theme.colors.textSecondary,
          borderRadius: '9px',
          padding: '8px 16px',
          fontSize: '12px',
          fontWeight: 800,
          cursor: 'pointer',
          letterSpacing: '0.3px',
          transition: 'all 0.15s ease',
        }}>
          {label}
          {badge ? (
            <span style={{ marginLeft: '6px', background: theme.colors.warning, color: '#000', borderRadius: '999px', padding: '1px 7px', fontSize: '10px', fontWeight: 800 }}>{badge}</span>
          ) : null}
        </button>
      );
    })}
  </div>
);

export const Empty = ({ children }) => (
  <div style={{ padding: '32px 20px', textAlign: 'center', color: theme.colors.textMuted, fontSize: '13px', border: '1px dashed rgba(255, 255, 255, 0.08)', borderRadius: '12px' }}>{children}</div>
);

export const Note = ({ children, tone }) => (
  <div style={{
    padding: '12px 14px', borderRadius: '10px', fontSize: '12px', lineHeight: 1.5,
    background: tone ? `${tone}15` : 'rgba(255,255,255,0.03)',
    border: `1px solid ${tone ? tone + '45' : 'rgba(255, 255, 255, 0.08)'}`,
    color: tone || theme.colors.textSecondary,
  }}>
    {children}
  </div>
);

export const th = {
  padding: '12px 10px',
  fontWeight: 800,
  fontSize: '11px',
  textTransform: 'uppercase',
  letterSpacing: '0.6px',
  color: theme.colors.textMuted,
  whiteSpace: 'nowrap',
};

export const td = {
  padding: '12px 10px',
  fontSize: '13px',
  fontVariantNumeric: 'tabular-nums',
};

export const tableHeadRow = {
  textAlign: 'left',
  borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
};
