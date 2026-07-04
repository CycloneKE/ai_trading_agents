import { useState, useEffect } from 'react';
import { Clock } from 'lucide-react';
import { theme } from './DashboardStyles';

// US equity session boundaries in minutes since midnight, Eastern Time.
const PREMARKET_START = 4 * 60;
const MARKET_OPEN = 9 * 60 + 30;
const MARKET_CLOSE = 16 * 60;
const AFTER_HOURS_END = 20 * 60;

function getEasternNow() {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour12: false,
    weekday: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit',
  }).formatToParts(new Date());
  const get = (type) => parts.find(p => p.type === type)?.value;
  return {
    weekday: get('weekday'),
    hours: parseInt(get('hour'), 10) % 24,
    minutes: parseInt(get('minute'), 10),
    seconds: parseInt(get('second'), 10),
  };
}

function getSession(et) {
  const isWeekend = et.weekday === 'Sat' || et.weekday === 'Sun';
  const m = et.hours * 60 + et.minutes;
  if (isWeekend) return { label: 'CLOSED', color: theme.colors.textMuted, next: null };
  if (m < PREMARKET_START) return { label: 'CLOSED', color: theme.colors.textMuted, next: { label: 'PREMARKET', at: PREMARKET_START } };
  if (m < MARKET_OPEN) return { label: 'PREMARKET', color: theme.colors.warning, next: { label: 'OPEN', at: MARKET_OPEN } };
  if (m < MARKET_CLOSE) return { label: 'MARKET OPEN', color: theme.colors.primary, next: { label: 'CLOSE', at: MARKET_CLOSE } };
  if (m < AFTER_HOURS_END) return { label: 'AFTER-HOURS', color: theme.colors.accent, next: { label: 'CLOSE', at: AFTER_HOURS_END } };
  // Late evening: next premarket is 4:00 AM tomorrow (unless Friday night).
  const next = et.weekday === 'Fri' ? null : { label: 'PREMARKET', at: 24 * 60 + PREMARKET_START };
  return { label: 'CLOSED', color: theme.colors.textMuted, next };
}

function countdown(et, targetMinutes) {
  const nowSec = (et.hours * 60 + et.minutes) * 60 + et.seconds;
  const remaining = targetMinutes * 60 - nowSec;
  if (remaining <= 0) return '';
  const h = Math.floor(remaining / 3600);
  const mm = Math.floor((remaining % 3600) / 60);
  return h > 0 ? `${h}h ${mm}m` : `${mm}m`;
}

const MarketClock = () => {
  const [et, setEt] = useState(null);

  useEffect(() => {
    setEt(getEasternNow());
    const id = setInterval(() => setEt(getEasternNow()), 1000);
    return () => clearInterval(id);
  }, []);

  if (!et) return null;
  const session = getSession(et);
  const clock = `${String(et.hours).padStart(2, '0')}:${String(et.minutes).padStart(2, '0')}:${String(et.seconds).padStart(2, '0')}`;

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '10px',
      border: `1px solid ${theme.colors.border}`, borderRadius: '8px', padding: '6px 12px',
      fontFamily: 'JetBrains Mono, monospace', fontSize: '12px',
    }}>
      <Clock size={14} color={session.color} />
      <span style={{ color: theme.colors.textSecondary }}>{clock} ET</span>
      <span style={{ color: session.color, fontWeight: 700 }}>{session.label}</span>
      {session.next && (
        <span style={{ color: theme.colors.textMuted }}>
          {session.next.label} in {countdown(et, session.next.at)}
        </span>
      )}
    </div>
  );
};

export default MarketClock;
