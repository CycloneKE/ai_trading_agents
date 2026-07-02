import { X, Shield, Activity, Globe, Cpu, Flag, BookOpen } from 'lucide-react';
import { theme, glassCard } from './DashboardStyles';

const TABS = [
  { icon: Activity, name: 'DASHBOARD', text: 'Consolidated equity, P&L, open positions, live news with AI sentiment scoring. Start here.' },
  { icon: Flag, name: 'NSE KENYA', text: 'Nairobi Securities Exchange quotes, top movers, and sector performance.' },
  { icon: Shield, name: 'RISK', text: 'How much the portfolio can lose and how close the agent is to its limits.' },
  { icon: Globe, name: 'MARKET', text: 'Sector heatmap and broad market context the agent is reacting to.' },
  { icon: Cpu, name: 'SYSTEM', text: 'Service health and the raw agent activity log - what the AI is doing and why.' },
];

const GLOSSARY = [
  ['Consolidated Equity', 'Total account value: cash plus the market value of every open position.'],
  ['P&L', 'Profit and loss. Unrealized = open positions at current prices; realized = closed trades.'],
  ['VaR (Value at Risk)', 'Statistical estimate of the most the portfolio should lose in a day at 95% confidence. The agent halts if limits are breached.'],
  ['Sharpe Ratio', 'Return earned per unit of risk taken. Above 1 is good, above 2 is strong. Comparing strategies? Compare Sharpe, not raw return.'],
  ['Win Rate', 'Share of closed trades that were profitable. High win rate with small wins can still lose money - read it next to P&L.'],
  ['Max Drawdown', 'Worst peak-to-trough decline. The system is capped at 10% before risk controls step in.'],
  ['Sentiment Score', 'AI-scored tone of each news item from -1 (bearish) to +1 (bullish), shown on the news feed.'],
];

const HelpPanel = ({ onClose }) => (
  <div
    onClick={onClose}
    style={{
      position: 'fixed', inset: 0, backgroundColor: 'rgba(2, 6, 23, 0.85)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '20px',
    }}
  >
    <div
      onClick={(e) => e.stopPropagation()}
      style={{ ...glassCard, maxWidth: '680px', width: '100%', maxHeight: '85vh', overflowY: 'auto', padding: '32px' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '10px' }}>
          <BookOpen size={20} color={theme.colors.primary} /> Getting Started
        </h2>
        <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: theme.colors.textSecondary, cursor: 'pointer' }}>
          <X size={20} />
        </button>
      </div>

      <div style={{
        border: `1px solid ${theme.colors.warning}`, borderRadius: '8px', padding: '12px 16px',
        marginBottom: '24px', fontSize: '13px', color: theme.colors.textSecondary,
      }}>
        <strong style={{ color: theme.colors.warning }}>Paper trading session.</strong>{' '}
        No real money moves. Everyone in this test session shares one simulated
        portfolio - you are observing and evaluating the AI agent together, not
        trading against each other.
      </div>

      <h3 style={{ fontSize: '13px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase' }}>The Tabs</h3>
      {TABS.map(({ icon: Icon, name, text }) => (
        <div key={name} style={{ display: 'flex', gap: '12px', marginBottom: '12px', fontSize: '13px' }}>
          <Icon size={16} color={theme.colors.primary} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div><strong>{name}</strong> — <span style={{ color: theme.colors.textSecondary }}>{text}</span></div>
        </div>
      ))}

      <h3 style={{ fontSize: '13px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase', marginTop: '24px' }}>Reading the Numbers</h3>
      {GLOSSARY.map(([term, def]) => (
        <div key={term} style={{ marginBottom: '10px', fontSize: '13px' }}>
          <strong style={{ color: theme.colors.primary }}>{term}</strong>{' '}
          <span style={{ color: theme.colors.textSecondary }}>{def}</span>
        </div>
      ))}

      <h3 style={{ fontSize: '13px', fontWeight: 800, color: theme.colors.textMuted, textTransform: 'uppercase', marginTop: '24px' }}>Session Notes</h3>
      <ul style={{ fontSize: '13px', color: theme.colors.textSecondary, paddingLeft: '18px', margin: 0 }}>
        <li style={{ marginBottom: '6px' }}>Panels refresh every 20–30 seconds; the header clock shows the current US market session.</li>
        <li style={{ marginBottom: '6px' }}>Premarket quotes can be sparse or delayed - that is the data vendor, not a fault.</li>
        <li style={{ marginBottom: '6px' }}>Your login is valid for 24 hours; after that, sign in again.</li>
        <li>Reopen this guide any time with the <strong>?</strong> button in the header.</li>
      </ul>
    </div>
  </div>
);

export default HelpPanel;
