import React from 'react';
import { ExternalLink, Globe, Layers } from 'lucide-react';
import AdvancedAnalytics from '../AdvancedAnalytics';
import AiScorecard from './AiScorecard';
import StrategiesLearning from './StrategiesLearning';
import WatchOnlyCard from './WatchOnlyCard';
import GrowthView from '../GrowthView';
import { theme } from '../DashboardStyles';
import { card, columns, SectionHeader, SubTabs } from '../ui';

const NewsFeed = ({ news = [], mobile }) => (
  <div style={card({ padding: '16px' })}>
    <SectionHeader
      icon={Globe}
      title="Global Market Feeds"
      badge="Reuters / Bloomberg"
      desc="Curated macro and global markets coverage from international desks."
    />
    <div style={{ display: 'grid', gap: '8px' }}>
      {news.length === 0 && <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>No active news items.</div>}
      {news.map((item, i) => (
        <a
          key={i}
          href={item.url || '#'}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '10px 12px',
            background: 'rgba(255,255,255,0.02)',
            borderRadius: theme.radius.sm,
            border: `1px solid ${theme.colors.border}20`,
            textDecoration: 'none',
            color: 'inherit',
          }}
        >
          <div style={{ flex: 1, minWidth: 0, marginRight: '12px' }}>
            <div style={{ fontSize: '13px', fontWeight: 600, color: theme.colors.text }}>{item.title || item.headline}</div>
            <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '2px' }}>
              {item.source} · {item.published ? new Date(item.published).toLocaleTimeString() : ''}
            </div>
          </div>
          <ExternalLink size={14} color={theme.colors.textMuted} />
        </a>
      ))}
    </div>
  </div>
);

const MarketFeeds = ({ data, mobile, onSector }) => (
  <div style={columns(mobile ? '1fr' : '1.2fr 0.8fr', '16px')}>
    <NewsFeed news={data.news} mobile={mobile} />
    <div style={card({ padding: '16px' })}>
      <SectionHeader
        icon={Layers}
        title="Sector Breakdown"
        badge="Coverage"
        desc="Distribution of scanned symbols across key industries."
      />
      <div style={{ display: 'grid', gap: '6px' }}>
        {Object.entries(data.sectors || {}).map(([sector, count]) => (
          <div
            key={sector}
            onClick={() => onSector && onSector(sector)}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 10px',
              background: 'rgba(255,255,255,0.02)',
              borderRadius: theme.radius.sm,
              fontSize: '12px',
              cursor: onSector ? 'pointer' : 'default',
            }}
          >
            <span style={{ color: theme.colors.text, fontWeight: 500 }}>{sector}</span>
            <span style={{ color: theme.colors.primary, fontWeight: 700 }}>{count}</span>
          </div>
        ))}
        {Object.keys(data.sectors || {}).length === 0 && (
          <div style={{ color: theme.colors.textMuted, fontSize: '13px' }}>No sector distribution available.</div>
        )}
      </div>
    </div>
  </div>
);

export default function InsightsSection({ sub, onSub, data, mobile, onSector }) {
  const tabs = [
    { id: 'analytics', label: 'Analytics' },
    { id: 'growth', label: 'Continuous Growth & Arena' },
    { id: 'learning', label: 'Strategies & Learning' },
    { id: 'feeds', label: 'Market Feeds' },
    { id: 'watch', label: 'Watched Markets' }
  ];
  const current = tabs.some((t) => t.id === sub) ? sub : 'analytics';
  return (
    <div>
      <SubTabs tabs={tabs} active={current} onChange={onSub} />
      {current === 'analytics' && <><AiScorecard mobile={mobile} /><AdvancedAnalytics data={data} mobile={mobile} /></>}
      {current === 'growth' && <GrowthView mobile={mobile} onDrill={(sym) => { if (typeof window !== 'undefined') window.location.hash = `symbol/${sym}`; }} />}
      {current === 'learning' && <StrategiesLearning mobile={mobile} />}
      {current === 'feeds' && <MarketFeeds data={data} mobile={mobile} onSector={onSector} />}
      {current === 'watch' && <WatchOnlyCard mobile={mobile} />}
    </div>
  );
}
