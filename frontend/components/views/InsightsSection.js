// Insights: performance analytics and the market feeds (news and the
// sector heatmap), as two tabs of one section.
import { ExternalLink, Globe, Layers } from 'lucide-react';
import AdvancedAnalytics from '../AdvancedAnalytics';
import { theme } from '../DashboardStyles';
import { card, columns, SectionHeader, SubTabs } from '../ui';

const MarketFeeds = ({ data, mobile, onSector }) => (
  <div style={columns('2fr 1fr', mobile)}>
    <div style={card(mobile)}>
      <SectionHeader title="Intelligence Feed" icon={Globe} />
      {data.news.map((n, i) => {
        const Wrapper = n.url ? 'a' : 'div';
        const wrapperProps = n.url
          ? { href: n.url, target: '_blank', rel: 'noopener noreferrer', style: { display: 'block', textDecoration: 'none', color: 'inherit', cursor: 'pointer' } }
          : {};
        return (
          <Wrapper key={i} {...wrapperProps}>
            <div style={{ padding: '15px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '10px', fontWeight: '800', color: n.sentiment_label === 'positive' ? theme.colors.primary : (n.sentiment_label === 'negative' ? theme.colors.danger : theme.colors.textMuted) }}>
                  {n.sentiment_label?.toUpperCase() || 'NEUTRAL'}{n.region === 'east_africa' ? ' · EAST AFRICA' : ''}
                </span>
                <span style={{ fontSize: '10px', color: theme.colors.textMuted }}>{new Date(n.time).toLocaleTimeString()}</span>
              </div>
              <div style={{ fontWeight: '700', marginBottom: '5px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                {n.title}{n.url && <ExternalLink size={12} color={theme.colors.textMuted} />}
              </div>
              <div style={{ fontSize: '12px', color: theme.colors.textSecondary }}>{n.summary}</div>
              {n.source && <div style={{ fontSize: '10px', color: theme.colors.textMuted, marginTop: '4px' }}>{n.source}</div>}
            </div>
          </Wrapper>
        );
      })}
    </div>
    <div style={card(mobile)}>
      <SectionHeader title="Market Heatmap" icon={Layers} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
        {data.heatmap.map((s, i) => {
          const score = s.outlook_score ?? 0.0;
          let cellBg = 'rgba(255, 255, 255, 0.04)';
          if (score > 0.05) {
            cellBg = `rgba(16, 185, 129, ${0.1 + score * 0.7})`;
          } else if (score < -0.05) {
            cellBg = `rgba(244, 63, 94, ${0.1 + Math.abs(score) * 0.7})`;
          }
          
          return (
            <div 
              key={i} 
              onClick={() => onSector(s.sector)}
              title={`Drill into ${s.sector} sector outlook`}
              style={{ 
                aspectRatio: '1', 
                borderRadius: '10px', 
                background: cellBg,
                border: `1px solid ${score > 0.1 ? theme.colors.primary : (score < -0.1 ? theme.colors.danger : theme.colors.border)}40`,
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center', 
                justifyContent: 'center', 
                textAlign: 'center', 
                fontSize: '10px', 
                fontWeight: '800',
                cursor: 'pointer',
                padding: '8px',
                boxShadow: score > 0.4 ? '0 0 10px rgba(16, 185, 129, 0.15)' : (score < -0.4 ? '0 0 10px rgba(244, 63, 94, 0.15)' : 'none'),
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ color: '#fff', fontSize: '10px', marginBottom: '4px' }}>{s.sector}</div>
              <div style={{ fontSize: '9px', color: theme.colors.textSecondary, fontWeight: 'normal' }}>
                ETF: {s.performance >= 0 ? '+' : ''}{((s.performance || 0) * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: '9px', color: score >= 0.1 ? theme.colors.primary : (score <= -0.1 ? theme.colors.danger : theme.colors.textMuted), fontWeight: 'bold', marginTop: '2px' }}>
                Score: {score >= 0 ? '+' : ''}{score.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  </div>
);

export default function InsightsSection({ sub, onSub, data, mobile, onSector }) {
  const tabs = [{ id: 'analytics', label: 'Analytics' }, { id: 'feeds', label: 'Market Feeds' }];
  const current = tabs.some((t) => t.id === sub) ? sub : 'analytics';
  return (
    <div>
      <SubTabs tabs={tabs} active={current} onChange={onSub} />
      {current === 'analytics' ? <AdvancedAnalytics data={data} mobile={mobile} /> : <MarketFeeds data={data} mobile={mobile} onSector={onSector} />}
    </div>
  );
}
