import React from 'react';

const AgentActivity = ({ theme = 'dark', activities = [] }) => {

  const colors = {
    dark: { bg: 'rgba(30, 41, 59, 0.5)', text: '#f8fafc', border: 'rgba(255,255,255,0.1)', buy: '#10b981', sell: '#ef4444', hold: '#f59e0b' },
    light: { bg: '#ffffff', text: '#1e293b', border: '#e2e8f0', buy: '#059669', sell: '#dc2626', hold: '#d97706' }
  };
  
  // Handle case where theme might be an object or a string
  const themeKey = typeof theme === 'string' ? theme : 'dark';
  const currentTheme = colors[themeKey] || colors.dark;

  const getTypeStyle = (type) => {
    const baseStyle = {
      padding: '4px 10px',
      borderRadius: '6px',
      fontWeight: '800',
      fontSize: '11px',
      letterSpacing: '0.5px'
    };
    if (type === 'BUY') return { ...baseStyle, backgroundColor: `${currentTheme.buy}20`, color: currentTheme.buy, border: `1px solid ${currentTheme.buy}40` };
    if (type === 'SELL') return { ...baseStyle, backgroundColor: `${currentTheme.sell}20`, color: currentTheme.sell, border: `1px solid ${currentTheme.sell}40` };
    if (type === 'HOLD') return { ...baseStyle, backgroundColor: `${currentTheme.hold}20`, color: currentTheme.hold, border: `1px solid ${currentTheme.hold}40` };
    return baseStyle;
  };

  return (
    <div style={{ backgroundColor: currentTheme.bg, padding: '24px', borderRadius: '12px', border: `1px solid ${currentTheme.border}`, backdropFilter: 'blur(10px)' }}>
      <h3 style={{ color: '#fff', marginBottom: '20px', fontSize: '16px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px' }}>Agent Activity</h3>
      <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
        {activities.length === 0 && (
          <div style={{ color: currentTheme.text, opacity: 0.5, fontSize: '13px', padding: '12px 0' }}>
            No agent decisions yet this session.
          </div>
        )}
        {activities.map((activity, i) => (
          <div key={activity.id || i} style={{
            display: 'flex',
            alignItems: 'center',
            padding: '12px 0',
            borderBottom: `1px solid ${currentTheme.border}`
          }}>
            <div style={{ width: '100px', color: currentTheme.text, fontSize: '14px' }}>{activity.time || ''}</div>
            <div style={{ width: '80px' }}>
              <span style={getTypeStyle(activity.type)}>{activity.type || '—'}</span>
            </div>
            <div style={{ flex: 1, color: currentTheme.text, fontSize: '14px', fontWeight: '500' }}>
              {activity.symbol ? `${activity.symbol}${activity.price ? ` @ ${activity.price}` : ''}` : ''}
            </div>
            <div style={{ flex: 2, color: currentTheme.text, fontSize: '14px' }}>{activity.reason || activity.message || ''}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentActivity;
