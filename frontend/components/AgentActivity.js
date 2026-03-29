import React from 'react';

const AgentActivity = ({ theme, activities }) => {

  const colors = {
    dark: { bg: '#1a1a2e', text: '#e2e8f0', border: '#334155', buy: '#10b981', sell: '#ef4444', hold: '#f59e0b' },
    light: { bg: '#f8fafc', text: '#1e293b', border: '#e2e8f0', buy: '#059669', sell: '#dc2626', hold: '#d97706' }
  };
  const currentTheme = colors[theme];

  const getTypeStyle = (type) => {
    const baseStyle = {
      padding: '2px 8px',
      borderRadius: '4px',
      fontWeight: '600',
      fontSize: '12px',
    };
    if (type === 'BUY') return { ...baseStyle, backgroundColor: currentTheme.buy, color: 'white' };
    if (type === 'SELL') return { ...baseStyle, backgroundColor: currentTheme.sell, color: 'white' };
    if (type === 'HOLD') return { ...baseStyle, backgroundColor: currentTheme.hold, color: 'white' };
    return {};
  };

  return (
    <div style={{ backgroundColor: currentTheme.bg, padding: '24px', borderRadius: '8px' }}>
      <h3 style={{ color: currentTheme.text, marginBottom: '16px', fontSize: '18px', fontWeight: '600' }}>Agent Activity</h3>
      <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
        {activities.map(activity => (
          <div key={activity.id} style={{ 
            display: 'flex', 
            alignItems: 'center', 
            padding: '12px 0', 
            borderBottom: `1px solid ${currentTheme.border}` 
          }}>
            <div style={{ width: '100px', color: currentTheme.text, fontSize: '14px' }}>{activity.time}</div>
            <div style={{ width: '80px' }}>
              <span style={getTypeStyle(activity.type)}>{activity.type}</span>
            </div>
            <div style={{ flex: 1, color: currentTheme.text, fontSize: '14px', fontWeight: '500' }}>
              {activity.symbol && `${activity.symbol} - ${activity.quantity} @ $${activity.price}`}
            </div>
            <div style={{ flex: 2, color: currentTheme.text, fontSize: '14px' }}>{activity.reason}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentActivity;
