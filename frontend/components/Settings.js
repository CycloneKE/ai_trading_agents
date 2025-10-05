import { useState } from 'react';

const Settings = ({ theme, onSave }) => {
  const [config, setConfig] = useState({
    trading: {
      initial_capital: 100000,
      max_position_size: 0.08,
      commission: 0.001,
      slippage: 0.0005
    },
    strategies: {
      momentum: { enabled: true, weight: 1.5, lookback: 20 },
      mean_reversion: { enabled: true, weight: 1.2, lookback: 14 },
      sentiment: { enabled: true, weight: 0.8, lookback: 5 },
      reinforcement: { enabled: true, weight: 1.0, lookback: 30 }
    },
    risk_management: {
      max_drawdown: 0.15,
      var_limit: 0.015,
      correlation_limit: 0.7,
      position_timeout: 3600
    },
    notifications: {
      email_alerts: false,
      trade_notifications: true,
      error_alerts: true,
      performance_reports: true
    }
  });

  const [activeSection, setActiveSection] = useState('trading');
  const [hasChanges, setHasChanges] = useState(false);
  const currentTheme = theme;

  const updateConfig = (section, key, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
    setHasChanges(true);
  };

  const updateNestedConfig = (section, subsection, key, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [subsection]: {
          ...prev[section][subsection],
          [key]: value
        }
      }
    }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    try {
      await onSave(config);
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save config:', error);
    }
  };

  const renderTradingSettings = () => (
    <div style={{ display: 'grid', gap: '20px' }}>
      <div>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
          Initial Capital ($)
        </label>
        <input
          type="number"
          value={config.trading.initial_capital}
          onChange={(e) => updateConfig('trading', 'initial_capital', parseFloat(e.target.value))}
          style={{
            width: '100%',
            backgroundColor: currentTheme.bgTertiary,
            color: currentTheme.text,
            border: `1px solid ${currentTheme.border}`,
            borderRadius: '6px',
            padding: '10px 12px',
            fontSize: '14px'
          }}
        />
      </div>

      <div>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
          Max Position Size (%)
        </label>
        <input
          type="range"
          min="0.01"
          max="0.20"
          step="0.01"
          value={config.trading.max_position_size}
          onChange={(e) => updateConfig('trading', 'max_position_size', parseFloat(e.target.value))}
          style={{ width: '100%', marginBottom: '8px' }}
        />
        <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
          {(config.trading.max_position_size * 100).toFixed(1)}%
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
            Commission (%)
          </label>
          <input
            type="number"
            step="0.0001"
            value={config.trading.commission}
            onChange={(e) => updateConfig('trading', 'commission', parseFloat(e.target.value))}
            style={{
              width: '100%',
              backgroundColor: currentTheme.bgTertiary,
              color: currentTheme.text,
              border: `1px solid ${currentTheme.border}`,
              borderRadius: '6px',
              padding: '10px 12px',
              fontSize: '14px'
            }}
          />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
            Slippage (%)
          </label>
          <input
            type="number"
            step="0.0001"
            value={config.trading.slippage}
            onChange={(e) => updateConfig('trading', 'slippage', parseFloat(e.target.value))}
            style={{
              width: '100%',
              backgroundColor: currentTheme.bgTertiary,
              color: currentTheme.text,
              border: `1px solid ${currentTheme.border}`,
              borderRadius: '6px',
              padding: '10px 12px',
              fontSize: '14px'
            }}
          />
        </div>
      </div>
    </div>
  );

  const renderStrategySettings = () => (
    <div style={{ display: 'grid', gap: '24px' }}>
      {Object.entries(config.strategies).map(([strategy, settings]) => (
        <div key={strategy} style={{
          backgroundColor: currentTheme.bgTertiary,
          border: `1px solid ${currentTheme.border}`,
          borderRadius: '8px',
          padding: '16px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h4 style={{ fontSize: '16px', fontWeight: '600', margin: 0, color: currentTheme.text, textTransform: 'capitalize' }}>
              {strategy.replace('_', ' ')} Strategy
            </h4>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={settings.enabled}
                onChange={(e) => updateNestedConfig('strategies', strategy, 'enabled', e.target.checked)}
                style={{ accentColor: currentTheme.accent }}
              />
              <span style={{ fontSize: '14px', color: currentTheme.textSecondary }}>Enabled</span>
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
                Weight
              </label>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={settings.weight}
                onChange={(e) => updateNestedConfig('strategies', strategy, 'weight', parseFloat(e.target.value))}
                disabled={!settings.enabled}
                style={{ width: '100%', marginBottom: '4px' }}
              />
              <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
                {settings.weight.toFixed(1)}
              </div>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
                Lookback Period
              </label>
              <input
                type="number"
                min="1"
                max="100"
                value={settings.lookback}
                onChange={(e) => updateNestedConfig('strategies', strategy, 'lookback', parseInt(e.target.value))}
                disabled={!settings.enabled}
                style={{
                  width: '100%',
                  backgroundColor: currentTheme.bgSecondary,
                  color: currentTheme.text,
                  border: `1px solid ${currentTheme.border}`,
                  borderRadius: '6px',
                  padding: '8px 12px',
                  fontSize: '14px'
                }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderRiskSettings = () => (
    <div style={{ display: 'grid', gap: '20px' }}>
      <div>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
          Max Drawdown (%)
        </label>
        <input
          type="range"
          min="0.05"
          max="0.30"
          step="0.01"
          value={config.risk_management.max_drawdown}
          onChange={(e) => updateConfig('risk_management', 'max_drawdown', parseFloat(e.target.value))}
          style={{ width: '100%', marginBottom: '8px' }}
        />
        <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
          {(config.risk_management.max_drawdown * 100).toFixed(1)}%
        </div>
      </div>

      <div>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
          VaR Limit (%)
        </label>
        <input
          type="range"
          min="0.005"
          max="0.050"
          step="0.001"
          value={config.risk_management.var_limit}
          onChange={(e) => updateConfig('risk_management', 'var_limit', parseFloat(e.target.value))}
          style={{ width: '100%', marginBottom: '8px' }}
        />
        <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
          {(config.risk_management.var_limit * 100).toFixed(1)}%
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
            Correlation Limit
          </label>
          <input
            type="number"
            min="0.1"
            max="1.0"
            step="0.1"
            value={config.risk_management.correlation_limit}
            onChange={(e) => updateConfig('risk_management', 'correlation_limit', parseFloat(e.target.value))}
            style={{
              width: '100%',
              backgroundColor: currentTheme.bgTertiary,
              color: currentTheme.text,
              border: `1px solid ${currentTheme.border}`,
              borderRadius: '6px',
              padding: '10px 12px',
              fontSize: '14px'
            }}
          />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
            Position Timeout (seconds)
          </label>
          <input
            type="number"
            min="300"
            max="86400"
            step="300"
            value={config.risk_management.position_timeout}
            onChange={(e) => updateConfig('risk_management', 'position_timeout', parseInt(e.target.value))}
            style={{
              width: '100%',
              backgroundColor: currentTheme.bgTertiary,
              color: currentTheme.text,
              border: `1px solid ${currentTheme.border}`,
              borderRadius: '6px',
              padding: '10px 12px',
              fontSize: '14px'
            }}
          />
        </div>
      </div>
    </div>
  );

  const renderNotificationSettings = () => (
    <div style={{ display: 'grid', gap: '16px' }}>
      {Object.entries(config.notifications).map(([key, value]) => (
        <label key={key} style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          padding: '12px 16px',
          backgroundColor: currentTheme.bgTertiary,
          border: `1px solid ${currentTheme.border}`,
          borderRadius: '8px',
          cursor: 'pointer'
        }}>
          <span style={{ fontSize: '14px', color: currentTheme.text, textTransform: 'capitalize' }}>
            {key.replace('_', ' ')}
          </span>
          <input
            type="checkbox"
            checked={value}
            onChange={(e) => updateConfig('notifications', key, e.target.checked)}
            style={{ accentColor: currentTheme.accent }}
          />
        </label>
      ))}
    </div>
  );

  const sections = [
    { key: 'trading', label: 'Trading', icon: '💰' },
    { key: 'strategies', label: 'Strategies', icon: '🎯' },
    { key: 'risk', label: 'Risk Management', icon: '🛡️' },
    { key: 'notifications', label: 'Notifications', icon: '🔔' }
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: '24px' }}>
      <div style={{
        backgroundColor: currentTheme.bgSecondary,
        border: `1px solid ${currentTheme.border}`,
        borderRadius: '12px',
        padding: '20px',
        height: 'fit-content'
      }}>
        <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: currentTheme.text }}>
          Settings
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {sections.map(section => (
            <button
              key={section.key}
              onClick={() => setActiveSection(section.key)}
              style={{
                backgroundColor: activeSection === section.key ? currentTheme.accent + '20' : 'transparent',
                color: activeSection === section.key ? currentTheme.accent : currentTheme.textSecondary,
                border: activeSection === section.key ? `1px solid ${currentTheme.accent}` : '1px solid transparent',
                borderRadius: '8px',
                padding: '12px 16px',
                fontSize: '14px',
                fontWeight: '500',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.2s ease'
              }}
            >
              <span>{section.icon}</span>
              {section.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{
        backgroundColor: currentTheme.bgSecondary,
        border: `1px solid ${currentTheme.border}`,
        borderRadius: '12px',
        padding: '24px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <h3 style={{ fontSize: '18px', fontWeight: '600', margin: 0, color: currentTheme.text }}>
            {sections.find(s => s.key === activeSection)?.label} Settings
          </h3>
          {hasChanges && (
            <button
              onClick={handleSave}
              style={{
                backgroundColor: currentTheme.success,
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '10px 20px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Save Changes
            </button>
          )}
        </div>

        {activeSection === 'trading' && renderTradingSettings()}
        {activeSection === 'strategies' && renderStrategySettings()}
        {activeSection === 'risk' && renderRiskSettings()}
        {activeSection === 'notifications' && renderNotificationSettings()}
      </div>
    </div>
  );
};

export default Settings;