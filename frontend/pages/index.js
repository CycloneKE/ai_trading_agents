import { useState, useEffect } from 'react';
import Head from 'next/head';
import AdvancedDashboard from '../components/AdvancedDashboard';
import TradingControls from '../components/TradingControls';
import AdvancedAnalytics from '../components/AdvancedAnalytics';
import MarketData from '../components/MarketData';
import Settings from '../components/Settings';
import AgentActivity from '../components/AgentActivity';
import supabase from '../src/utils/supabaseClient';

export default function TradingDashboard() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [theme, setTheme] = useState('dark');
  const [data, setData] = useState({});
  const [mounted, setMounted] = useState(false);

  const [activities, setActivities] = useState([]);

  // Theme colors
  const colors = {
    dark: {
      bg: '#0f0f23', bgSecondary: '#1a1a2e', bgTertiary: '#16213e',
      text: '#e2e8f0', textSecondary: '#94a3b8', textMuted: '#64748b',
      accent: '#3b82f6', success: '#10b981', danger: '#ef4444', warning: '#f59e0b',
      border: '#334155', hover: '#475569'
    },
    light: {
      bg: '#ffffff', bgSecondary: '#f8fafc', bgTertiary: '#f1f5f9',
      text: '#1e293b', textSecondary: '#475569', textMuted: '#64748b',
      accent: '#3b82f6', success: '#059669', danger: '#dc2626', warning: '#d97706',
      border: '#e2e8f0', hover: '#f1f5f9'
    }
  };

  const currentTheme = colors[theme];

  useEffect(() => {
    setMounted(true);

    if (supabase) {
      // Fetch initial trades
      const fetchInitialTrades = async () => {
        const { data: initialTrades, error } = await supabase
          .from('trades')
          .select('*')
          .order('timestamp', { ascending: false })
          .limit(20);

        if (error) {
          console.error('Error fetching initial trades:', error);
          return;
        }

        if (initialTrades) {
          const formattedTrades = initialTrades.map(trade => ({
            id: trade.id,
            time: new Date(trade.timestamp).toLocaleTimeString(),
            type: (trade.action || 'HOLD').toUpperCase(),
            symbol: trade.symbol,
            quantity: trade.quantity,
            price: trade.price,
            reason: trade.strategy || 'Trade execution',
          }));
          setActivities(formattedTrades);
        }
      };

      fetchInitialTrades();

      // Subscribe to real-time changes
      const channel = supabase
        .channel('realtime_trades')
        .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'trades' }, payload => {
          const newTrade = payload.new;
          const formattedTrade = {
            id: newTrade.id,
            time: new Date(newTrade.timestamp).toLocaleTimeString(),
            type: (newTrade.action || 'HOLD').toUpperCase(),
            symbol: newTrade.symbol,
            quantity: newTrade.quantity,
            price: newTrade.price,
            reason: newTrade.strategy || 'Trade execution',
          };
          setActivities(prev => [formattedTrade, ...prev].slice(0, 50));
        })
        .subscribe();

      return () => {
        supabase.removeChannel(channel);
      };
    }
  }, []);

  const handleAction = async (action, data) => {
    try {
      const response = await fetch(`http://localhost:5001/api/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const result = await response.json();
      console.log('Action result:', result);
    } catch (error) {
      console.error('Action error:', error);
    }
  };

  if (!mounted) return null;

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: currentTheme.bg,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <Head>
        <title>AI Trading Dashboard</title>
        <link
          rel="stylesheet"
          href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
          integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z"
          crossOrigin="anonymous"
        />
      </Head>
      {/* Navigation Tabs */}
      <nav style={{
        backgroundColor: currentTheme.bgSecondary,
        borderBottom: `1px solid ${currentTheme.border}`,
        padding: '0 24px'
      }}>
        <div style={{ display: 'flex', gap: '32px' }}>
          {[
            { key: 'dashboard', label: 'Dashboard', icon: '📊' },
            { key: 'trading', label: 'Trading', icon: '💹' },
            { key: 'market', label: 'Market Data', icon: '📈' },
            { key: 'analytics', label: 'Analytics', icon: '🔬' },
            { key: 'settings', label: 'Settings', icon: '⚙️' }
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                backgroundColor: 'transparent',
                color: activeTab === tab.key ? currentTheme.accent : currentTheme.textSecondary,
                border: 'none',
                borderBottom: activeTab === tab.key ? `2px solid ${currentTheme.accent}` : '2px solid transparent',
                padding: '16px 0',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.2s ease'
              }}
            >
              <span>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Tab Content */}
      <div style={{ padding: '24px' }}>
        {activeTab === 'dashboard' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '24px' }}>
            <div>
              <AdvancedDashboard theme={theme} />
            </div>
            <AgentActivity theme={theme} activities={activities} />
          </div>
        )}
        
        {activeTab === 'trading' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '24px' }}>
            <div>
              <AdvancedDashboard theme={theme} />
            </div>
            <TradingControls theme={currentTheme} onAction={handleAction} />
          </div>
        )}
        
        {activeTab === 'market' && (
          <MarketData theme={currentTheme} />
        )}
        
        {activeTab === 'analytics' && (
          <div style={{ display: 'grid', gap: '24px' }}>
            <AdvancedAnalytics theme={currentTheme} data={data} />
            <AdvancedDashboard theme={theme} />
          </div>
        )}
        
        {activeTab === 'settings' && (
          <Settings 
            theme={currentTheme} 
            onSave={async (config) => {
              try {
                const response = await fetch('http://localhost:5001/api/config', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(config)
                });
                if (response.ok) {
                  console.log('Configuration saved successfully');
                } else {
                  throw new Error('Failed to save configuration');
                }
              } catch (error) {
                console.error('Error saving configuration:', error);
                throw error;
              }
            }}
          />
        )}
      </div>
    </div>
  );
}