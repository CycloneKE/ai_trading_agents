import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const MarketData = ({ theme }) => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1D');
  const [marketData, setMarketData] = useState([]);
  const [watchlist, setWatchlist] = useState(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']);
  const [marketSummary, setMarketSummary] = useState({});

  const currentTheme = theme;

  useEffect(() => {
    const generateMarketData = () => {
      const timeMultipliers = { '1H': 60 * 60 * 1000, '1D': 24 * 60 * 60 * 1000, '1W': 7 * 24 * 60 * 60 * 1000, '1M': 30 * 24 * 60 * 60 * 1000 };
      const dataPoints = { '1H': 60, '1D': 24, '1W': 7, '1M': 30 };
      const multiplier = timeMultipliers[timeframe] || timeMultipliers['1D'];
      const points = dataPoints[timeframe] || 24;
      
      const data = Array.from({ length: points }, (_, i) => {
        const basePrice = 150 + Math.sin(i * 0.1) * 20;
        const timeFormat = timeframe === '1H' ? { hour: '2-digit', minute: '2-digit' } : 
                          timeframe === '1D' ? { hour: '2-digit', minute: '2-digit' } :
                          { month: 'short', day: 'numeric' };
        return {
          time: new Date(Date.now() - (points - 1 - i) * multiplier / points).toLocaleString('en-US', timeFormat),
          price: basePrice + (Math.random() - 0.5) * 10,
          volume: Math.random() * 1000000,
          high: basePrice + Math.random() * 5,
          low: basePrice - Math.random() * 5,
          open: basePrice + (Math.random() - 0.5) * 3,
          close: basePrice + (Math.random() - 0.5) * 3
        };
      });
      setMarketData(data);
    };

    const generateMarketSummary = () => {
      const summary = {};
      watchlist.forEach(symbol => {
        summary[symbol] = {
          price: (Math.random() * 200 + 100).toFixed(2),
          change: ((Math.random() - 0.5) * 10).toFixed(2),
          changePercent: ((Math.random() - 0.5) * 5).toFixed(2),
          volume: Math.floor(Math.random() * 10000000),
          marketCap: (Math.random() * 2000 + 500).toFixed(1) + 'B'
        };
      });
      setMarketSummary(summary);
    };

    generateMarketData();
    generateMarketSummary();
    
    const interval = setInterval(() => {
      generateMarketData();
      generateMarketSummary();
    }, 5000);

    return () => clearInterval(interval);
  }, [selectedSymbol, watchlist, timeframe]);

  const addToWatchlist = (symbol) => {
    if (!watchlist.includes(symbol)) {
      setWatchlist([...watchlist, symbol]);
    }
  };

  const removeFromWatchlist = (symbol) => {
    setWatchlist(watchlist.filter(s => s !== symbol));
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '24px' }}>
      <div style={{
        backgroundColor: currentTheme.bgSecondary,
        border: `1px solid ${currentTheme.border}`,
        borderRadius: '12px',
        padding: '20px',
        height: 'fit-content'
      }}>
        <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', color: currentTheme.text }}>
          Watchlist
        </h3>
        
        <div style={{ marginBottom: '16px' }}>
          <input
            type="text"
            placeholder="Add symbol (e.g., AMZN)"
            onKeyPress={(e) => {
              if (e.key === 'Enter' && e.target.value) {
                addToWatchlist(e.target.value.toUpperCase());
                e.target.value = '';
              }
            }}
            style={{
              width: '100%',
              backgroundColor: currentTheme.bgTertiary,
              color: currentTheme.text,
              border: `1px solid ${currentTheme.border}`,
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '14px'
            }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {watchlist.map(symbol => {
            const data = marketSummary[symbol] || {};
            const isPositive = parseFloat(data.change) >= 0;
            
            return (
              <div
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                style={{
                  backgroundColor: selectedSymbol === symbol ? currentTheme.accent + '20' : currentTheme.bgTertiary,
                  border: selectedSymbol === symbol ? `1px solid ${currentTheme.accent}` : `1px solid ${currentTheme.border}`,
                  borderRadius: '8px',
                  padding: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <span style={{ fontSize: '14px', fontWeight: '600', color: currentTheme.text }}>
                    {symbol}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFromWatchlist(symbol);
                    }}
                    style={{
                      backgroundColor: 'transparent',
                      color: currentTheme.textMuted,
                      border: 'none',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    ×
                  </button>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '16px', fontWeight: '700', color: currentTheme.text }}>
                    ${data.price}
                  </span>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ 
                      fontSize: '12px', 
                      color: isPositive ? currentTheme.success : currentTheme.danger,
                      fontWeight: '600'
                    }}>
                      {isPositive ? '+' : ''}{data.change}
                    </div>
                    <div style={{ 
                      fontSize: '10px', 
                      color: isPositive ? currentTheme.success : currentTheme.danger
                    }}>
                      ({isPositive ? '+' : ''}{data.changePercent}%)
                    </div>
                  </div>
                </div>
                <div style={{ 
                  fontSize: '10px', 
                  color: currentTheme.textMuted,
                  marginTop: '4px'
                }}>
                  Vol: {(data.volume / 1000000).toFixed(1)}M | Cap: {data.marketCap}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div style={{
        backgroundColor: currentTheme.bgSecondary,
        border: `1px solid ${currentTheme.border}`,
        borderRadius: '12px',
        padding: '24px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h3 style={{ fontSize: '18px', fontWeight: '600', margin: 0, color: currentTheme.text }}>
              {selectedSymbol} - ${marketSummary[selectedSymbol]?.price || '0.00'}
            </h3>
            <div style={{ 
              fontSize: '14px', 
              color: parseFloat(marketSummary[selectedSymbol]?.change) >= 0 ? currentTheme.success : currentTheme.danger,
              marginTop: '4px'
            }}>
              {parseFloat(marketSummary[selectedSymbol]?.change) >= 0 ? '+' : ''}{marketSummary[selectedSymbol]?.change} 
              ({parseFloat(marketSummary[selectedSymbol]?.changePercent) >= 0 ? '+' : ''}{marketSummary[selectedSymbol]?.changePercent}%)
            </div>
          </div>
          
          <div style={{ display: 'flex', gap: '8px' }}>
            {['1H', '1D', '1W', '1M'].map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                style={{
                  backgroundColor: timeframe === tf ? currentTheme.accent : 'transparent',
                  color: timeframe === tf ? 'white' : currentTheme.textSecondary,
                  border: `1px solid ${currentTheme.border}`,
                  borderRadius: '6px',
                  padding: '6px 12px',
                  fontSize: '12px',
                  cursor: 'pointer'
                }}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={marketData}>
            <CartesianGrid strokeDasharray="3 3" stroke={currentTheme.border} />
            <XAxis dataKey="time" stroke={currentTheme.textMuted} fontSize={12} />
            <YAxis stroke={currentTheme.textMuted} fontSize={12} />
            <Tooltip 
              contentStyle={{
                backgroundColor: currentTheme.bgTertiary,
                border: `1px solid ${currentTheme.border}`,
                borderRadius: '8px',
                color: currentTheme.text
              }}
            />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke={currentTheme.accent} 
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>

        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', 
          gap: '16px', 
          marginTop: '20px',
          paddingTop: '20px',
          borderTop: `1px solid ${currentTheme.border}`
        }}>
          {[
            { label: 'Volume', value: (marketSummary[selectedSymbol]?.volume / 1000000).toFixed(1) + 'M' },
            { label: 'Market Cap', value: marketSummary[selectedSymbol]?.marketCap },
            { label: 'High', value: '$' + (parseFloat(marketSummary[selectedSymbol]?.price) + Math.random() * 5).toFixed(2) },
            { label: 'Low', value: '$' + (parseFloat(marketSummary[selectedSymbol]?.price) - Math.random() * 5).toFixed(2) }
          ].map((stat, index) => (
            <div key={index} style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '12px', color: currentTheme.textSecondary, marginBottom: '4px' }}>
                {stat.label}
              </div>
              <div style={{ fontSize: '14px', fontWeight: '600', color: currentTheme.text }}>
                {stat.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MarketData;