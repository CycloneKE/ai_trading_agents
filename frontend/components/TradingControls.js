import { useState } from 'react';

const TradingControls = ({ theme, onAction }) => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [orderType, setOrderType] = useState('market');
  const [quantity, setQuantity] = useState(10);
  const [price, setPrice] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [takeProfit, setTakeProfit] = useState('');

  const currentTheme = theme;
  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX'];

  const handleTrade = (side) => {
    if (quantity <= 0) {
      alert('Please enter a valid quantity');
      return;
    }
    if (orderType === 'limit' && (!price || parseFloat(price) <= 0)) {
      alert('Please enter a valid limit price');
      return;
    }
    
    const tradeData = {
      symbol: selectedSymbol,
      side,
      quantity,
      order_type: orderType,
      price: orderType === 'limit' ? parseFloat(price) : null,
      stop_loss: stopLoss ? parseFloat(stopLoss) : null,
      take_profit: takeProfit ? parseFloat(takeProfit) : null
    };
    onAction('trade', tradeData);
    
    // Reset form after successful trade
    setQuantity(10);
    setPrice('');
    setStopLoss('');
    setTakeProfit('');
  };

  return (
    <div style={{
      backgroundColor: currentTheme.bgSecondary,
      border: `1px solid ${currentTheme.border}`,
      borderRadius: '12px',
      padding: '24px'
    }}>
      <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '20px', color: currentTheme.text }}>
        Trading Controls
      </h3>

      {/* Symbol Selection */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
          Symbol
        </label>
        <select
          value={selectedSymbol}
          onChange={(e) => setSelectedSymbol(e.target.value)}
          style={{
            width: '100%',
            backgroundColor: currentTheme.bgTertiary,
            color: currentTheme.text,
            border: `1px solid ${currentTheme.border}`,
            borderRadius: '6px',
            padding: '10px 12px',
            fontSize: '14px'
          }}
        >
          {symbols.map(symbol => (
            <option key={symbol} value={symbol}>{symbol}</option>
          ))}
        </select>
      </div>

      {/* Order Type */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
          Order Type
        </label>
        <div style={{ display: 'flex', gap: '8px' }}>
          {['market', 'limit', 'stop'].map(type => (
            <button
              key={type}
              onClick={() => setOrderType(type)}
              style={{
                flex: 1,
                backgroundColor: orderType === type ? currentTheme.accent : 'transparent',
                color: orderType === type ? 'white' : currentTheme.textSecondary,
                border: `1px solid ${currentTheme.border}`,
                borderRadius: '6px',
                padding: '8px 12px',
                fontSize: '12px',
                cursor: 'pointer',
                textTransform: 'capitalize'
              }}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Quantity */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
          Quantity
        </label>
        <input
          type="number"
          value={quantity}
          onChange={(e) => setQuantity(parseInt(e.target.value) || 0)}
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

      {/* Price (for limit/stop orders) */}
      {(orderType === 'limit' || orderType === 'stop') && (
        <div style={{ marginBottom: '16px' }}>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
            {orderType === 'limit' ? 'Limit Price' : 'Stop Price'}
          </label>
          <input
            type="number"
            step="0.01"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            placeholder={`Enter ${orderType} price`}
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
      )}

      {/* Risk Management */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '20px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: currentTheme.textSecondary }}>
            Stop Loss
          </label>
          <input
            type="number"
            step="0.01"
            value={stopLoss}
            onChange={(e) => setStopLoss(e.target.value)}
            placeholder="Optional"
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
            Take Profit
          </label>
          <input
            type="number"
            step="0.01"
            value={takeProfit}
            onChange={(e) => setTakeProfit(e.target.value)}
            placeholder="Optional"
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

      {/* Trade Buttons */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <button
          onClick={() => handleTrade('buy')}
          style={{
            backgroundColor: currentTheme.success,
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '12px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => e.target.style.opacity = '0.9'}
          onMouseLeave={(e) => e.target.style.opacity = '1'}
        >
          BUY {selectedSymbol}
        </button>
        <button
          onClick={() => handleTrade('sell')}
          style={{
            backgroundColor: currentTheme.danger,
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '12px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => e.target.style.opacity = '0.9'}
          onMouseLeave={(e) => e.target.style.opacity = '1'}
        >
          SELL {selectedSymbol}
        </button>
      </div>

      {/* Quick Actions */}
      <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: `1px solid ${currentTheme.border}` }}>
        <h4 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: currentTheme.textSecondary }}>
          Quick Actions
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <button
            onClick={() => onAction('close_all_positions')}
            style={{
              backgroundColor: currentTheme.warning,
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Close All
          </button>
          <button
            onClick={() => onAction('emergency_stop')}
            style={{
              backgroundColor: currentTheme.danger,
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Emergency Stop
          </button>
        </div>
      </div>
    </div>
  );
};

export default TradingControls;