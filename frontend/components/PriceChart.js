// A real daily price chart for one symbol, drawn with TradingView's open
// source Lightweight Charts library (Apache 2.0; its logo on the chart is the
// attribution the licence asks for). Bars come from /api/chart/<symbol>:
// the NSE's own ticker and price lists for NSE stocks, the live feed's daily
// history for US stocks and crypto, never synthetic seed data. The agent's
// trades are marked on the candles.
import { useEffect, useRef, useState } from 'react';
import { ExternalLink } from 'lucide-react';
import { theme } from './DashboardStyles';
import { apiGet, money, pct } from './ui';

const RANGES = [['1M', 22], ['3M', 66], ['6M', 130], ['1Y', 252], ['ALL', 0]];

const MARKER_STYLE = {
  buy: { position: 'belowBar', shape: 'arrowUp', color: '#10b981', text: 'BUY' },
  add: { position: 'belowBar', shape: 'arrowUp', color: '#0ea5e9', text: 'ADD' },
  trim: { position: 'aboveBar', shape: 'arrowDown', color: '#f59e0b', text: 'TRIM' },
  sell: { position: 'aboveBar', shape: 'arrowDown', color: '#f43f5e', text: 'SELL' },
  stop: { position: 'aboveBar', shape: 'square', color: '#f43f5e', text: 'STOP' },
};

const UP = '#10b981';
const DOWN = '#f43f5e';

// Simple moving average of the closes, from the n-th bar on.
export function sma(bars, n) {
  const out = [];
  let sum = 0;
  bars.forEach((b, i) => {
    sum += b.close;
    if (i >= n) sum -= bars[i - n].close;
    if (i >= n - 1) out.push({ time: b.time, value: +(sum / n).toFixed(4) });
  });
  return out;
}

// Wilder's RSI over n days.
export function rsi(bars, n = 14) {
  const out = [];
  let gain = 0;
  let loss = 0;
  for (let i = 1; i < bars.length; i += 1) {
    const change = bars[i].close - bars[i - 1].close;
    const g = Math.max(change, 0);
    const l = Math.max(-change, 0);
    if (i <= n) {
      gain += g;
      loss += l;
      if (i < n) continue;
      gain /= n;
      loss /= n;
    } else {
      gain = (gain * (n - 1) + g) / n;
      loss = (loss * (n - 1) + l) / n;
    }
    out.push({ time: bars[i].time, value: +(loss === 0 ? 100 : 100 - 100 / (1 + gain / loss)).toFixed(2) });
  }
  return out;
}

// Each trade pinned to its day's bar (or the next bar if that day has none).
export function placeMarkers(bars, markers) {
  if (!bars.length) return [];
  return (markers || [])
    .map((m) => {
      const bar = bars.find((b) => b.time >= m.time) || bars[bars.length - 1];
      const style = MARKER_STYLE[m.kind] || MARKER_STYLE.buy;
      return { ...style, time: bar.time, id: `${m.kind}-${m.time}-${m.quantity}` };
    })
    .sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));
}

const ToggleButton = ({ on, color, onClick, children }) => (
  <button onClick={onClick} style={{
    background: on ? `${color}25` : 'transparent', color: on ? color : theme.colors.textMuted,
    border: `1px solid ${on ? color : theme.colors.border}`, borderRadius: '6px',
    padding: '4px 8px', fontSize: '11px', fontWeight: 800, cursor: 'pointer',
  }}>
    {children}
  </button>
);

export default function PriceChart({ symbol, height = 360, onLoaded }) {
  const box = useRef(null);
  const chartRef = useRef(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [range, setRange] = useState('6M');
  const [show, setShow] = useState({ sma20: true, sma50: true, rsi: false });
  const [hover, setHover] = useState(null);

  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    apiGet(`/api/chart/${encodeURIComponent(symbol)}?days=750`)
      .then(({ ok, status, json }) => {
        if (!alive) return;
        if (ok && json && Array.isArray(json.bars)) {
          setData(json);
          if (onLoaded) onLoaded(json);
        }
        else setError(status === 404 ? 'untracked' : (json && json.error) || `HTTP ${status}`);
      })
      .catch((e) => alive && setError(e.message));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  const bars = data?.bars || [];
  const currency = data?.currency || 'USD';

  // Build the chart. The library touches the browser window, so it is loaded
  // here, on the client, rather than imported at the top of the file.
  useEffect(() => {
    if (!bars.length || !box.current) return undefined;
    let disposed = false;
    let chart = null;
    (async () => {
      const lc = await import('lightweight-charts');
      if (disposed || !box.current) return;
      chart = lc.createChart(box.current, {
        autoSize: true,
        layout: {
          background: { type: 'solid', color: 'transparent' },
          textColor: theme.colors.textSecondary, fontSize: 11, attributionLogo: true,
          panes: { separatorColor: 'rgba(51, 65, 85, 0.6)' },
        },
        grid: { vertLines: { color: 'rgba(51, 65, 85, 0.25)' }, horzLines: { color: 'rgba(51, 65, 85, 0.25)' } },
        rightPriceScale: { borderColor: theme.colors.border },
        timeScale: { borderColor: theme.colors.border, rightOffset: 3 },
        crosshair: { mode: lc.CrosshairMode.Normal },
      });
      const candles = chart.addSeries(lc.CandlestickSeries, {
        upColor: UP, downColor: DOWN, wickUpColor: UP, wickDownColor: DOWN, borderVisible: false,
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
      });
      candles.setData(bars);
      candles.priceScale().applyOptions({ scaleMargins: { top: 0.08, bottom: 0.24 } });

      const volume = chart.addSeries(lc.HistogramSeries, {
        priceFormat: { type: 'volume' }, priceScaleId: '', lastValueVisible: false, priceLineVisible: false,
      });
      volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      volume.setData(bars.map((b) => ({ time: b.time, value: b.volume || 0, color: b.close >= b.open ? `${UP}55` : `${DOWN}55` })));

      const line = (color, points, pane) => {
        const s = chart.addSeries(lc.LineSeries, {
          color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        }, pane);
        s.setData(points);
        return s;
      };
      if (show.sma20) line(theme.colors.secondary, sma(bars, 20), 0);
      if (show.sma50) line(theme.colors.warning, sma(bars, 50), 0);
      if (show.rsi) {
        const r = line('#c084fc', rsi(bars, 14), 1);
        r.createPriceLine({ price: 70, color: `${DOWN}90`, lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
        r.createPriceLine({ price: 30, color: `${UP}90`, lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
        const panes = chart.panes();
        if (panes[1]) panes[1].setHeight(Math.round(height * 0.25));
      }
      lc.createSeriesMarkers(candles, placeMarkers(bars, data.markers));

      chart.subscribeCrosshairMove((param) => {
        const bar = param && param.time ? param.seriesData.get(candles) : null;
        setHover(bar ? { ...bar, time: param.time } : null);
      });
      chartRef.current = chart;
      applyRange(chart, bars.length, range);
    })();
    return () => {
      disposed = true;
      chartRef.current = null;
      if (chart) chart.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, show, height]);

  useEffect(() => {
    if (chartRef.current) applyRange(chartRef.current, bars.length, range);
  }, [range, bars.length]);

  const last = bars[bars.length - 1];
  const prev = bars[bars.length - 2];
  const shown = hover || last;
  const shownPrev = hover ? bars[bars.findIndex((b) => b.time === hover.time) - 1] : prev;
  const change = shown && shownPrev ? (shown.close / shownPrev.close - 1) * 100 : null;
  const tvUrl = data?.tradingview_symbol
    ? `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(data.tradingview_symbol)}` : null;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '8px' }}>
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          {RANGES.map(([id]) => (
            <button key={id} onClick={() => setRange(id)} style={{
              background: range === id ? 'rgba(255,255,255,0.1)' : 'transparent',
              color: range === id ? '#fff' : theme.colors.textMuted, border: 'none', borderRadius: '6px',
              padding: '4px 8px', fontSize: '11px', fontWeight: 800, cursor: 'pointer',
            }}>{id}</button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          <ToggleButton on={show.sma20} color={theme.colors.secondary} onClick={() => setShow((s) => ({ ...s, sma20: !s.sma20 }))}>SMA 20</ToggleButton>
          <ToggleButton on={show.sma50} color={theme.colors.warning} onClick={() => setShow((s) => ({ ...s, sma50: !s.sma50 }))}>SMA 50</ToggleButton>
          <ToggleButton on={show.rsi} color="#c084fc" onClick={() => setShow((s) => ({ ...s, rsi: !s.rsi }))}>RSI 14</ToggleButton>
        </div>
      </div>

      {shown && (
        <div style={{ fontSize: '11px', color: theme.colors.textSecondary, marginBottom: '6px', display: 'flex', gap: '10px', flexWrap: 'wrap', fontFamily: 'JetBrains Mono, monospace' }}>
          <span style={{ color: '#fff', fontWeight: 700 }}>{shown.time}</span>
          <span>O {shown.open?.toFixed(2)}</span>
          <span>H {shown.high?.toFixed(2)}</span>
          <span>L {shown.low?.toFixed(2)}</span>
          <span>C <strong style={{ color: '#fff' }}>{shown.close?.toFixed(2)}</strong></span>
          {change != null && <span style={{ color: change >= 0 ? UP : DOWN }}>{pct(change)}</span>}
        </div>
      )}

      {error && (
        <div style={{ padding: '24px', textAlign: 'center', color: theme.colors.textMuted, fontSize: '13px' }}>
          {error === 'untracked' ? `${symbol} is not a symbol the agent tracks, so there is no chart.` : `The chart could not load (${error}).`}
        </div>
      )}
      {!data && !error && (
        <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: theme.colors.textMuted, fontSize: '13px' }}>Loading daily prices…</div>
      )}
      {data && !bars.length && (
        <div style={{ padding: '24px', textAlign: 'center', color: theme.colors.textMuted, fontSize: '13px' }}>
          No real daily prices are stored for {symbol} yet. The chart fills in as each trading day is recorded.
        </div>
      )}
      {bars.length > 0 && <div ref={box} style={{ height, width: '100%' }} />}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginTop: '8px', fontSize: '11px', color: theme.colors.textMuted }}>
        <span>
          {bars.length > 0 && <>{bars.length} daily bars in {currency}{last ? `, last close ${money(last.close, currency)}` : ''}. </>}
          {(data?.markers || []).length > 0
            ? <>
                The agent's {data.markers.length} trade{data.markers.length === 1 ? ' is' : 's are'} marked:{' '}
                {['buy', 'add', 'trim', 'sell', 'stop'].map((k, i) => (
                  <span key={k} style={{ color: MARKER_STYLE[k].color, fontWeight: 700 }}>{i ? ', ' : ''}{MARKER_STYLE[k].text.toLowerCase()}</span>
                ))}.
              </>
            : data ? <>No agent trades on this symbol yet.</> : null}
        </span>
        {tvUrl && (
          <a href={tvUrl} target="_blank" rel="noopener noreferrer" style={{ color: theme.colors.secondary, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '4px', fontWeight: 700 }}>
            Open in TradingView <ExternalLink size={12} />
          </a>
        )}
      </div>
    </div>
  );
}

function applyRange(chart, count, range) {
  const days = (RANGES.find(([id]) => id === range) || [])[1];
  const ts = chart.timeScale();
  if (!days || count <= days) ts.fitContent();
  else ts.setVisibleLogicalRange({ from: count - days, to: count + 2 });
}
