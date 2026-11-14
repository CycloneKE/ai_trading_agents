// The dashboard's line and area chart (account value, drawdown, paper
// accounts), drawn with TradingView's open source Lightweight Charts library
// (Apache 2.0; its logo on the chart is the attribution the licence asks for).
//
// - Range buttons show a span of time (1D is the last day, 1M the last 30
//   days), not a number of points: the account is valued every few minutes,
//   so counting points made "1D" the last minute. Ranges longer than the
//   history are greyed out.
// - Lines join the readings with straight segments. Curved lines drew values
//   between readings that never happened.
// - The chart is built once and its series updated when the data changes,
//   so a refresh every 20 seconds no longer resets the zoom.
// - Trades are marked on the bar of their day (daily data) or the nearest
//   reading (intraday); trades before the curve starts are left off rather
//   than piled on its first point.
// - Weekends are left out of daily stock data (removeWeekendGaps), so the
//   line does not run flat across them.
import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { theme } from './DashboardStyles';

const DAY = 86400;

export const RANGES = [
  { id: '1D', days: 1, label: '1D' },
  { id: '1W', days: 7, label: '1W' },
  { id: '1M', days: 30, label: '1M' },
  { id: '3M', days: 91, label: '3M' },
  { id: '6M', days: 182, label: '6M' },
  { id: '1Y', days: 365, label: '1Y' },
  { id: 'ALL', days: 0, label: 'ALL' },
];

// ISO text with or without a zone ('Z', '+03:00', '-05:00'); text with no
// zone is UTC, which is how the server writes its timestamps.
export function parseTime(raw) {
  if (raw == null || raw === '') return null;
  if (raw instanceof Date) return Number.isNaN(raw.getTime()) ? null : raw;
  if (typeof raw === 'number') return new Date(raw < 2e10 ? raw * 1000 : raw);
  const text = String(raw);
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) {
    const [y, m, d] = text.split('-').map(Number);
    return new Date(Date.UTC(y, m - 1, d, 12, 0, 0));
  }
  const zoned = /([zZ]|[+-]\d{2}:?\d{2})$/.test(text);
  const d = new Date(zoned ? text : `${text}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

// Seconds since 1970 for a chart time (a 'YYYY-MM-DD' day or seconds).
export function timeSeconds(t) {
  if (typeof t === 'number') return t;
  const d = parseTime(t);
  return d ? Math.floor(d.getTime() / 1000) : 0;
}

/**
 * Sort a series, drop unreadable times and (optionally) weekend readings,
 * and give each point the time the chart library wants: a 'YYYY-MM-DD' day
 * for daily data, seconds for intraday readings (one point per second).
 */
export function cleanTimeSeries(rawData = [], options = {}) {
  const { timeKey = null, removeWeekends = true } = options;
  if (!Array.isArray(rawData) || rawData.length === 0) return [];

  const parsed = [];
  for (const item of rawData) {
    if (!item) continue;
    const raw = (timeKey && item[timeKey]) || item.timestamp || item.day || item.time || item.date;
    const d = parseTime(raw);
    if (!d) continue;
    const day = d.getUTCDay();
    // Forex opens on Sunday evening (about 21:00 UTC), so only earlier Sunday readings go.
    if (removeWeekends && (day === 6 || (day === 0 && d.getUTCHours() < 21))) continue;
    parsed.push({
      sec: Math.floor(d.getTime() / 1000),
      dateStr: d.toISOString().slice(0, 10),
      pureDate: typeof raw === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(raw),
      item,
    });
  }
  if (!parsed.length) return [];
  parsed.sort((a, b) => a.sec - b.sec);

  const days = new Set(parsed.map((p) => p.dateStr));
  const intraday = !parsed.every((p) => p.pureDate) && parsed.length > days.size;
  if (intraday) {
    const bySecond = new Map();
    for (const p of parsed) bySecond.set(p.sec, p.item);          // the last reading of each second
    return Array.from(bySecond.entries()).map(([time, item]) => ({ ...item, time }));
  }
  const byDay = new Map();
  for (const p of parsed) byDay.set(p.dateStr, p.item);           // the last reading of each day
  return Array.from(byDay.keys()).sort().map((time) => ({ ...byDay.get(time), time }));
}

/**
 * The bar a trade belongs on: its own day's bar for daily data (or the next
 * bar, if that day has none), the nearest reading for intraday data. Null
 * for a trade outside the series.
 */
export function findClosestBar(cleanedData, rawTime) {
  if (!cleanedData || !cleanedData.length) return null;
  const d = parseTime(rawTime);
  if (!d) return null;
  const first = cleanedData[0];
  const last = cleanedData[cleanedData.length - 1];

  if (typeof first.time === 'string') {
    const day = d.toISOString().slice(0, 10);
    if (day < first.time || day > last.time) return null;
    return cleanedData.find((b) => b.time >= day) || null;
  }

  const sec = Math.floor(d.getTime() / 1000);
  if (sec < first.time - DAY || sec > last.time + DAY) return null;
  let closest = first;
  for (const b of cleanedData) {
    if (Math.abs(b.time - sec) < Math.abs(closest.time - sec)) closest = b;
  }
  return closest;
}

/** Simple moving average of `key` over `period` points. */
export function calculateSMA(data, key, period = 20) {
  const valid = data.filter((p) => typeof p[key] === 'number' && !Number.isNaN(p[key]));
  if (valid.length < 3) return [];
  const n = Math.min(period, Math.max(2, Math.floor(valid.length / 2)));
  const out = [];
  let sum = 0;
  valid.forEach((p, i) => {
    sum += p[key];
    if (i >= n) sum -= valid[i - n][key];
    if (i >= n - 1) out.push({ time: p.time, value: +(sum / n).toFixed(4) });
  });
  return out;
}

// The index of the first point inside the last `days` days.
export function rangeStart(data, days) {
  if (!days || !data.length) return 0;
  const cutoff = timeSeconds(data[data.length - 1].time) - days * DAY;
  const i = data.findIndex((p) => timeSeconds(p.time) >= cutoff);
  return i < 0 ? 0 : i;
}

// Intraday readings at one even step, so equal distances on the time axis
// are equal spans of time. The chart spaces points evenly, and the server
// sends recent readings more densely than old ones (5 minutes over the last
// two days, hourly to 30 days, daily before), which squeezed a month into
// one half of the chart and two days into the other. The step suits the
// chosen range: about 600 points across it, never finer than the readings
// themselves. Points before the range use the same step.
export function evenSteps(data, days) {
  if (data.length < 3 || typeof data[0].time !== 'number') return data;
  const lastT = data[data.length - 1].time;
  const from = days ? lastT - days * DAY : data[0].time;
  const gaps = [];
  for (let i = 1; i < data.length; i += 1) {
    if (data[i].time >= from) gaps.push(data[i].time - data[i - 1].time);
  }
  if (!gaps.length) return data;
  // The spacing that covers 80% of the range's time: daily when most of it
  // is daily history, hourly for a month of hourly readings. An outage of a
  // day or two in a long range does not count.
  gaps.sort((a, b) => a - b);
  const total = gaps.reduce((t, g) => t + g, 0);
  let covered = 0;
  let typical = gaps[gaps.length - 1];
  for (const g of gaps) {
    covered += g;
    if (covered >= total * 0.8) { typical = g; break; }
  }
  const step = Math.max(Math.ceil((lastT - from) / 600), typical, 60);
  const out = new Map();
  for (const p of data) out.set(Math.floor((p.time - lastT) / step), p);   // the last reading of each step
  return Array.from(out.values());
}

function seriesSignature(series) {
  return JSON.stringify(series.map((s) => [s.key, s.name, s.type, s.color, s.lineStyle, s.topColor, s.bottomColor, s.baseValue]));
}

const pill = (on, color) => ({
  background: on ? `${color}22` : 'transparent',
  color: on ? color : theme.colors.textMuted,
  border: `1px solid ${on ? `${color}88` : 'rgba(255,255,255,0.08)'}`,
  borderRadius: '6px', padding: '3px 8px', fontSize: '10px', fontWeight: 800, cursor: 'pointer',
});

export default function TradingViewDrawer({
  data = [],
  series = [{ key: 'value', name: 'Account value', type: 'area', color: theme.colors.primary }],
  markers = [],
  height = 280,
  timeKey = null,
  removeWeekendGaps = true,
  currency = 'USD',
  formatValue = null,
  showRanges = true,
  defaultRange = 'ALL',
  emptyMessage = 'No chart data recorded yet.',
  referenceLine = null,
  referenceTitle = 'Start',
  onDrill = null,
  className = '',
  style = {},
}) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const lcRef = useRef(null);
  const instancesRef = useRef([]);
  const markersRef = useRef({ list: [], data: [] });
  const drillRef = useRef(onDrill);
  const [ready, setReady] = useState(false);
  const [range, setRange] = useState(defaultRange);
  const [hover, setHover] = useState(null);
  const [showSMA, setShowSMA] = useState(false);
  const [showBenchmark, setShowBenchmark] = useState(true);
  const [showTrades, setShowTrades] = useState(true);
  drillRef.current = onDrill;

  const allData = useMemo(
    () => cleanTimeSeries(data, { timeKey, removeWeekends: removeWeekendGaps }),
    [data, timeKey, removeWeekendGaps],
  );
  const rangeDays = (RANGES.find((r) => r.id === range) || {}).days || 0;
  const cleanedData = useMemo(() => {
    const stepped = evenSteps(allData, rangeDays);
    // The benchmark starts where the account starts in the chosen range: what
    // the account's value then would be worth in the benchmark now. Drawn
    // from the very first reading, a strong or weak start set the two lines
    // far apart and flattened both on a short range.
    const key0 = series[0]?.key || 'value';
    const at = stepped[rangeStart(stepped, rangeDays)];
    if (!rangeDays || !at || !(at.benchmark > 0) || !(at[key0] > 0)) return stepped;
    const factor = at[key0] / at.benchmark;
    return stepped.map((p) => (typeof p.benchmark === 'number' ? { ...p, benchmark: p.benchmark * factor } : p));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allData, rangeDays, series[0]?.key]);
  const drawnRangeRef = useRef(null);
  // Signatures, so a poll that returns the same numbers in new arrays changes nothing.
  const primaryKey = series[0]?.key || 'value';
  const seriesKey = seriesSignature(series);
  const last = cleanedData[cleanedData.length - 1] || {};
  const dataKey = `${cleanedData.length}|${cleanedData[0]?.time}|${last.time}|${series.map((s) => last[s.key]).join(',')}`;
  const intraday = typeof cleanedData[0]?.time === 'number';

  const placedMarkers = useMemo(() => (markers || []).map((m) => {
    const bar = findClosestBar(cleanedData, m.time);
    if (!bar) return null;
    const isBuy = String(m.kind || m.side || '').toLowerCase().includes('buy');
    return {
      source: m,
      time: bar.time,
      position: m.position || (isBuy ? 'belowBar' : 'aboveBar'),
      shape: m.shape || (isBuy ? 'arrowUp' : 'arrowDown'),
      color: m.color || (isBuy ? theme.colors.primary : theme.colors.danger),
      text: m.text || `${String(m.side || 'trade').toUpperCase()} ${m.symbol || ''}`.trim(),
      id: m.id || `${m.symbol || 'trade'}-${bar.time}`,
    };
  }).filter(Boolean).sort((a, b) => timeSeconds(a.time) - timeSeconds(b.time)),
  // eslint-disable-next-line react-hooks/exhaustive-deps
  [JSON.stringify((markers || []).map((m) => [m.time, m.symbol, m.side, m.kind])), dataKey]);
  markersRef.current = { list: placedMarkers };

  const format = useCallback((val) => {
    if (typeof val !== 'number' || Number.isNaN(val)) return '—';
    if (formatValue) return formatValue(val);
    if (currency === 'KES') return `KES ${Math.round(val).toLocaleString()}`;
    if (currency === 'USD') {
      const digits = Math.abs(val) >= 1000 ? 0 : 2;
      return `$${val.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
    }
    return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }, [formatValue, currency]);
  const formatRef = useRef(format);
  formatRef.current = format;

  const timeLabel = useCallback((t) => {
    if (t == null) return '';
    if (typeof t === 'number') {
      const d = new Date(t * 1000);
      return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    return parseTime(t).toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
  }, []);

  // 1. The chart itself, once (and again only if its height changes).
  useEffect(() => {
    if (!containerRef.current || !cleanedData.length) return undefined;
    let disposed = false;
    (async () => {
      const lc = await import('lightweight-charts');
      if (disposed || !containerRef.current) return;
      lcRef.current = lc;
      const chart = lc.createChart(containerRef.current, {
        autoSize: true,
        layout: {
          background: { type: lc.ColorType.Solid, color: 'transparent' },
          textColor: theme.colors.textSecondary,
          fontSize: 10,
          fontFamily: 'JetBrains Mono, monospace',
          attributionLogo: true,
        },
        localization: { priceFormatter: (v) => formatRef.current(v) },
        grid: { vertLines: { visible: false }, horzLines: { color: 'rgba(51, 65, 85, 0.22)' } },
        rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.12, bottom: 0.08 } },
        timeScale: { borderVisible: false, rightOffset: 2, fixLeftEdge: true, fixRightEdge: true, minBarSpacing: 0.5 },
        crosshair: {
          mode: lc.CrosshairMode.Magnet,
          vertLine: { color: 'rgba(148, 163, 184, 0.45)', width: 1, style: lc.LineStyle.Dashed, labelBackgroundColor: '#1e293b' },
          horzLine: { color: 'rgba(148, 163, 184, 0.45)', width: 1, style: lc.LineStyle.Dashed, labelBackgroundColor: '#1e293b' },
        },
        handleScroll: { vertTouchDrag: false },
      });
      chart.subscribeCrosshairMove((param) => {
        if (!param || param.time == null || !param.seriesData) { setHover(null); return; }
        const values = {};
        instancesRef.current.forEach(({ key, instance }) => {
          const v = param.seriesData.get(instance);
          if (v && typeof v.value === 'number') values[key] = v.value;
        });
        setHover({ time: param.time, values });
      });
      chart.subscribeClick((param) => {
        if (!param || param.time == null || !drillRef.current) return;
        const hit = markersRef.current.list.find((m) => m.time === param.time && m.source.symbol);
        if (hit) drillRef.current(hit.source.symbol);
      });
      chartRef.current = chart;
      setReady(true);
    })();
    return () => {
      disposed = true;
      instancesRef.current = [];
      if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }
      setReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, cleanedData.length > 0]);

  // 2. Its series, redrawn when the data or the toggles change.
  useEffect(() => {
    const chart = chartRef.current;
    const lc = lcRef.current;
    if (!ready || !chart || !lc) return;
    const before = chart.timeScale().getVisibleLogicalRange();
    const firstDraw = instancesRef.current.length === 0;
    instancesRef.current.forEach(({ instance }) => chart.removeSeries(instance));
    instancesRef.current = [];
    chart.applyOptions({ timeScale: { timeVisible: intraday, secondsVisible: false } });

    series.forEach((s, idx) => {
      if (s.key === 'benchmark' && !showBenchmark) return;
      const color = s.color || (idx === 0 ? theme.colors.primary : theme.colors.secondary);
      const type = s.type || (idx === 0 ? 'area' : 'line');
      const common = {
        lineWidth: s.lineWidth || (type === 'line' ? 1.5 : 2),
        lineType: lc.LineType.Simple,
        priceLineVisible: idx === 0,
        lastValueVisible: true,
        crosshairMarkerRadius: 3,
      };
      let instance;
      if (type === 'area') {
        instance = chart.addSeries(lc.AreaSeries, {
          ...common, lineColor: color,
          topColor: s.topColor || `${color}40`, bottomColor: s.bottomColor || `${color}00`,
          priceLineColor: `${color}90`, priceLineStyle: lc.LineStyle.Dotted,
        });
      } else if (type === 'baseline') {
        instance = chart.addSeries(lc.BaselineSeries, {
          ...common, baseValue: { type: 'price', price: s.baseValue || 0 },
          topLineColor: s.topColor || theme.colors.primary, bottomLineColor: s.bottomColor || theme.colors.danger,
        });
      } else {
        instance = chart.addSeries(lc.LineSeries, {
          ...common, color, lineStyle: s.lineStyle === 'dashed' ? lc.LineStyle.Dashed : lc.LineStyle.Solid,
        });
      }
      instance.setData(cleanedData
        .filter((p) => typeof p[s.key] === 'number' && !Number.isNaN(p[s.key]))
        .map((p) => ({ time: p.time, value: p[s.key] })));
      instancesRef.current.push({ key: s.key, instance });
    });

    const primary = instancesRef.current[0]?.instance;
    if (primary && showSMA) {
      const sma = chart.addSeries(lc.LineSeries, {
        color: '#38bdf8', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
      });
      sma.setData(calculateSMA(cleanedData, primaryKey, 20));
      instancesRef.current.push({ key: '_sma', instance: sma });
    }
    if (primary && showTrades && placedMarkers.length) {
      // Labels overlap once trades crowd together; past a few, the arrows
      // alone are drawn and the header names the trade under the cursor.
      const labelled = placedMarkers.length <= 4;
      lc.createSeriesMarkers(primary, placedMarkers.map(({ source, ...m }) => (labelled ? m : { ...m, text: '' })));
    }
    if (primary && typeof referenceLine === 'number') {
      primary.createPriceLine({
        price: referenceLine, color: 'rgba(148, 163, 184, 0.5)', lineWidth: 1,
        lineStyle: lc.LineStyle.Dashed, axisLabelVisible: true, title: referenceTitle,
      });
    }
    // A new range (or the first drawing) frames that range; a refresh keeps
    // whatever the reader has zoomed or scrolled to.
    if (firstDraw || !before || drawnRangeRef.current !== range) applyRange(chart, cleanedData, range);
    else chart.timeScale().setVisibleLogicalRange(before);
    drawnRangeRef.current = range;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, dataKey, seriesKey, placedMarkers, showSMA, showBenchmark, showTrades, referenceLine]);

  // 3. The range buttons.
  useEffect(() => {
    if (ready && chartRef.current) applyRange(chartRef.current, cleanedData, range);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [range, ready]);

  if (!cleanedData.length) {
    return (
      <div className={className} style={{
        height, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center',
        color: theme.colors.textMuted, fontSize: '13px', padding: '16px', ...style,
      }}>
        {emptyMessage}
      </div>
    );
  }

  // What the header shows: the reading under the cursor, or the latest one.
  const index = hover ? cleanedData.findIndex((p) => p.time === hover.time) : cleanedData.length - 1;
  const point = cleanedData[index] || last;
  const valueOf = (key) => (hover && hover.values[key] != null ? hover.values[key] : point[key]);
  const spanDays = (timeSeconds(last.time) - timeSeconds(allData[0].time)) / DAY;
  const start = cleanedData[rangeStart(cleanedData, (RANGES.find((r) => r.id === range) || {}).days)];
  const startValue = start ? start[primaryKey] : null;
  const shownValue = valueOf(primaryKey);
  const change = typeof shownValue === 'number' && typeof startValue === 'number' && startValue
    ? (shownValue / startValue - 1) * 100 : null;
  const trade = placedMarkers.find((m) => m.time === point.time);

  return (
    <div className={className} style={{ position: 'relative', width: '100%', ...style }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px', flexWrap: 'wrap', marginBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', flexWrap: 'wrap', minHeight: '22px' }}>
          {series.map((s) => {
            if (s.key === 'benchmark' && !showBenchmark) return null;
            const v = valueOf(s.key);
            if (typeof v !== 'number') return null;
            return (
              <span key={s.key} style={{ display: 'inline-flex', alignItems: 'baseline', gap: '6px', fontSize: '11px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: s.color || theme.colors.primary, display: 'inline-block', alignSelf: 'center' }} />
                <span style={{ color: theme.colors.textMuted }}>{s.name}</span>
                <strong style={{ color: '#fff', fontSize: s.key === primaryKey ? '15px' : '12px', fontVariantNumeric: 'tabular-nums' }}>{format(v)}</strong>
              </span>
            );
          })}
          {change != null && (
            <span title={`Change from ${timeLabel(start.time)}`} style={{
              fontSize: '11px', fontWeight: 800, fontVariantNumeric: 'tabular-nums',
              color: change >= 0 ? theme.colors.primary : theme.colors.danger,
            }}>
              {change >= 0 ? '+' : ''}{change.toFixed(2)}% {range === 'ALL' ? 'since start' : `over ${range}`}
            </span>
          )}
          <span style={{ fontSize: '11px', color: theme.colors.textMuted }}>{timeLabel(point.time)}</span>
          {trade && (
            <button
              onClick={() => onDrill && trade.source.symbol && onDrill(trade.source.symbol)}
              style={{ ...pill(true, trade.color), cursor: onDrill && trade.source.symbol ? 'pointer' : 'default' }}
            >
              {trade.text}{onDrill && trade.source.symbol ? ' · open' : ''}
            </button>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap' }}>
          <button onClick={() => setShowSMA(!showSMA)} title="20-point moving average" style={pill(showSMA, '#38bdf8')}>Average</button>
          {series.some((s) => s.key === 'benchmark') && (
            <button onClick={() => setShowBenchmark(!showBenchmark)} title="Show the benchmark" style={pill(showBenchmark, theme.colors.secondary)}>Benchmark</button>
          )}
          {placedMarkers.length > 0 && (
            <button onClick={() => setShowTrades(!showTrades)} title="Show the agent's trades" style={pill(showTrades, theme.colors.primary)}>
              Trades {placedMarkers.length}
            </button>
          )}
          {showRanges && (
            <div role="group" aria-label="Time range" style={{ display: 'flex', gap: '2px', marginLeft: '4px', padding: '2px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)' }}>
              {RANGES.map((r) => {
                const tooLong = r.days && r.days > spanDays * 1.5 && r.days > 1;
                return (
                  <button
                    key={r.id}
                    onClick={() => setRange(r.id)}
                    disabled={!!tooLong}
                    aria-pressed={range === r.id}
                    title={tooLong ? 'Longer than the history recorded so far' : `Last ${r.days ? `${r.days} days` : 'everything'}`}
                    style={{
                      background: range === r.id ? 'rgba(255,255,255,0.14)' : 'transparent',
                      color: range === r.id ? '#fff' : theme.colors.textMuted,
                      opacity: tooLong ? 0.35 : 1, border: 'none', borderRadius: '6px',
                      padding: '3px 7px', fontSize: '10px', fontWeight: 800, cursor: tooLong ? 'default' : 'pointer',
                    }}
                  >
                    {r.label}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
      <div ref={containerRef} style={{ height, width: '100%', position: 'relative' }} />
    </div>
  );
}

function applyRange(chart, data, rangeId) {
  if (!chart || !data.length) return;
  const days = (RANGES.find((r) => r.id === rangeId) || {}).days;
  const from = rangeStart(data, days);
  if (!from) chart.timeScale().fitContent();
  else chart.timeScale().setVisibleLogicalRange({ from, to: data.length - 1 });
}
