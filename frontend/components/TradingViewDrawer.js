// TradingView Drawer: A unified, professional financial chart drawer powered by
// TradingView's Lightweight Charts (Apache 2.0). Replaces standard linear/calendar charts
// across Overview, Portfolio, NSE, and Analytics views.
//
// Key Features:
// - Strictly removes weekend gaps (Saturdays/Sundays & non-trading intervals)
// - Organic spline curvature (lc.LineType.Curved) eliminating shallow/polygonal look
// - Full-width edge anchoring (fixLeftEdge & fixRightEdge) with zero empty right margin
// - Preserves high-density intraday/hourly resolution without collapsing points
// - Trade execution markers (BUY/SELL/STOP) pinned directly to the curve with click-to-drilldown
// - Full timeframe drilldown pills: 1D, 1W, 1M, 3M, 6M, 1Y, ALL (always visible)
// - Feature indicator toggles: SMA Trend, Benchmark overlay, Trade markers
// - Dotted live price line with badge tag on the right axis
// - Interactive crosshair HUD with formatted metrics, session returns, and active trade drilldown
import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { theme } from './DashboardStyles';

export const RANGES = [
  { id: '1D', days: 1, label: '1D' },
  { id: '1W', days: 7, label: '1W' },
  { id: '1M', days: 22, label: '1M' },
  { id: '3M', days: 66, label: '3M' },
  { id: '6M', days: 130, label: '6M' },
  { id: '1Y', days: 252, label: '1Y' },
  { id: 'ALL', days: 0, label: 'ALL' }
];

/**
 * Clean and sort time series data, strictly removing weekend gaps (Sat/Sun).
 * Preserves granular intraday timestamps if present; handles daily dates cleanly.
 */
export function cleanTimeSeries(rawData = [], options = {}) {
  const {
    timeKey = null,
    removeWeekends = true,
  } = options;

  if (!Array.isArray(rawData) || rawData.length === 0) return [];

  const parsed = [];
  for (const item of rawData) {
    if (!item) continue;
    const rawTime = (timeKey && item[timeKey]) || item.timestamp || item.day || item.time || item.date;
    if (!rawTime) continue;

    let d;
    let isPureDateString = false;
    if (typeof rawTime === 'string') {
      if (/^\d{4}-\d{2}-\d{2}$/.test(rawTime)) {
        isPureDateString = true;
        const [y, m, day] = rawTime.split('-').map(Number);
        d = new Date(Date.UTC(y, m - 1, day, 12, 0, 0));
      } else {
        d = new Date(rawTime.endsWith('Z') || rawTime.includes('+') ? rawTime : `${rawTime}Z`);
      }
    } else if (typeof rawTime === 'number') {
      d = new Date(rawTime < 2e10 ? rawTime * 1000 : rawTime);
    } else if (rawTime instanceof Date) {
      d = rawTime;
    }

    if (!d || Number.isNaN(d.getTime())) continue;

    // Filter out weekend days if enabled (0 = Sunday, 6 = Saturday)
    // Forex/Asian open starts Sunday ~21:00 UTC
    const uDay = d.getUTCDay();
    const uHour = d.getUTCHours();
    if (removeWeekends && (uDay === 6 || (uDay === 0 && uHour < 21))) {
      continue;
    }

    parsed.push({
      date: d,
      sec: Math.floor(d.getTime() / 1000),
      dateStr: d.toISOString().slice(0, 10),
      isPureDateString,
      item
    });
  }

  if (parsed.length === 0) return [];

  // Sort strictly ascending by time
  parsed.sort((a, b) => a.sec - b.sec);

  // Check whether data has granular intraday timestamps (e.g. hourly or minute snapshots)
  const allPureDate = parsed.every((p) => p.isPureDateString);
  const daySet = new Set(parsed.map((p) => p.dateStr));
  const hasIntraday = !allPureDate && (parsed.length > daySet.size || parsed.some((p) => p.date.getUTCHours() !== 0 || p.date.getUTCMinutes() !== 0));

  if (hasIntraday) {
    // Intraday: Use UNIX timestamp in seconds with strictly increasing seconds
    let lastSec = -1;
    const out = [];
    for (const p of parsed) {
      let sec = p.sec;
      if (sec <= lastSec) sec = lastSec + 1;
      lastSec = sec;
      out.push({ time: sec, ...p.item });
    }
    return out;
  }

  // Pure daily: Deduplicate to latest point per calendar day (YYYY-MM-DD)
  const dayMap = new Map();
  for (const p of parsed) {
    dayMap.set(p.dateStr, p.item);
  }
  const sortedDays = Array.from(dayMap.keys()).sort();
  return sortedDays.map((t) => ({ time: t, ...dayMap.get(t) }));
}

/**
 * Match an external timestamp (e.g. trade execution) to the closest bar on the chart
 */
export function findClosestBar(cleanedData, rawTime) {
  if (!cleanedData || !cleanedData.length) return null;
  if (!rawTime) return cleanedData[cleanedData.length - 1];

  let targetSec = 0;
  let targetDateStr = '';
  if (typeof rawTime === 'string') {
    targetDateStr = rawTime.slice(0, 10);
    const d = new Date(rawTime.endsWith('Z') || rawTime.includes('+') ? rawTime : `${rawTime}Z`);
    if (!Number.isNaN(d.getTime())) targetSec = Math.floor(d.getTime() / 1000);
  } else if (typeof rawTime === 'number') {
    targetSec = rawTime < 2e10 ? Math.floor(rawTime) : Math.floor(rawTime / 1000);
    targetDateStr = new Date(targetSec * 1000).toISOString().slice(0, 10);
  }

  // If chart uses string dates (daily bars)
  if (typeof cleanedData[0].time === 'string') {
    const directMatch = cleanedData.find((b) => b.time === targetDateStr);
    if (directMatch) return directMatch;
    const past = cleanedData.filter((b) => b.time <= targetDateStr);
    if (past.length > 0) return past[past.length - 1];
    return cleanedData[0];
  }

  // If chart uses UNIX timestamp seconds (intraday)
  let closest = cleanedData[0];
  let minDiff = Math.abs(cleanedData[0].time - targetSec);
  for (const b of cleanedData) {
    const diff = Math.abs(b.time - targetSec);
    if (diff < minDiff) {
      minDiff = diff;
      closest = b;
    }
  }
  return closest;
}

/**
 * Calculate simple moving average over valid numeric values
 */
export function calculateSMA(data, key, period = 20) {
  const valid = [];
  for (let i = 0; i < data.length; i++) {
    const val = data[i][key];
    if (typeof val === 'number' && !Number.isNaN(val)) {
      valid.push({ time: data[i].time, val });
    }
  }
  if (valid.length < 3) return [];
  const effPeriod = Math.min(period, Math.max(2, Math.floor(valid.length / 2)));
  const result = [];
  let sum = 0;
  for (let i = 0; i < valid.length; i++) {
    sum += valid[i].val;
    if (i >= effPeriod) {
      sum -= valid[i - effPeriod].val;
    }
    if (i >= effPeriod - 1) {
      result.push({ time: valid[i].time, value: +(sum / effPeriod).toFixed(2) });
    }
  }
  return result;
}

export default function TradingViewDrawer({
  data = [],
  series = [
    { key: 'value', name: 'Account value', type: 'area', color: theme.colors.primary }
  ],
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
  referenceTitle = 'Baseline',
  onDrill = null,
  className = '',
  style = {}
}) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesInstancesRef = useRef([]);
  const [range, setRange] = useState(defaultRange);
  const [hover, setHover] = useState(null);
  const [showSMA, setShowSMA] = useState(false);
  const [showBenchmark, setShowBenchmark] = useState(true);
  const [showTrades, setShowTrades] = useState(true);

  // Clean data, preserve intraday density, and eliminate weekend gaps
  const cleanedData = useMemo(() => {
    return cleanTimeSeries(data, { timeKey, removeWeekends: removeWeekendGaps });
  }, [data, timeKey, removeWeekendGaps]);

  // Default value formatter based on currency or generic numbers
  const defaultFormatter = useCallback((val) => {
    if (typeof val !== 'number' || Number.isNaN(val)) return '—';
    if (formatValue) return formatValue(val);
    if (currency === 'KES') {
      return `KES ${Math.round(val).toLocaleString()}`;
    }
    if (currency === 'USD') {
      return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }, [formatValue, currency]);

  // Format date display for HUD
  const formatTimeLabel = useCallback((t) => {
    if (!t) return '';
    if (typeof t === 'number') {
      const d = new Date(t * 1000);
      return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    return String(t);
  }, []);

  // Check if benchmark series is configured
  const hasBenchmarkConfig = useMemo(() => {
    return series.some((s) => s.key === 'benchmark');
  }, [series]);

  // Primary key for account value
  const primaryKey = series[0]?.key || 'value';

  useEffect(() => {
    if (!cleanedData.length || !containerRef.current) return undefined;
    let disposed = false;
    let chart = null;

    (async () => {
      const lc = await import('lightweight-charts');
      if (disposed || !containerRef.current) return;

      chart = lc.createChart(containerRef.current, {
        autoSize: true,
        layout: {
          background: { type: lc.ColorType?.Solid || 'solid', color: 'transparent' },
          textColor: theme.colors.textSecondary,
          fontSize: 10,
          attributionLogo: true,
          panes: { separatorColor: 'rgba(51, 65, 85, 0.4)' },
        },
        grid: {
          vertLines: { color: 'rgba(51, 65, 85, 0.18)' },
          horzLines: { color: 'rgba(51, 65, 85, 0.18)' }
        },
        rightPriceScale: {
          borderColor: 'rgba(51, 65, 85, 0.5)',
          scaleMargins: { top: 0.16, bottom: 0.16 },
          autoScale: true,
        },
        timeScale: {
          borderColor: 'rgba(51, 65, 85, 0.5)',
          rightOffset: 0,
          fixLeftEdge: true,
          fixRightEdge: true,
          lockVisibleTimeRangeOnResize: true,
          barSpacing: 10,
          minBarSpacing: 1,
        },
        crosshair: {
          mode: lc.CrosshairMode.Normal,
          vertLine: { color: 'rgba(148, 163, 184, 0.4)', width: 1, style: lc.LineStyle.Dashed },
          horzLine: { color: 'rgba(148, 163, 184, 0.4)', width: 1, style: lc.LineStyle.Dashed },
        },
      });

      const instances = [];

      series.forEach((sConf, idx) => {
        // If benchmark is toggled off, skip adding it
        if (sConf.key === 'benchmark' && !showBenchmark) return;

        const color = sConf.color || (idx === 0 ? theme.colors.primary : theme.colors.secondary);
        const lineStyle = sConf.lineStyle === 'dashed' ? lc.LineStyle.Dashed : lc.LineStyle.Solid;
        const sType = sConf.type || (idx === 0 ? 'area' : 'line');

        // Organic spline curvature (Curved) to eliminate shallow/polygonal look
        const lineType = sConf.lineType === 'simple' ? lc.LineType.Simple : lc.LineType.Curved;

        let sInstance;
        if (sType === 'area') {
          sInstance = chart.addSeries(lc.AreaSeries, {
            lineColor: color,
            topColor: sConf.topColor || `${color}38`,
            bottomColor: sConf.bottomColor || `${color}02`,
            lineWidth: sConf.lineWidth || 2.5,
            lineType,
            priceLineVisible: true,
            priceLineColor: `${color}90`,
            priceLineStyle: lc.LineStyle.Dotted,
            lastValueVisible: true,
            crosshairMarkerVisible: true,
            crosshairMarkerRadius: 4,
          });
        } else if (sType === 'baseline') {
          sInstance = chart.addSeries(lc.BaselineSeries, {
            baseValue: { type: 'price', price: sConf.baseValue || 0 },
            topLineColor: sConf.topColor || theme.colors.primary,
            bottomLineColor: sConf.bottomColor || theme.colors.danger,
            lineWidth: sConf.lineWidth || 2.5,
            lineType,
            priceLineVisible: true,
            lastValueVisible: true,
          });
        } else {
          sInstance = chart.addSeries(lc.LineSeries, {
            color,
            lineWidth: sConf.lineWidth || 2,
            lineStyle,
            lineType,
            priceLineVisible: false,
            lastValueVisible: true,
            crosshairMarkerVisible: true,
            crosshairMarkerRadius: 3,
          });
        }

        // Filter points for this specific series
        const seriesData = cleanedData
          .filter((p) => typeof p[sConf.key] === 'number' && !Number.isNaN(p[sConf.key]))
          .map((p) => ({ time: p.time, value: p[sConf.key] }));

        if (seriesData.length > 0) {
          sInstance.setData(seriesData);
        }

        instances.push({ conf: sConf, instance: sInstance, data: seriesData });
      });

      // Optional SMA Trend Overlay (Curved spline)
      if (showSMA && cleanedData.length >= 3 && instances.length > 0) {
        try {
          const smaData = calculateSMA(cleanedData, primaryKey, 20);
          if (smaData.length > 0) {
            const smaInstance = chart.addSeries(lc.LineSeries, {
              color: '#38bdf8',
              lineWidth: 1.5,
              lineType: lc.LineType.Curved,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: true,
              crosshairMarkerRadius: 2,
            });
            smaInstance.setData(smaData);
            instances.push({ conf: { key: '_sma', name: 'SMA 20', color: '#38bdf8' }, instance: smaInstance, data: smaData });
          }
        } catch (e) {
          console.debug('Failed to add SMA overlay:', e);
        }
      }

      // Trade Markers on the Primary Curve (e.g. BUY, SELL, STOP) with Symbol Drilldown
      if (showTrades && markers && markers.length > 0 && instances.length > 0) {
        try {
          const mappedMarkers = markers.map((m) => {
            const bar = findClosestBar(cleanedData, m.time);
            const isBuy = (m.kind || m.side || '').toLowerCase().includes('buy');
            return {
              time: bar ? bar.time : cleanedData[cleanedData.length - 1].time,
              position: m.position || (isBuy ? 'belowBar' : 'aboveBar'),
              shape: m.shape || (isBuy ? 'arrowUp' : 'arrowDown'),
              color: m.color || (isBuy ? theme.colors.primary : theme.colors.danger),
              text: m.text || `${(m.side || 'TRADE').toUpperCase()} ${m.symbol || ''}`.trim(),
              id: m.id || `${m.symbol || 'trade'}-${bar ? bar.time : 'last'}`
            };
          }).sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));

          lc.createSeriesMarkers(instances[0].instance, mappedMarkers);
        } catch (e) {
          console.debug('Failed to place markers on chart:', e);
        }
      }

      // Baseline reference price line if provided
      if (referenceLine != null && instances.length > 0 && typeof referenceLine === 'number') {
        try {
          instances[0].instance.createPriceLine({
            price: referenceLine,
            color: 'rgba(148, 163, 184, 0.45)',
            lineWidth: 1,
            lineStyle: lc.LineStyle.Dashed,
            axisLabelVisible: true,
            title: referenceTitle,
          });
        } catch (e) {
          console.debug('Failed to add reference price line:', e);
        }
      }

      seriesInstancesRef.current = instances;

      // Subscribe crosshair move for interactive HUD
      chart.subscribeCrosshairMove((param) => {
        if (!param || !param.time) {
          setHover(null);
          return;
        }
        const values = {};
        instances.forEach((item) => {
          const valObj = param.seriesData.get(item.instance);
          if (valObj && typeof valObj.value === 'number') {
            values[item.conf.key] = valObj.value;
          }
        });
        setHover({ time: param.time, values });
      });

      // Subscribe click on chart canvas: click marker to drill down into symbol
      chart.subscribeClick((param) => {
        if (!param || !param.time || !onDrill || !markers.length) return;
        const clickedTime = param.time;
        const matchingMarker = markers.find((m) => {
          const bar = findClosestBar(cleanedData, m.time);
          return bar && bar.time === clickedTime && m.symbol;
        });
        if (matchingMarker && matchingMarker.symbol) {
          onDrill(matchingMarker.symbol);
        }
      });

      chartRef.current = chart;
      applyTimeRange(chart, cleanedData.length, range);

      // Edge-to-edge layout guarantee on container resize
      const ro = new ResizeObserver(() => {
        if (!disposed && chart) {
          applyTimeRange(chart, cleanedData.length, range);
        }
      });
      if (containerRef.current) {
        ro.observe(containerRef.current);
      }

      return () => {
        ro.disconnect();
      };
    })();

    return () => {
      disposed = true;
      chartRef.current = null;
      seriesInstancesRef.current = [];
      if (chart) {
        chart.remove();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cleanedData, series, markers, height, referenceLine, showSMA, showBenchmark, showTrades]);

  // Adjust zoom when range changes
  useEffect(() => {
    if (chartRef.current) {
      applyTimeRange(chartRef.current, cleanedData.length, range);
    }
  }, [range, cleanedData.length]);

  if (!cleanedData.length) {
    return (
      <div style={{
        height, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: theme.colors.textMuted, fontSize: '13px', textAlign: 'center', padding: '16px',
        ...style
      }} className={className}>
        {emptyMessage}
      </div>
    );
  }

  // Active values (hover or last point)
  const lastPoint = cleanedData[cleanedData.length - 1] || {};
  const prevPoint = cleanedData[cleanedData.length - 2] || null;
  const activeTime = hover ? hover.time : lastPoint.time;
  const activePrimaryVal = hover ? hover.values[primaryKey] : lastPoint[primaryKey];
  const prevPrimaryVal = prevPoint ? prevPoint[primaryKey] : null;
  const primaryChange = (typeof activePrimaryVal === 'number' && typeof prevPrimaryVal === 'number' && prevPrimaryVal > 0)
    ? ((activePrimaryVal - prevPrimaryVal) / prevPrimaryVal) * 100
    : null;

  // Active trade marker matching current hovered/active bar
  const activeTrade = useMemo(() => {
    if (!markers.length || !activeTime) return null;
    return markers.find((m) => {
      const bar = findClosestBar(cleanedData, m.time);
      return bar && bar.time === activeTime;
    }) || null;
  }, [markers, activeTime, cleanedData]);

  return (
    <div style={{ position: 'relative', width: '100%', ...style }} className={className}>
      {/* Top Header / HUD Bar */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        gap: '8px', flexWrap: 'wrap', marginBottom: '8px', fontSize: '11px'
      }}>
        {/* Active Point Badges & HUD */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          {activeTime && (
            <span style={{ color: theme.colors.textSecondary, fontWeight: 700, fontFamily: 'monospace' }}>
              {formatTimeLabel(activeTime)}
            </span>
          )}

          {series.map((sConf) => {
            if (sConf.key === 'benchmark' && !showBenchmark) return null;
            const val = hover ? hover.values[sConf.key] : lastPoint[sConf.key];
            if (typeof val !== 'number') return null;
            return (
              <div key={sConf.key} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <span style={{
                  width: '8px', height: '8px', borderRadius: '50%',
                  backgroundColor: sConf.color || theme.colors.primary,
                  display: 'inline-block'
                }} />
                <span style={{ color: theme.colors.textMuted }}>{sConf.name}:</span>
                <strong style={{ color: '#fff', fontFamily: 'monospace' }}>
                  {defaultFormatter(val)}
                </strong>
              </div>
            );
          })}

          {showSMA && hover?.values?._sma != null && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#38bdf8', display: 'inline-block' }} />
              <span style={{ color: theme.colors.textMuted }}>SMA 20:</span>
              <strong style={{ color: '#38bdf8', fontFamily: 'monospace' }}>{defaultFormatter(hover.values._sma)}</strong>
            </div>
          )}

          {primaryChange != null && (
            <span style={{
              fontSize: '10px', fontWeight: 800, padding: '1px 6px', borderRadius: '4px',
              backgroundColor: primaryChange >= 0 ? 'rgba(16,185,129,0.15)' : 'rgba(244,63,94,0.15)',
              color: primaryChange >= 0 ? theme.colors.primary : theme.colors.danger,
              border: `1px solid ${primaryChange >= 0 ? theme.colors.primary : theme.colors.danger}40`
            }}>
              {primaryChange >= 0 ? `↗ +${primaryChange.toFixed(2)}%` : `↘ ${primaryChange.toFixed(2)}%`}
            </span>
          )}

          {/* Drilldown interactive pill for trade at this bar */}
          {activeTrade && (
            <button
              onClick={() => onDrill && activeTrade.symbol && onDrill(activeTrade.symbol)}
              title={activeTrade.symbol ? `Drill down into ${activeTrade.symbol}` : ''}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: '4px',
                padding: '2px 8px', borderRadius: '5px',
                backgroundColor: (activeTrade.kind || activeTrade.side || '').toLowerCase().includes('buy')
                  ? 'rgba(16,185,129,0.2)'
                  : 'rgba(244,63,94,0.2)',
                color: (activeTrade.kind || activeTrade.side || '').toLowerCase().includes('buy')
                  ? theme.colors.primary
                  : theme.colors.danger,
                border: `1px solid ${(activeTrade.kind || activeTrade.side || '').toLowerCase().includes('buy') ? theme.colors.primary : theme.colors.danger}50`,
                fontSize: '11px', fontWeight: 800, cursor: onDrill && activeTrade.symbol ? 'pointer' : 'default',
              }}
            >
              <span>{activeTrade.text || `${activeTrade.side || 'TRADE'} ${activeTrade.symbol || ''}`}</span>
              {onDrill && activeTrade.symbol && <span style={{ fontSize: '9px', opacity: 0.85 }}>↗ DRILLDOWN</span>}
            </button>
          )}
        </div>

        {/* Feature Toggles & Drilldown Timeframe Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
          {/* Indicator toggles like Apple chart */}
          <div style={{ display: 'flex', gap: '3px' }}>
            <button
              onClick={() => setShowSMA(!showSMA)}
              title="Toggle 20-period moving average curve"
              style={{
                background: showSMA ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: showSMA ? '#38bdf8' : theme.colors.textMuted,
                border: `1px solid ${showSMA ? '#38bdf8' : 'rgba(255,255,255,0.08)'}`,
                borderRadius: '5px', padding: '2px 7px', fontSize: '10px', fontWeight: 800, cursor: 'pointer',
              }}
            >
              SMA 20
            </button>

            {hasBenchmarkConfig && (
              <button
                onClick={() => setShowBenchmark(!showBenchmark)}
                title="Toggle benchmark curve"
                style={{
                  background: showBenchmark ? `${theme.colors.secondary}25` : 'transparent',
                  color: showBenchmark ? theme.colors.secondary : theme.colors.textMuted,
                  border: `1px solid ${showBenchmark ? theme.colors.secondary : 'rgba(255,255,255,0.08)'}`,
                  borderRadius: '5px', padding: '2px 7px', fontSize: '10px', fontWeight: 800, cursor: 'pointer',
                }}
              >
                Benchmark
              </button>
            )}

            {markers.length > 0 && (
              <button
                onClick={() => setShowTrades(!showTrades)}
                title="Toggle trade execution markers on chart"
                style={{
                  background: showTrades ? `${theme.colors.primary}25` : 'transparent',
                  color: showTrades ? theme.colors.primary : theme.colors.textMuted,
                  border: `1px solid ${showTrades ? theme.colors.primary : 'rgba(255,255,255,0.08)'}`,
                  borderRadius: '5px', padding: '2px 7px', fontSize: '10px', fontWeight: 800, cursor: 'pointer',
                }}
              >
                Trades ({markers.length})
              </button>
            )}
          </div>

          <div style={{ width: '1px', height: '14px', backgroundColor: 'rgba(255,255,255,0.12)', margin: '0 2px' }} />

          {/* Timeframe Drilldown Buttons (1D, 1W, 1M, 3M, 6M, 1Y, ALL) */}
          {showRanges && (
            <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
              {RANGES.map((r) => (
                <button
                  key={r.id}
                  onClick={() => setRange(r.id)}
                  style={{
                    background: range === r.id ? 'rgba(255,255,255,0.16)' : 'transparent',
                    color: range === r.id ? '#fff' : theme.colors.textMuted,
                    border: `1px solid ${range === r.id ? theme.colors.border : 'rgba(255,255,255,0.05)'}`,
                    borderRadius: '5px',
                    padding: '2px 7px',
                    fontSize: '10px',
                    fontWeight: 700,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                >
                  {r.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Chart Canvas Container */}
      <div ref={containerRef} style={{ height, width: '100%', position: 'relative' }} />
    </div>
  );
}

function applyTimeRange(chart, count, rangeId) {
  if (!chart || !count) return;
  const match = RANGES.find((r) => r.id === rangeId);
  const days = match ? match.days : 0;
  const ts = chart.timeScale();
  if (!days || count <= days) {
    ts.fitContent();
  } else {
    ts.setVisibleLogicalRange({ from: Math.max(0, count - days), to: count - 1 });
  }
}
