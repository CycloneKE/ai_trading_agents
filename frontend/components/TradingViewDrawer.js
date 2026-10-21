// TradingView Drawer: A unified, high-performance financial chart drawer powered by
// TradingView's Lightweight Charts (Apache 2.0). Replaces standard linear/calendar charts
// across Overview, Portfolio, NSE, and Analytics views.
//
// Key Features:
// - Strictly removes weekend gaps (Saturdays/Sundays & non-trading intervals)
// - Supports Area, Line, and Baseline series with customizable colors & gradients
// - Multi-series support (e.g. Account Equity Area + Benchmark Lines)
// - Built-in interactive crosshair with real-time HUD metrics
// - Built-in range toggles (1M, 3M, 6M, 1Y, ALL)
// - Seamless auto-resizing and dark theme styling
import React, { useEffect, useRef, useState, useMemo } from 'react';
import { theme } from './DashboardStyles';

const RANGES = [
  { id: '1M', days: 22 },
  { id: '3M', days: 66 },
  { id: '6M', days: 130 },
  { id: '1Y', days: 252 },
  { id: 'ALL', days: 0 }
];

/**
 * Clean and sort time series data, strictly removing weekend gaps (Sat/Sun).
 * Deduplicates to the latest close per trading day to maintain monotonic order.
 */
export function cleanTimeSeries(rawData = [], options = {}) {
  const {
    timeKey = null,
    removeWeekends = true,
    seriesKeys = ['value'],
  } = options;

  if (!Array.isArray(rawData) || rawData.length === 0) return [];

  const parsed = [];
  for (const item of rawData) {
    if (!item) continue;
    const rawTime = (timeKey && item[timeKey]) || item.timestamp || item.day || item.time || item.date;
    if (!rawTime) continue;

    let d;
    if (typeof rawTime === 'string') {
      if (/^\d{4}-\d{2}-\d{2}$/.test(rawTime)) {
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
    const dayOfWeek = d.getUTCDay();
    if (removeWeekends && (dayOfWeek === 0 || dayOfWeek === 6)) {
      continue;
    }

    parsed.push({ date: d, item });
  }

  if (parsed.length === 0) return [];

  // Sort strictly ascending
  parsed.sort((a, b) => a.date.getTime() - b.date.getTime());

  // Count unique calendar days
  const daySet = new Set(parsed.map((p) => p.date.toISOString().slice(0, 10)));

  if (daySet.size > 1) {
    // Multi-day: 1 consolidated point per trading day (YYYY-MM-DD)
    const dayMap = new Map();
    for (const p of parsed) {
      const dayStr = p.date.toISOString().slice(0, 10);
      dayMap.set(dayStr, p.item);
    }
    const sortedDays = Array.from(dayMap.keys()).sort();
    return sortedDays.map((t) => ({ time: t, ...dayMap.get(t) }));
  }

  // Single-day intraday: unix timestamp in seconds with strictly increasing seconds
  let lastSec = -1;
  const out = [];
  for (const p of parsed) {
    let sec = Math.floor(p.date.getTime() / 1000);
    if (sec <= lastSec) sec = lastSec + 1;
    lastSec = sec;
    out.push({ time: sec, ...p.item });
  }
  return out;
}

export default function TradingViewDrawer({
  data = [],
  series = [
    { key: 'value', name: 'Account value', type: 'area', color: theme.colors.primary }
  ],
  height = 260,
  timeKey = null,
  removeWeekendGaps = true,
  currency = 'USD',
  formatValue = null,
  showRanges = true,
  defaultRange = 'ALL',
  emptyMessage = 'No chart data recorded yet.',
  referenceLine = null,
  referenceTitle = 'Baseline',
  className = '',
  style = {}
}) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesInstancesRef = useRef([]);
  const [range, setRange] = useState(defaultRange);
  const [hover, setHover] = useState(null);

  // Clean data and eliminate weekend gaps
  const cleanedData = useMemo(() => {
    const keys = series.map((s) => s.key);
    return cleanTimeSeries(data, { timeKey, removeWeekends: removeWeekendGaps, seriesKeys: keys });
  }, [data, timeKey, removeWeekendGaps, series]);

  // Default value formatter based on currency or generic numbers
  const defaultFormatter = (val) => {
    if (typeof val !== 'number' || Number.isNaN(val)) return '—';
    if (formatValue) return formatValue(val);
    if (currency === 'KES') {
      return `KES ${Math.round(val).toLocaleString()}`;
    }
    if (currency === 'USD') {
      return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };

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
          vertLines: { color: 'rgba(51, 65, 85, 0.2)' },
          horzLines: { color: 'rgba(51, 65, 85, 0.2)' }
        },
        rightPriceScale: {
          borderColor: 'rgba(51, 65, 85, 0.5)',
          scaleMargins: { top: 0.1, bottom: 0.1 },
        },
        timeScale: {
          borderColor: 'rgba(51, 65, 85, 0.5)',
          rightOffset: 3,
          barSpacing: 8,
          minBarSpacing: 3,
        },
        crosshair: {
          mode: lc.CrosshairMode.Normal,
          vertLine: { color: 'rgba(148, 163, 184, 0.4)', width: 1, style: lc.LineStyle.Dashed },
          horzLine: { color: 'rgba(148, 163, 184, 0.4)', width: 1, style: lc.LineStyle.Dashed },
        },
      });

      const instances = [];

      series.forEach((sConf, idx) => {
        const color = sConf.color || (idx === 0 ? theme.colors.primary : theme.colors.secondary);
        const lineStyle = sConf.lineStyle === 'dashed' ? lc.LineStyle.Dashed : lc.LineStyle.Solid;
        const sType = sConf.type || (idx === 0 ? 'area' : 'line');

        let sInstance;
        if (sType === 'area') {
          sInstance = chart.addSeries(lc.AreaSeries, {
            lineColor: color,
            topColor: sConf.topColor || `${color}38`,
            bottomColor: sConf.bottomColor || `${color}00`,
            lineWidth: sConf.lineWidth || 2,
            priceLineVisible: false,
            lastValueVisible: true,
            crosshairMarkerVisible: true,
          });
        } else if (sType === 'baseline') {
          sInstance = chart.addSeries(lc.BaselineSeries, {
            baseValue: { type: 'price', price: sConf.baseValue || 0 },
            topLineColor: sConf.topColor || theme.colors.primary,
            bottomLineColor: sConf.bottomColor || theme.colors.danger,
            lineWidth: sConf.lineWidth || 2,
            priceLineVisible: false,
            lastValueVisible: true,
          });
        } else {
          sInstance = chart.addSeries(lc.LineSeries, {
            color,
            lineWidth: sConf.lineWidth || 1.5,
            lineStyle,
            priceLineVisible: false,
            lastValueVisible: true,
            crosshairMarkerVisible: true,
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

      // Add reference baseline if provided
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

      chartRef.current = chart;
      applyTimeRange(chart, cleanedData.length, range);
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
  }, [cleanedData, series, height, referenceLine]);

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

  // Derive active values (hover or last point)
  const lastPoint = cleanedData[cleanedData.length - 1] || {};
  const prevPoint = cleanedData[cleanedData.length - 2] || null;
  const activeTime = hover ? hover.time : lastPoint.time;
  const primaryKey = series[0]?.key;
  const activePrimaryVal = hover ? hover.values[primaryKey] : lastPoint[primaryKey];
  const prevPrimaryVal = prevPoint ? prevPoint[primaryKey] : null;
  const primaryChange = (typeof activePrimaryVal === 'number' && typeof prevPrimaryVal === 'number' && prevPrimaryVal > 0)
    ? ((activePrimaryVal - prevPrimaryVal) / prevPrimaryVal) * 100
    : null;

  return (
    <div style={{ position: 'relative', width: '100%', ...style }} className={className}>
      {/* Top Header / HUD Bar */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        gap: '8px', flexWrap: 'wrap', marginBottom: '8px', fontSize: '11px'
      }}>
        {/* Active Point Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          {activeTime && (
            <span style={{ color: theme.colors.textSecondary, fontWeight: 700, fontFamily: 'monospace' }}>
              {String(activeTime)}
            </span>
          )}
          {series.map((sConf) => {
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
          {primaryChange != null && (
            <span style={{
              fontSize: '10px', fontWeight: 800, padding: '1px 5px', borderRadius: '4px',
              backgroundColor: primaryChange >= 0 ? 'rgba(16,185,129,0.15)' : 'rgba(244,63,94,0.15)',
              color: primaryChange >= 0 ? theme.colors.primary : theme.colors.danger,
              border: `1px solid ${primaryChange >= 0 ? theme.colors.primary : theme.colors.danger}40`
            }}>
              {primaryChange >= 0 ? `↗ +${primaryChange.toFixed(2)}%` : `↘ ${primaryChange.toFixed(2)}%`}
            </span>
          )}
        </div>

        {/* Range Buttons (1M, 3M, 6M, 1Y, ALL) */}
        {showRanges && cleanedData.length > 10 && (
          <div style={{ display: 'flex', gap: '3px' }}>
            {RANGES.map((r) => (
              <button
                key={r.id}
                onClick={() => setRange(r.id)}
                style={{
                  background: range === r.id ? 'rgba(255,255,255,0.12)' : 'transparent',
                  color: range === r.id ? '#fff' : theme.colors.textMuted,
                  border: `1px solid ${range === r.id ? theme.colors.border : 'transparent'}`,
                  borderRadius: '4px',
                  padding: '2px 7px',
                  fontSize: '10px',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                {r.id}
              </button>
            ))}
          </div>
        )}
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
    ts.setVisibleLogicalRange({ from: count - days, to: count + 2 });
  }
}
