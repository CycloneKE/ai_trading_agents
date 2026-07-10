import { useState, useEffect } from 'react';
import { X, ShieldAlert, Award, TrendingUp, TrendingDown, Layers } from 'lucide-react';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

const Stat = ({ label, value, color }) => (
  <div style={{ flex: 1, minWidth: '120px' }}>
    <div style={{ fontSize: '11px', color: theme.colors.textMuted, textTransform: 'uppercase', fontWeight: 800 }}>{label}</div>
    <div style={{ fontSize: '20px', fontWeight: 800, color: color || theme.colors.text }}>{value}</div>
  </div>
);

const SectorDrilldown = ({ sector, onClose }) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/sector-specialist/${sector.toLowerCase()}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) { setError(`HTTP ${res.status}`); return; }
        const json = await res.json();
        if (alive) setData(json);
      } catch (e) {
        if (alive) setError(e.message);
      }
    };
    load();
    return () => { alive = false; };
  }, [sector]);

  const profile = data?.profile || {};
  const outlook = profile.outlook_score ?? 0.0;
  const assets = data?.associated_assets || [];

  const getOutlookLabel = (score) => {
    if (score > 0.4) return 'STRONG BULLISH';
    if (score > 0.1) return 'BULLISH';
    if (score < -0.4) return 'STRONG BEARISH';
    if (score < -0.1) return 'BEARISH';
    return 'NEUTRAL';
  };

  const getOutlookColor = (score) => {
    if (score > 0.1) return theme.colors.primary;
    if (score < -0.1) return theme.colors.danger;
    return theme.colors.textSecondary;
  };

  // Convert outlook score (-1.0 to +1.0) to slider percentage (0% to 100%)
  const outlookPct = ((outlook + 1) / 2) * 100;

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, backgroundColor: 'rgba(2,6,23,0.92)',
      display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
      zIndex: 1001, padding: '24px', overflowY: 'auto',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        ...glassCard, maxWidth: '850px', width: '100%', padding: '28px', marginTop: '40px',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Layers size={24} color={theme.colors.primary} />
            <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 800, letterSpacing: '1px' }}>
              {sector.toUpperCase()} INDUSTRY PROFILE
            </h2>
          </div>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: theme.colors.textSecondary, cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
            <X size={22} />
          </button>
        </div>

        {error && <div style={{ color: theme.colors.danger, padding: '20px' }}>Failed to load sector data: {error}</div>}
        {!data && !error && <div style={{ color: theme.colors.textMuted, padding: '40px', textAlign: 'center' }}>Querying Industry Specialist Agent...</div>}

        {data && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Visual Outlook Gauge */}
            <div style={{ backgroundColor: 'rgba(255,255,255,0.02)', border: `1px solid ${theme.colors.border}40`, borderRadius: '12px', padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '12px', fontWeight: '800', color: theme.colors.textSecondary, textTransform: 'uppercase' }}>
                  Specialist Outlook Indicator
                </span>
                <span style={{ fontSize: '14px', fontWeight: '800', color: getOutlookColor(outlook) }}>
                  {getOutlookLabel(outlook)} ({outlook >= 0 ? '+' : ''}{outlook.toFixed(2)})
                </span>
              </div>
              
              {/* Slider Track */}
              <div style={{ position: 'relative', height: '10px', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '5px', overflow: 'visible', marginBottom: '8px' }}>
                {/* Center marker */}
                <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: '2px', backgroundColor: 'rgba(255,255,255,0.15)' }} />
                
                {/* Slider bar color fill */}
                <div style={{ 
                  position: 'absolute', 
                  left: outlook >= 0 ? '50%' : `${outlookPct}%`,
                  right: outlook >= 0 ? `${100 - outlookPct}%` : '50%',
                  top: 0, bottom: 0,
                  backgroundColor: getOutlookColor(outlook),
                  borderRadius: '5px',
                  opacity: 0.8
                }} />
                
                {/* Slider knob */}
                <div style={{
                  position: 'absolute',
                  left: `${outlookPct}%`,
                  top: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: '18px',
                  height: '18px',
                  borderRadius: '50%',
                  backgroundColor: '#fff',
                  border: `3px solid ${getOutlookColor(outlook)}`,
                  boxShadow: '0 2px 6px rgba(0,0,0,0.5)',
                  zIndex: 2,
                  transition: 'all 0.1s ease'
                }} />
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: theme.colors.textMuted }}>
                <span>BEARISH (-1.0)</span>
                <span>NEUTRAL (0.0)</span>
                <span>BULLISH (+1.0)</span>
              </div>
            </div>

            {/* Profile Narrative */}
            <div>
              <h3 style={{ margin: '0 0 10px 0', fontSize: '14px', fontWeight: '800', color: theme.colors.primary, textTransform: 'uppercase' }}>
                Agent Analysis
              </h3>
              <p style={{ margin: 0, fontSize: '13px', lineHeight: '1.6', color: theme.colors.textSecondary }}>
                {profile.updated_profile_text}
              </p>
            </div>

            {/* Risks & Catalysts */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', flexWrap: 'wrap' }}>
              <div style={{ backgroundColor: 'rgba(244,63,94,0.03)', border: `1px solid ${theme.colors.danger}20`, borderRadius: '12px', padding: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: theme.colors.danger }}>
                  <ShieldAlert size={16} />
                  <span style={{ fontSize: '12px', fontWeight: '800', textTransform: 'uppercase' }}>Key Industry Risks</span>
                </div>
                <ul style={{ margin: 0, paddingLeft: '16px', fontSize: '12px', color: theme.colors.textSecondary, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {profile.risk_factors && profile.risk_factors.length > 0 ? (
                    profile.risk_factors.map((r, idx) => <li key={idx}>{r}</li>)
                  ) : (
                    <li>No significant risk factors flagged by the specialist.</li>
                  )}
                </ul>
              </div>

              <div style={{ backgroundColor: 'rgba(16,185,129,0.03)', border: `1px solid ${theme.colors.primary}20`, borderRadius: '12px', padding: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: theme.colors.primary }}>
                  <Award size={16} />
                  <span style={{ fontSize: '12px', fontWeight: '800', textTransform: 'uppercase' }}>Upcoming Catalysts</span>
                </div>
                <ul style={{ margin: 0, paddingLeft: '16px', fontSize: '12px', color: theme.colors.textSecondary, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {profile.catalysts && profile.catalysts.length > 0 ? (
                    profile.catalysts.map((c, idx) => <li key={idx}>{c}</li>)
                  ) : (
                    <li>No immediate catalysts identified by the specialist.</li>
                  )}
                </ul>
              </div>
            </div>

            {/* Portfolio Exposure */}
            <div>
              <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '800', color: theme.colors.secondary, textTransform: 'uppercase' }}>
                Portfolio & Watchlist Assets in this Sector
              </h3>
              <div style={{ overflowX: 'auto', border: `1px solid ${theme.colors.border}30`, borderRadius: '10px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ backgroundColor: 'rgba(255,255,255,0.02)', borderBottom: `1px solid ${theme.colors.border}40`, color: theme.colors.textSecondary }}>
                      <th style={{ padding: '10px 14px' }}>SYMBOL</th>
                      <th style={{ padding: '10px 14px' }}>EXPOSURE TYPE</th>
                      <th style={{ padding: '10px 14px' }}>QUANTITY</th>
                      <th style={{ padding: '10px 14px' }}>CURRENT PRICE</th>
                      <th style={{ padding: '10px 14px', textAlign: 'right' }}>UNREALIZED P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {assets.length > 0 ? (
                      assets.map((asset, idx) => (
                        <tr key={idx} style={{ borderBottom: idx < assets.length - 1 ? `1px solid ${theme.colors.border}20` : 'none' }}>
                          <td style={{ padding: '12px 14px', fontWeight: 'bold' }}>{asset.symbol}</td>
                          <td style={{ padding: '12px 14px' }}>
                            <span style={{ 
                              padding: '2px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: '800',
                              backgroundColor: asset.type === 'position' ? `${theme.colors.primary}20` : 'rgba(255,255,255,0.05)',
                              color: asset.type === 'position' ? theme.colors.primary : theme.colors.textSecondary
                            }}>
                              {asset.type.toUpperCase()}
                            </span>
                          </td>
                          <td style={{ padding: '12px 14px', color: theme.colors.textSecondary }}>
                            {asset.type === 'position' ? asset.quantity.toLocaleString() : '—'}
                          </td>
                          <td style={{ padding: '12px 14px', color: theme.colors.textSecondary }}>
                            {asset.current_price ? `$${asset.current_price.toFixed(2)}` : '—'}
                          </td>
                          <td style={{ padding: '12px 14px', textAlign: 'right', fontWeight: 'bold', color: asset.unrealized_pl >= 0 ? theme.colors.primary : theme.colors.danger }}>
                            {asset.type === 'position' 
                              ? `${asset.unrealized_pl >= 0 ? '+' : ''}$${asset.unrealized_pl.toFixed(2)}` 
                              : '—'}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="5" style={{ padding: '24px', textAlign: 'center', color: theme.colors.textMuted }}>
                          No assets from watchlists or active positions mapped to this industry.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SectorDrilldown;
