import { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell 
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Activity, Shield, Zap, AlertTriangle, 
  Clock, Layers, ChevronRight, Maximize2, Globe, Cpu, RefreshCw,
  Search, Info, ExternalLink, Play, Square, Pause, Terminal, Flag,
  Lock, Bell, BarChart as BarChartIcon, LogOut, FileText, CheckCircle, XCircle, Upload
} from 'lucide-react';
import AgentActivity from './AgentActivity';
import AgentFocus from './AgentFocus';
import SleeveDashboard from './SleeveDashboard';
import MarketClock from './MarketClock';
import HelpPanel from './HelpPanel';
import SymbolDrilldown from './SymbolDrilldown';
import SectorDrilldown from './SectorDrilldown';
import AdvancedAnalytics from './AdvancedAnalytics';
import UnifiedPortfolio from './UnifiedPortfolio';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

const StatCard = ({ label, value, icon: Icon, color }) => (
  <div style={{ ...glassCard, flex: 1, padding: '20px' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
      <span style={{ fontSize: '11px', fontWeight: '800', color: theme.colors.textMuted, textTransform: 'uppercase' }}>{label}</span>
      <Icon size={18} color={color} />
    </div>
    <div style={{ fontSize: '24px', fontWeight: '800', color: '#fff' }}>{value}</div>
  </div>
);

const SectionHeader = ({ title, icon: Icon }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
    <Icon size={20} color={theme.colors.secondary} />
    <h3 style={{ fontSize: '18px', fontWeight: '700', margin: 0, color: '#fff' }}>{title}</h3>
    <div style={{ height: '1px', flex: 1, background: `linear-gradient(90deg, ${theme.colors.border}, transparent)` }} />
  </div>
);

const HUDCard = ({ title, value, subValue, icon: Icon, color, footer }) => (
  <div style={{ ...glassCard, flex: 1, minWidth: '220px', position: 'relative', overflow: 'hidden' }}>
    <div style={{ position: 'absolute', right: '-10px', bottom: '-10px', opacity: 0.05 }}>
      <Icon size={100} color={color} />
    </div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
      <p style={{ fontSize: '12px', color: theme.colors.textSecondary, margin: 0, fontWeight: '700', textTransform: 'uppercase', letterSpacing: '1px' }}>{title}</p>
      <div style={{ backgroundColor: `${color}20`, padding: '6px', borderRadius: '8px', color }}>
        <Icon size={18} />
      </div>
    </div>
    <h2 style={{ fontSize: '28px', fontWeight: '800', margin: 0, color: '#fff' }}>{value}</h2>
    <p style={{ fontSize: '13px', margin: '4px 0 0 0', color: subValue?.includes('-') || subValue?.includes('↘') ? theme.colors.danger : theme.colors.primary }}>
      {subValue}
    </p>
    {footer && (
      <p style={{ fontSize: '11px', margin: '6px 0 0 0', color: theme.colors.textMuted, position: 'relative', zIndex: 1 }}>
        {footer}
      </p>
    )}
  </div>
);

const ResearchView = ({ activeTab, fetchData, onDrill }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [escalations, setEscalations] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [uploads, setUploads] = useState([]);

  const token = typeof window !== 'undefined' ? localStorage.getItem('trading_token') : null;

  const loadResearchData = useCallback(async () => {
    try {
      const hRes = await fetch(`${getApiBase()}/api/operator/upload-history`, { headers: { Authorization: `Bearer ${token}` } });
      if (hRes.ok) {
        const hJson = await hRes.json();
        setUploads(hJson.uploads || []);
      }
      
      const escRes = await fetch(`${getApiBase()}/api/operator/escalations`, { headers: { Authorization: `Bearer ${token}` } });
      if (escRes.ok) {
        const escJson = await escRes.json();
        setEscalations(escJson.escalations || []);
      }
      
      const wlRes = await fetch(`${getApiBase()}/api/operator/watchlist`, { headers: { Authorization: `Bearer ${token}` } });
      if (wlRes.ok) {
        const wlJson = await wlRes.json();
        setWatchlist(wlJson.watchlist || []);
      }
    } catch (e) {
      console.error("Failed to fetch research data", e);
    }
  }, [token]);

  useEffect(() => {
    if (activeTab === 'research') {
      loadResearchData();
    }
  }, [activeTab, loadResearchData]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setUploadResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadResult(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const res = await fetch(`${getApiBase()}/api/operator/upload-research`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData
      });
      const json = await res.json();
      setUploadResult(json);
      setSelectedFile(null);
      loadResearchData();
    } catch (e) {
      setUploadResult({ status: 'failed', error: 'Upload request failed' });
    } finally {
      setUploading(false);
    }
  };

  const handleResolveEscalation = async (escId, status) => {
    try {
      const res = await fetch(`${getApiBase()}/api/operator/escalations/${escId}/resolve`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ status, notes: `Resolved via Web Dashboard` })
      });
      if (res.ok) {
        loadResearchData();
        fetchData();
      }
    } catch (e) {
      console.error("Resolution failed", e);
    }
  };

  const handleWatchlistAction = async (symbol, action) => {
    try {
      const res = await fetch(`${getApiBase()}/api/operator/watchlist/${symbol}/pause`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ action })
      });
      if (res.ok) {
        loadResearchData();
      }
    } catch (e) {
      console.error("Watchlist action failed", e);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap' }}>
        <div style={{ ...glassCard, flex: 1, minWidth: '400px' }}>
          <SectionHeader title="Research PDF Ingestion" icon={Upload} />
          <div style={{ border: `2px dashed ${theme.colors.border}`, borderRadius: '12px', padding: '30px', textAlign: 'center', backgroundColor: 'rgba(255,255,255,0.01)', position: 'relative' }}>
            {!selectedFile && (
              <input type="file" onChange={handleFileChange} accept=".pdf" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }} />
            )}
            <FileText size={40} color={selectedFile ? theme.colors.primary : theme.colors.textMuted} style={{ marginBottom: '12px' }} />
            {selectedFile ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                <p style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: 'bold' }}>{selectedFile.name}</p>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button onClick={(e) => { e.stopPropagation(); handleUpload(); }} disabled={uploading} style={{ backgroundColor: theme.colors.primary, color: '#000', border: 'none', padding: '8px 20px', borderRadius: '8px', fontSize: '13px', fontWeight: '800', cursor: 'pointer' }}>
                    {uploading ? 'Processing PDF...' : 'Audit Document'}
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }} style={{ backgroundColor: 'transparent', color: theme.colors.danger, border: `1px solid ${theme.colors.danger}`, padding: '8px 20px', borderRadius: '8px', fontSize: '13px', fontWeight: '800', cursor: 'pointer' }}>
                    Clear
                  </button>
                </div>
              </div>
            ) : (
              <div>
                <p style={{ margin: 0, fontSize: '13px', color: theme.colors.textSecondary }}>Drag & drop or click to select AIB AXYS broker report (PDF)</p>
                <p style={{ margin: '5px 0 0 0', fontSize: '11px', color: theme.colors.textMuted }}>Directly extracts symbols, recommendations & investment rationales via LLM</p>
              </div>
            )}
          </div>
          
          {uploadResult && (
            <div style={{ marginTop: '20px', padding: '15px', borderRadius: '8px', border: `1px solid ${uploadResult.status === 'completed' ? theme.colors.primary : theme.colors.danger}`, backgroundColor: `${uploadResult.status === 'completed' ? theme.colors.primary : theme.colors.danger}10` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '700', fontSize: '14px', color: uploadResult.status === 'completed' ? theme.colors.primary : theme.colors.danger }}>
                {uploadResult.status === 'completed' ? <CheckCircle size={16} /> : <XCircle size={16} />}
                <span>{uploadResult.status === 'completed' ? 'Processing Complete' : 'Processing Failed'}</span>
              </div>
              {uploadResult.status === 'completed' ? (
                <div style={{ fontSize: '12px', marginTop: '5px', color: theme.colors.textSecondary }}>
                  <p style={{ margin: '3px 0' }}>Successfully processed <strong>{uploadResult.signals_processed}</strong> recommendations.</p>
                  {uploadResult.auto_followed?.length > 0 && <p style={{ margin: '3px 0' }}>Auto-followed watchlist: <span style={{ color: theme.colors.primary }}>{uploadResult.auto_followed.join(', ')}</span></p>}
                  {uploadResult.escalated?.length > 0 && <p style={{ margin: '3px 0' }}>Escalated to Operator queue: <span style={{ color: theme.colors.warning }}>{uploadResult.escalated.map(x => x[0]).join(', ')}</span></p>}
                </div>
              ) : (
                <p style={{ fontSize: '12px', margin: '5px 0 0 0', color: theme.colors.textMuted }}>{uploadResult.error || 'Check server logs for details'}</p>
              )}
            </div>
          )}
        </div>

        <div style={{ ...glassCard, flex: 1, minWidth: '400px' }}>
          <SectionHeader title="Ingest Archives" icon={Clock} />
          <div style={{ maxHeight: '190px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, textAlign: 'left' }}>
                  <th style={{ padding: '8px 0' }}>FILENAME</th>
                  <th style={{ padding: '8px 0' }}>DATE</th>
                  <th style={{ padding: '8px 0' }}>EXTRACTS</th>
                  <th style={{ padding: '8px 0' }}>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {uploads.length > 0 ? (
                  uploads.map(u => (
                    <tr key={u.id} style={{ borderBottom: `1px solid ${theme.colors.border}20` }}>
                      <td style={{ padding: '10px 0', fontWeight: 'bold' }}>{u.filename}</td>
                      <td style={{ padding: '10px 0', color: theme.colors.textMuted }}>{new Date(u.uploaded_at).toLocaleString()}</td>
                      <td style={{ padding: '10px 0' }}>{u.signals_count} positions</td>
                      <td style={{ padding: '10px 0', color: u.status === 'completed' ? theme.colors.primary : theme.colors.danger }}>{u.status.toUpperCase()}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan="4" style={{ padding: '20px 0', textAlign: 'center', color: theme.colors.textMuted }}>No documents audited yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div style={glassCard}>
        <SectionHeader title="Operator Approval Queue" icon={AlertTriangle} />
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, textAlign: 'left' }}>
                <th style={{ padding: '10px 0' }}>ASSET</th>
                <th style={{ padding: '10px 0' }}>ACTION</th>
                <th style={{ padding: '10px 0' }}>BROKER RATING</th>
                <th style={{ padding: '10px 0' }}>TARGET (UPSIDE)</th>
                <th style={{ padding: '10px 0' }}>RISK / ESCALATION REASON</th>
                <th style={{ padding: '10px 0', textAlign: 'right' }}>DECISION</th>
              </tr>
            </thead>
            <tbody>
              {escalations.length > 0 ? (
                escalations.map(esc => (
                  <tr key={esc.id} style={{ borderBottom: `1px solid ${theme.colors.border}30` }}>
                    <td style={{ padding: '12px 0', fontWeight: 'bold' }}>{esc.symbol}</td>
                    <td style={{ padding: '12px 0' }}><span style={{ backgroundColor: `${theme.colors.accent}15`, color: theme.colors.accent, padding: '3px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: '800' }}>{esc.action.toUpperCase()}</span></td>
                    <td style={{ padding: '12px 0' }}><span style={{ color: esc.recommendation === 'SELL' ? theme.colors.danger : theme.colors.primary, fontWeight: '700' }}>{esc.recommendation || 'UNKNOWN'}</span></td>
                    <td style={{ padding: '12px 0' }}>
                      {esc.target_price ? (
                        <span>KES {esc.target_price.toFixed(2)} ({esc.upside_pct ? `${esc.upside_pct.toFixed(1)}%` : '-%'})</span>
                      ) : (
                        <span style={{ color: theme.colors.textMuted }}>N/A</span>
                      )}
                    </td>
                    <td style={{ padding: '12px 0', maxWidth: '380px', whiteSpace: 'normal', fontSize: '12px' }}>
                      <span style={{ color: theme.colors.warning, fontWeight: '700', marginRight: '5px' }}>[{esc.risk_level.toUpperCase()}]</span>
                      <span style={{ color: theme.colors.textSecondary, fontWeight: '600' }}>{esc.reason}</span>
                      {esc.rationale && (
                        <p style={{ margin: '4px 0 0 0', fontSize: '11px', color: theme.colors.textMuted, fontStyle: 'italic', lineHeight: '1.4' }}>
                          "{esc.rationale}"
                        </p>
                      )}
                    </td>
                    <td style={{ padding: '12px 0', textAlign: 'right' }}>
                      <button onClick={() => handleResolveEscalation(esc.id, 'approved')} style={{ backgroundColor: theme.colors.primary, color: '#000', border: 'none', padding: '6px 12px', borderRadius: '6px', fontSize: '11px', fontWeight: '800', cursor: 'pointer', marginRight: '8px' }}>APPROVE</button>
                      <button onClick={() => handleResolveEscalation(esc.id, 'rejected')} style={{ backgroundColor: 'rgba(244, 63, 94, 0.1)', color: theme.colors.danger, border: `1px solid ${theme.colors.danger}`, padding: '5px 12px', borderRadius: '6px', fontSize: '11px', fontWeight: '800', cursor: 'pointer' }}>REJECT</button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr><td colSpan="6" style={{ padding: '30px 0', textAlign: 'center', color: theme.colors.textMuted }}>Approval queue is empty. System running autonomously.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div style={glassCard}>
        <SectionHeader title="Active Symbol Watchlist" icon={Layers} />
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, textAlign: 'left' }}>
                <th style={{ padding: '10px 0' }}>SYMBOL</th>
                <th style={{ padding: '10px 0' }}>MARKET</th>
                <th style={{ padding: '10px 0' }}>RATING</th>
                <th style={{ padding: '10px 0' }}>TARGET PRICE</th>
                <th style={{ padding: '10px 0' }}>SOURCE</th>
                <th style={{ padding: '10px 0' }}>STATUS</th>
                <th style={{ padding: '10px 0', textAlign: 'right' }}>ACTION</th>
              </tr>
            </thead>
            <tbody>
              {watchlist.length > 0 ? (
                watchlist.map(item => (
                  <tr key={item.symbol} onClick={() => onDrill(item.symbol)} title={`Open ${item.symbol} performance drill-down`} style={{ borderBottom: `1px solid ${theme.colors.border}20`, cursor: 'pointer' }}>
                    <td style={{ padding: '12px 0', fontWeight: 'bold' }}>{item.symbol}</td>
                    <td style={{ padding: '12px 0', color: theme.colors.textSecondary }}>{item.market.toUpperCase()}</td>
                    <td style={{ padding: '12px 0', color: theme.colors.primary, fontWeight: '700' }}>{item.recommendation}</td>
                    <td style={{ padding: '12px 0' }}>{item.target_price ? `KES ${item.target_price.toFixed(2)}` : 'N/A'}</td>
                    <td style={{ padding: '12px 0', color: theme.colors.textMuted }}>{item.source}</td>
                    <td>
                      <span style={{ backgroundColor: item.status === 'active' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)', color: item.status === 'active' ? theme.colors.primary : theme.colors.warning, padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold' }}>
                        {item.status.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '12px 0', textAlign: 'right' }}>
                      {item.status === 'active' ? (
                        <button onClick={(e) => { e.stopPropagation(); handleWatchlistAction(item.symbol, 'pause'); }} style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: theme.colors.warning, border: `1px solid ${theme.colors.warning}`, padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '800', cursor: 'pointer', marginRight: '5px' }}>PAUSE</button>
                      ) : (
                        <button onClick={(e) => { e.stopPropagation(); handleWatchlistAction(item.symbol, 'resume'); }} style={{ backgroundColor: 'transparent', color: theme.colors.primary, border: `1px solid ${theme.colors.primary}`, padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '800', cursor: 'pointer', marginRight: '5px' }}>RESUME</button>
                      )}
                      <button onClick={(e) => { e.stopPropagation(); handleWatchlistAction(item.symbol, 'remove'); }} style={{ backgroundColor: 'transparent', color: theme.colors.danger, border: `1px solid ${theme.colors.danger}`, padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '800', cursor: 'pointer' }}>REMOVE</button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr><td colSpan="7" style={{ padding: '30px 0', textAlign: 'center', color: theme.colors.textMuted }}>No symbol watchlist records. Run research ingest or approve escalations to watch assets.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// NSE order tickets: the agent proposes NSE trades here (no broker API), and
// the operator keys them into the AIB-AXYS portal and records the fill.
const NseTicketsPanel = ({ isOperator, activeTab }) => {
  const [pending, setPending] = useState([]);
  const [fills, setFills] = useState([]);
  const token = typeof window !== 'undefined' ? localStorage.getItem('trading_token') : null;

  const load = useCallback(async () => {
    if (!isOperator) return;
    try {
      const res = await fetch(`${getApiBase()}/api/operator/nse-tickets`, { headers: { Authorization: `Bearer ${token}` } });
      if (res.ok) {
        const j = await res.json();
        setPending(j.pending || []);
        setFills(j.recent_fills || []);
      }
    } catch (e) { /* transient */ }
  }, [isOperator, token]);

  useEffect(() => {
    if (activeTab === 'nse' && isOperator) {
      load();
      const id = setInterval(load, 20000);
      return () => clearInterval(id);
    }
  }, [activeTab, isOperator, load]);

  const act = async (id, verb, body) => {
    try {
      const res = await fetch(`${getApiBase()}/api/operator/nse-tickets/${id}/${verb}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(body || {}),
      });
      if (res.ok) load();
      else { const j = await res.json().catch(() => ({})); window.alert(j.error || 'Action failed'); }
    } catch (e) { window.alert('Request failed'); }
  };

  const onFill = (t) => {
    const priceStr = window.prompt(`Actual fill price (KES) for ${t.side.toUpperCase()} ${t.quantity} ${t.symbol}:`, t.suggested_limit_price ?? '');
    if (priceStr === null) return;
    const qtyStr = window.prompt(`Actual fill quantity for ${t.symbol}:`, t.quantity);
    if (qtyStr === null) return;
    const fill_price = parseFloat(priceStr), fill_quantity = parseInt(qtyStr, 10);
    if (!(fill_price > 0) || !(fill_quantity > 0)) { window.alert('Fill price and quantity must be positive numbers.'); return; }
    act(t.id, 'fill', { fill_price, fill_quantity });
  };

  if (!isOperator) {
    return (
      <div style={glassCard}>
        <SectionHeader title="NSE Order Tickets" icon={Layers} />
        <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>Operator view only.</div>
      </div>
    );
  }

  return (
    <div style={glassCard}>
      <SectionHeader title="NSE Order Tickets" icon={Layers} />
      <div style={{ marginBottom: '12px', fontSize: '11px', color: theme.colors.textMuted }}>
        The agent proposes NSE trades here (no broker API). Place the order in your broker portal, then record the fill.
      </div>
      {pending.length > 0 ? (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: theme.colors.textMuted, fontSize: '12px', borderBottom: `1px solid ${theme.colors.border}` }}>
                <th style={{ padding: '10px' }}>SYMBOL</th><th style={{ padding: '10px' }}>SIDE</th><th style={{ padding: '10px' }}>QTY</th><th style={{ padding: '10px' }}>LIMIT (KES)</th><th style={{ padding: '10px' }}>CONF</th><th style={{ padding: '10px' }}>STATUS</th><th style={{ padding: '10px' }}>ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {pending.map((t) => (
                <tr key={t.id} style={{ borderBottom: `1px solid ${theme.colors.border}` }} title={t.llm_reasoning || t.rationale || ''}>
                  <td style={{ padding: '10px', fontWeight: 700 }}>{t.symbol}</td>
                  <td style={{ padding: '10px', color: t.side === 'buy' ? theme.colors.primary : theme.colors.danger, fontWeight: 700 }}>{t.side.toUpperCase()}</td>
                  <td style={{ padding: '10px' }}>{t.quantity}</td>
                  <td style={{ padding: '10px' }}>{t.suggested_limit_price?.toFixed?.(2) ?? t.suggested_limit_price}</td>
                  <td style={{ padding: '10px' }}>{((t.ensemble_confidence || 0) * 100).toFixed(0)}%</td>
                  <td style={{ padding: '10px', textTransform: 'uppercase', fontSize: '11px', color: t.status === 'placed' ? theme.colors.warning : theme.colors.textMuted }}>{t.status}</td>
                  <td style={{ padding: '10px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                    {t.status === 'pending' && <button onClick={() => act(t.id, 'place')} style={{ background: theme.colors.bgSecondary, color: '#fff', border: `1px solid ${theme.colors.border}`, borderRadius: '6px', padding: '4px 10px', cursor: 'pointer', fontSize: '11px' }}>Mark Placed</button>}
                    <button onClick={() => onFill(t)} style={{ background: theme.colors.primary, color: '#000', border: 'none', borderRadius: '6px', padding: '4px 10px', cursor: 'pointer', fontSize: '11px', fontWeight: 700 }}>Mark Filled</button>
                    <button onClick={() => act(t.id, 'cancel')} style={{ background: 'transparent', color: theme.colors.danger, border: `1px solid ${theme.colors.danger}55`, borderRadius: '6px', padding: '4px 10px', cursor: 'pointer', fontSize: '11px' }}>Cancel</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No pending NSE order tickets.</div>
      )}
      {fills.length > 0 && (
        <div style={{ marginTop: '18px' }}>
          <div style={{ fontSize: '11px', fontWeight: 800, color: theme.colors.textMuted, marginBottom: '8px' }}>RECENT NSE FILLS</div>
          {fills.slice(0, 8).map((f) => (
            <div key={f.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: `1px solid ${theme.colors.border}`, fontSize: '12px' }}>
              <span><span style={{ fontWeight: 700 }}>{f.symbol}</span> <span style={{ color: f.side === 'buy' ? theme.colors.primary : theme.colors.danger }}>{f.side.toUpperCase()}</span> {f.fill_quantity} @ {f.fill_price} KES</span>
              <span style={{ color: theme.colors.textMuted }}>{f.fill_at ? new Date(f.fill_at).toLocaleString() : ''}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const AdvancedDashboard = ({ onLogout }) => {

  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState({
    loading: true,
    status: { components: {} },
    performance: { portfolio_value: 0, total_pnl: 0, win_rate: 0, sharpe_ratio: 0, max_drawdown: 0, portfolio_chart: [] }, 
    positions: [], 
    alerts: [], 
    news: [], 
    riskMetrics: { portfolio_var: 0, beta: 1.0, volatility: 0.15, current_leverage: 1.0, risk_score: 5.0 }, 
    modelPerf: { accuracy: 0.5, feature_importance: [] }, 
    strategies: [], 
    heatmap: [], 
    systemHealth: { uptime: 0, cpu_usage: 0, memory_usage: 'stable' }, 
    agentActivity: [],
    allocation: []
  });
  const [nseData, setNseData] = useState({ quotes: [], movers: { gainers: [], losers: [] }, sectors: [], status: {}, kes_usd_rate: 0.0077, market_open: false });
  const [isConnected, setIsConnected] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [drilldownSymbol, setDrilldownSymbol] = useState(null);
  const [drilldownSector, setDrilldownSector] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [nseSort, setNseSort] = useState({ key: 'symbol', dir: 1 });
  const [nseFilter, setNseFilter] = useState('all'); // all | held | movers

  // First login: open the getting-started guide automatically.
  useEffect(() => {
    if (!localStorage.getItem('aegis_help_seen')) {
      setShowHelp(true);
      localStorage.setItem('aegis_help_seen', '1');
    }
  }, []);

  const isHalted = !!data.status.trading_halted;
  const isOperator = data.status.role === 'operator';

  // Operator-only anomaly scan: what's unusual today. 403 for viewers.
  useEffect(() => {
    if (!isOperator) { setAnomalies([]); return; }
    let alive = true;
    const load = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const res = await fetch(`${getApiBase()}/api/anomalies`, { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return;
        const json = await res.json();
        if (alive) setAnomalies(json.anomalies || []);
      } catch (e) {}
    };
    load();
    const id = setInterval(load, 30000);
    return () => { alive = false; clearInterval(id); };
  }, [isOperator]);

  const toggleHalt = async () => {
    const message = isHalted
      ? 'Resume automated trading?'
      : 'HALT TRADING?\n\nThe agent stops submitting new orders immediately. Open positions stay open (protective stop-losses keep working).';
    if (!window.confirm(message)) return;
    try {
      const token = localStorage.getItem('trading_token');
      const res = await fetch(`${getApiBase()}/api/trading/${isHalted ? 'resume' : 'halt'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ confirm: true }),
      });
      if (res.status === 401) { onLogout(); return; }
      await fetchData();
    } catch (e) {}
  };

  const fetchData = async () => {
    const endpoints = [
      'status', 'performance', 'positions', 'alerts', 'news-feed',
      'risk-metrics', 'model-performance', 'strategy-performance',
      'market-heatmap', 'system-health', 'agent-activity', 'portfolio-allocation'
    ];

    const token = localStorage.getItem('trading_token');
    if (!token) { onLogout(); return; }

    let unauthorized = false;
    const newResponses = await Promise.all(endpoints.map(async (endpoint) => {
      try {
        const res = await fetch(`${getApiBase()}/api/${endpoint}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.status === 401) { unauthorized = true; return null; }
        if (!res.ok) return null;
        return await res.json();
      } catch (err) {
        return null;
      }
    }));
    if (unauthorized) { onLogout(); return; }

    // A response only replaces cached state when it parsed and isn't an error envelope.
    const ok = (r) => r != null && !(typeof r === 'object' && !Array.isArray(r) && r.error);

    setData(prev => ({
      loading: false,
      status: ok(newResponses[0]) ? newResponses[0] : prev.status,
      performance: ok(newResponses[1]) ? newResponses[1] : prev.performance,
      positions: Array.isArray(newResponses[2]) ? newResponses[2] : prev.positions,
      alerts: Array.isArray(newResponses[3]) ? newResponses[3] : prev.alerts,
      news: Array.isArray(newResponses[4]) ? newResponses[4] : prev.news,
      riskMetrics: ok(newResponses[5]) ? newResponses[5] : prev.riskMetrics,
      modelPerf: ok(newResponses[6]) ? newResponses[6] : prev.modelPerf,
      strategies: ok(newResponses[7])
        ? Object.entries(newResponses[7])
            .filter(([, stats]) => stats && typeof stats === 'object' && 'realized_pnl' in stats)
            .map(([name, stats]) => ({ name, ...stats }))
        : prev.strategies,
      heatmap: Array.isArray(newResponses[8]) ? newResponses[8] : prev.heatmap,
      systemHealth: ok(newResponses[9]) ? newResponses[9] : prev.systemHealth,
      agentActivity: Array.isArray(newResponses[10]) ? newResponses[10] : prev.agentActivity,
      allocation: Array.isArray(newResponses[11]) ? newResponses[11] : prev.allocation
    }));
    setIsConnected(ok(newResponses[0]));
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 20000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchNSE = async () => {
      try {
        const token = localStorage.getItem('trading_token');
        const resp = await fetch(`${getApiBase()}/api/nse-market`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (resp.status === 401) { onLogout(); return; }
        const nse = await resp.json();
        if (nse && !nse.error) setNseData(nse);
      } catch (e) {}
    };
    fetchNSE();
    const nseInterval = setInterval(fetchNSE, 30000);
    return () => clearInterval(nseInterval);
  }, []);

  const renderOverview = () => {
    if (data.loading) {
      return (
        <div style={{ padding: '80px', textAlign: 'center', color: theme.colors.textMuted }}>
          <div style={{ fontSize: '14px', letterSpacing: '2px' }}>ESTABLISHING SECURE LINK…</div>
          <div style={{ fontSize: '11px', marginTop: '8px' }}>Loading portfolio, risk and market state</div>
        </div>
      );
    }
    return (
    <div style={{ display: 'grid', gap: '30px' }}>
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <HUDCard title="Consolidated Equity"
          value={`$${(data.performance.consolidated_equity ?? data.performance.portfolio_value ?? 0).toLocaleString()}`}
          subValue={data.performance.total_pnl > 0 ? `↗ $${data.performance.total_pnl.toFixed(2)}` : `↘ $${(data.performance.total_pnl || 0).toFixed(2)}`}
          footer={`Paper (US): $${(data.performance.us_paper_value ?? data.performance.portfolio_value ?? 0).toLocaleString()} · Real (NSE): $${(data.performance.nse_value_usd ?? 0).toLocaleString()}`}
          icon={TrendingUp} color={theme.colors.primary} />
        <HUDCard title="Exposure (VaR)" value={`$${(data.riskMetrics.portfolio_var || 0).toLocaleString()}`} subValue={`Risk Score: ${data.riskMetrics.risk_score?.toFixed(1) || '0.0'}/10`} icon={Shield} color={theme.colors.warning} />
        <HUDCard title="Win Rate" value={`${((data.performance.win_rate || 0) * 100).toFixed(1)}%`} subValue={`${data.performance.total_trades || 0} Trades`} icon={Zap} color={theme.colors.secondary} />
        <HUDCard title="System Health" value={isConnected ? 'OPTIMAL' : 'DEGRADED'} subValue={isConnected ? `${Object.keys(data.status.components || {}).length} Services Active` : 'Reconnecting…'} icon={Activity} color={isConnected ? theme.colors.accent : theme.colors.warning} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={glassCard}>
            <SectionHeader title="Performance Curve" icon={Activity} />
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={data.performance.portfolio_chart || []}>
                <defs><linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={theme.colors.primary} stopOpacity={0.3}/><stop offset="95%" stopColor={theme.colors.primary} stopOpacity={0}/></linearGradient></defs>
                <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.border} vertical={false} />
                <XAxis 
                  dataKey="timestamp" 
                  stroke={theme.colors.textMuted} 
                  fontSize={10} 
                  tickFormatter={(t) => {
                    try {
                      const d = new Date(t);
                      const isToday = d.toDateString() === new Date().toDateString();
                      return isToday 
                        ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
                        : d.toLocaleDateString([], { month: 'short', day: 'numeric' });
                    } catch (e) {
                      return t;
                    }
                  }} 
                />
                <YAxis stroke={theme.colors.textMuted} fontSize={10} domain={['dataMin - (dataMin * 0.01)', 'dataMax + (dataMax * 0.01)']} />
                <Tooltip contentStyle={{ backgroundColor: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, color: '#fff' }} />
                <Area type="monotone" dataKey="value" stroke={theme.colors.primary} fill="url(#colorVal)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <AgentFocus onDrill={setDrilldownSymbol} />
          <SleeveDashboard />
          <AgentActivity activities={data.agentActivity} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={glassCard}>
            <SectionHeader title="Top Positions" icon={RefreshCw} />
            {data.positions.length > 0 ? (
              data.positions.slice(0, 5).map((pos, i) => (
                <div key={i} onClick={() => setDrilldownSymbol(pos.symbol)} title={`Drill into ${pos.symbol}`} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: `1px solid ${theme.colors.border}`, cursor: 'pointer' }}>
                  <span style={{ fontWeight: '700' }}>{pos.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span></span>
                  <span style={{ color: pos.unrealized_pl >= 0 ? theme.colors.primary : theme.colors.danger }}>{pos.unrealized_pl >= 0 ? '+' : ''}{pos.unrealized_pl_pct?.toFixed(2)}%</span>
                </div>
              ))
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No active positions</div>
            )}
          </div>
          
          {/* Asset Allocation Pie/Donut Chart */}
          <div style={glassCard}>
            <SectionHeader title="Asset Allocation" icon={Layers} />
            <div style={{ height: '170px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', marginTop: '10px' }}>
              {data.allocation && data.allocation.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={data.allocation}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={70}
                      paddingAngle={3}
                      dataKey="value"
                    >
                      {data.allocation.map((entry, idx) => (
                        <Cell key={`cell-${idx}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: theme.colors.bgSecondary, border: `1px solid ${theme.colors.border}`, borderRadius: '8px', fontSize: '11px', color: '#fff' }}
                      formatter={(value) => [`$${value.toLocaleString()}`, 'Value']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>No allocation data</div>
              )}
            </div>
            {/* Scrollable Legend list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '12px', maxHeight: '110px', overflowY: 'auto', paddingRight: '4px' }}>
              {data.allocation && data.allocation.map((entry, idx) => (
                <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: entry.color }} />
                    <span style={{ fontWeight: '600' }}>{entry.name}</span>
                  </div>
                  <span style={{ color: theme.colors.textSecondary }}>${entry.value.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={glassCard}>
            <SectionHeader title="Strategy Attribution" icon={Layers} />
            {data.strategies.length > 0 ? (
              data.strategies.map((s, i) => (
                <div key={i} style={{ padding: '10px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontWeight: '700', fontSize: '12px', textTransform: 'uppercase' }}>{s.name}</span>
                    <span style={{ fontWeight: '800', color: (s.realized_pnl + (s.unrealized_pnl || 0)) >= 0 ? theme.colors.primary : theme.colors.danger }}>
                      {(s.realized_pnl + (s.unrealized_pnl || 0)) >= 0 ? '+' : ''}${(s.realized_pnl + (s.unrealized_pnl || 0)).toFixed(2)}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: theme.colors.textMuted }}>
                    {s.closed_trades} closed · {((s.win_rate || 0) * 100).toFixed(0)}% wins · {(s.open_positions || []).length} open
                  </div>
                </div>
              ))
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No attributed trades yet</div>
            )}
          </div>
          <div style={glassCard}>
            <SectionHeader title="Sector Performance" icon={Globe} />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              {data.heatmap.slice(0, 4).map((s, i) => (
                <div key={i} style={{ padding: '10px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', textAlign: 'center' }}>
                  <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>{(s.sector || 'N/A').toUpperCase()}</div>
                  <div style={{ fontWeight: '800', color: (s.performance || 0) >= 0 ? theme.colors.primary : theme.colors.danger }}>{((s.performance || 0) * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
    );
  };

  const nseSession = () => {
    const now = new Date();
    const eatMin = ((now.getUTCHours() + 3) % 24) * 60 + now.getUTCMinutes();
    const day = (now.getUTCDay() + (now.getUTCHours() + 3 >= 24 ? 1 : 0)) % 7;
    const OPEN = 9 * 60 + 30, CLOSE = 15 * 60, PRE = 9 * 60;
    if (day === 0 || day === 6) return { label: 'CLOSED · WEEKEND', color: theme.colors.textMuted };
    // Local time/weekday logic doesn't know gazetted public holidays — the
    // backend's is_market_open() does. If local math says we're inside
    // trading hours on a weekday but the backend disagrees, that gap is a
    // holiday closure; defer to the backend rather than showing a bogus
    // "OPEN · CLOSES IN Xh" countdown.
    const inHoursLocally = eatMin >= PRE && eatMin < CLOSE;
    if (inHoursLocally && !nseData.market_open) {
      return { label: 'CLOSED · HOLIDAY', color: theme.colors.textMuted };
    }
    if (eatMin < PRE) return { label: `PRE-OPEN IN ${Math.floor((PRE - eatMin) / 60)}h ${(PRE - eatMin) % 60}m`, color: theme.colors.textMuted };
    if (eatMin < OPEN) return { label: `PRE-OPEN · TRADING IN ${OPEN - eatMin}m`, color: theme.colors.warning };
    if (eatMin < CLOSE) return { label: `OPEN · CLOSES IN ${Math.floor((CLOSE - eatMin) / 60)}h ${(CLOSE - eatMin) % 60}m`, color: theme.colors.primary };
    return { label: 'CLOSED', color: theme.colors.warning };
  };

  const renderKenyaNSE = () => {
    const positionsBySymbol = Object.fromEntries((data.positions || []).map(p => [p.symbol, p]));
    const heldCount = (nseData.quotes || []).filter(q => positionsBySymbol[q.symbol]).length;
    const sorted = [...(nseData.quotes || [])]
      .filter(q => nseFilter === 'all' || (nseFilter === 'held' ? positionsBySymbol[q.symbol] : Math.abs(q.change_pct || 0) >= 1))
      .sort((a, b) => {
        const va = a[nseSort.key] ?? '', vb = b[nseSort.key] ?? '';
        return (typeof va === 'number' ? va - vb : String(va).localeCompare(String(vb))) * nseSort.dir;
      });
    const sortBtn = (key, label) => (
      <th key={key} onClick={() => setNseSort(s => ({ key, dir: s.key === key ? -s.dir : 1 }))}
          style={{ padding: '12px', cursor: 'pointer', userSelect: 'none' }}>
        {label}{nseSort.key === key ? (nseSort.dir === 1 ? ' ▲' : ' ▼') : ''}
      </th>
    );
    return (
    <div style={{ display: 'grid', gap: '30px' }}>
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <HUDCard title="KES / USD" value={`${(1 / (nseData.kes_usd_rate || 0.0077)).toFixed(2)}`} subValue="Central Bank Rate" icon={Globe} color={theme.colors.secondary} />
        {(() => { const s = nseSession(); return (
          <HUDCard title="NSE Session" value={s.label.split(' · ')[0]} subValue={s.label.includes('·') ? s.label.split(' · ')[1] : `${nseData.quotes?.length || 0} Symbols`} icon={Activity} color={s.color} />
        ); })()}
        <div onClick={() => nseData.movers?.gainers?.[0]?.symbol && setDrilldownSymbol(nseData.movers.gainers[0].symbol)} style={{ cursor: 'pointer', flex: 1, minWidth: '220px' }}>
          <HUDCard title="Top NSE Gainer" value={nseData.movers?.gainers?.[0]?.symbol || '—'} subValue={`+${nseData.movers?.gainers?.[0]?.change_pct?.toFixed(2) || 0}%`} icon={TrendingUp} color={theme.colors.primary} />
        </div>
        <div onClick={() => nseData.movers?.losers?.[0]?.symbol && setDrilldownSymbol(nseData.movers.losers[0].symbol)} style={{ cursor: 'pointer', flex: 1, minWidth: '220px' }}>
          <HUDCard title="Top NSE Loser" value={nseData.movers?.losers?.[0]?.symbol || '—'} subValue={`${nseData.movers?.losers?.[0]?.change_pct?.toFixed(2) || 0}%`} icon={TrendingDown} color={theme.colors.danger} />
        </div>
        <HUDCard title="Your NSE Positions" value={heldCount} subValue={`of ${nseData.quotes?.length || 0} watched`} icon={Shield} color={theme.colors.secondary} />
      </div>
      <div style={glassCard}>
        <SectionHeader title="NSE Market Watch" icon={Flag} />
        <div style={{ marginBottom: '12px', fontSize: '11px', color: theme.colors.textMuted }}>
          The agent watches all {nseData.quotes?.length || 0} symbols below every cycle. Rows highlighted are symbols currently held.
        </div>
        <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
          {[['all', 'ALL'], ['held', 'HELD'], ['movers', 'MOVERS ±1%']].map(([id, label]) => (
            <button key={id} onClick={() => setNseFilter(id)} style={{
              backgroundColor: nseFilter === id ? theme.colors.primary : 'rgba(255,255,255,0.05)',
              color: nseFilter === id ? '#000' : theme.colors.textSecondary,
              border: 'none', padding: '5px 12px', borderRadius: '6px', fontSize: '11px', fontWeight: 800, cursor: 'pointer'
            }}>{label}</button>
          ))}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: theme.colors.textMuted, fontSize: '12px', borderBottom: `1px solid ${theme.colors.border}` }}>
                {sortBtn('symbol', 'SYMBOL')}{sortBtn('price_kes', 'PRICE (KES)')}{sortBtn('change_pct', 'CHANGE')}{sortBtn('volume', 'VOLUME')}<th style={{ padding: '12px' }}>YOUR POSITION</th>
              </tr>
            </thead>
            <tbody>
              {sorted.length > 0 ? (
                sorted.map((q, i) => {
                  const pos = positionsBySymbol[q.symbol];
                  return (
                    <tr key={q.symbol || i}
                        onClick={() => setDrilldownSymbol(q.symbol)}
                        title={`Open ${q.symbol} performance drill-down`}
                        style={{ borderBottom: `1px solid ${theme.colors.border}`, background: pos ? `${theme.colors.primary}15` : 'transparent', cursor: 'pointer' }}
                        onMouseEnter={(e) => e.currentTarget.style.background = `${theme.colors.primary}25`}
                        onMouseLeave={(e) => e.currentTarget.style.background = pos ? `${theme.colors.primary}15` : 'transparent'}>
                      <td style={{ padding: '12px', fontWeight: '700' }}>{q.symbol} <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span></td>
                      <td style={{ padding: '12px' }}>{q.price_kes?.toFixed(2)}</td>
                      <td style={{ padding: '12px', color: q.change_pct >= 0 ? theme.colors.primary : theme.colors.danger }}>{q.change_pct >= 0 ? '+' : ''}{q.change_pct?.toFixed(2)}%</td>
                      <td style={{ padding: '12px' }}>{q.volume?.toLocaleString()}</td>
                      <td style={{ padding: '12px', color: pos ? (pos.unrealized_pl >= 0 ? theme.colors.primary : theme.colors.danger) : theme.colors.textMuted }}>
                        {pos ? `${pos.quantity} @ ${pos.unrealized_pl >= 0 ? '+' : ''}${pos.unrealized_pl_pct?.toFixed(2)}%` : '—'}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr><td colSpan="5" style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No NSE data available</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      <NseTicketsPanel isOperator={isOperator} activeTab={activeTab} />
    </div>
    );
  };

  const renderRiskView = () => {
    const rm = data.riskMetrics || {};
    const metrics = rm.current_metrics || {};
    return (
      <div style={{ display: 'grid', gap: '30px' }}>
        <div style={{ display: 'flex', gap: '20px' }}>
          <StatCard label="Portfolio VaR" value={`${((metrics.portfolio_var || 0) * 100).toFixed(4)}%`} icon={Shield} color={theme.colors.warning} />
          <StatCard label="Max Drawdown" value={`${((metrics.max_drawdown || 0) * 100).toFixed(2)}%`} icon={TrendingDown} color={theme.colors.danger} />
          <StatCard label="Leverage" value={`${(metrics.leverage || 1.0).toFixed(2)}x`} icon={Zap} color={theme.colors.primary} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          <div style={glassCard}>
            <SectionHeader title="Risk Limits" icon={Lock} />
            {(rm.risk_limits || []).map((l, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
                <span>{l.name}</span>
                <span style={{ color: l.status === 'ok' ? theme.colors.primary : theme.colors.danger }}>{(l.current_value || 0).toFixed(4)} / {l.threshold}</span>
              </div>
            ))}
          </div>
          <div style={glassCard}>
            <SectionHeader title="Recent Alerts" icon={Bell} />
            {(rm.alerts?.recent || []).length > 0 ? (
              (rm.alerts?.recent || []).map((a, i) => (
                <div key={i} style={{ padding: '10px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', marginBottom: '10px', borderLeft: `4px solid ${theme.colors.danger}` }}>
                  <div style={{ fontSize: '10px', color: theme.colors.textMuted }}>{new Date(a.timestamp).toLocaleTimeString()}</div>
                  <div style={{ fontWeight: '700' }}>{a.limit_name} Breach</div>
                </div>
              ))
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: theme.colors.textMuted }}>No active risk alerts</div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderMarketView = () => (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
      <div style={glassCard}>
        <SectionHeader title="Intelligence Feed" icon={Globe} />
        {data.news.map((n, i) => {
          const Wrapper = n.url ? 'a' : 'div';
          const wrapperProps = n.url
            ? { href: n.url, target: '_blank', rel: 'noopener noreferrer', style: { display: 'block', textDecoration: 'none', color: 'inherit', cursor: 'pointer' } }
            : {};
          return (
            <Wrapper key={i} {...wrapperProps}>
              <div style={{ padding: '15px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                  <span style={{ fontSize: '10px', fontWeight: '800', color: n.sentiment_label === 'positive' ? theme.colors.primary : (n.sentiment_label === 'negative' ? theme.colors.danger : theme.colors.textMuted) }}>
                    {n.sentiment_label?.toUpperCase() || 'NEUTRAL'}{n.region === 'east_africa' ? ' · EAST AFRICA' : ''}
                  </span>
                  <span style={{ fontSize: '10px', color: theme.colors.textMuted }}>{new Date(n.time).toLocaleTimeString()}</span>
                </div>
                <div style={{ fontWeight: '700', marginBottom: '5px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {n.title}{n.url && <ExternalLink size={12} color={theme.colors.textMuted} />}
                </div>
                <div style={{ fontSize: '12px', color: theme.colors.textSecondary }}>{n.summary}</div>
                {n.source && <div style={{ fontSize: '10px', color: theme.colors.textMuted, marginTop: '4px' }}>{n.source}</div>}
              </div>
            </Wrapper>
          );
        })}
      </div>
      <div style={glassCard}>
        <SectionHeader title="Market Heatmap" icon={Layers} />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
          {data.heatmap.map((s, i) => {
            const score = s.outlook_score ?? 0.0;
            let cellBg = 'rgba(255, 255, 255, 0.04)';
            if (score > 0.05) {
              cellBg = `rgba(16, 185, 129, ${0.1 + score * 0.7})`;
            } else if (score < -0.05) {
              cellBg = `rgba(244, 63, 94, ${0.1 + Math.abs(score) * 0.7})`;
            }
            
            return (
              <div 
                key={i} 
                onClick={() => setDrilldownSector(s.sector)}
                title={`Drill into ${s.sector} sector outlook`}
                style={{ 
                  aspectRatio: '1', 
                  borderRadius: '10px', 
                  background: cellBg,
                  border: `1px solid ${score > 0.1 ? theme.colors.primary : (score < -0.1 ? theme.colors.danger : theme.colors.border)}40`,
                  display: 'flex', 
                  flexDirection: 'column',
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  textAlign: 'center', 
                  fontSize: '10px', 
                  fontWeight: '800',
                  cursor: 'pointer',
                  padding: '8px',
                  boxShadow: score > 0.4 ? '0 0 10px rgba(16, 185, 129, 0.15)' : (score < -0.4 ? '0 0 10px rgba(244, 63, 94, 0.15)' : 'none'),
                  transition: 'all 0.2s ease',
                }}
              >
                <div style={{ color: '#fff', fontSize: '10px', marginBottom: '4px' }}>{s.sector}</div>
                <div style={{ fontSize: '9px', color: theme.colors.textSecondary, fontWeight: 'normal' }}>
                  ETF: {s.performance >= 0 ? '+' : ''}{((s.performance || 0) * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: '9px', color: score >= 0.1 ? theme.colors.primary : (score <= -0.1 ? theme.colors.danger : theme.colors.textMuted), fontWeight: 'bold', marginTop: '2px' }}>
                  Score: {score >= 0 ? '+' : ''}{score.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  const renderPulseView = () => (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div style={glassCard}>
          <SectionHeader title="System Status" icon={Cpu} />
          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Uptime</span><span>{Math.floor((data.systemHealth?.uptime || 0) / 3600)}h</span></div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px' }}><span>Memory</span><span style={{ color: theme.colors.primary }}>{(data.systemHealth?.memory_usage || 'stable').toUpperCase()}</span></div>
        </div>
        <div style={glassCard}>
          <SectionHeader title="Service Health" icon={Activity} />
          {(() => {
            const comps = data.status?.components || {};
            const names = Object.keys(comps);
            if (names.length === 0) {
              return <div style={{ color: theme.colors.textMuted, fontSize: '12px' }}>Awaiting status…</div>;
            }
            return names.map(name => {
              const c = comps[name] || {};
              // A component is healthy if it reports running/connected/initialized truthy.
              const up = c.is_running ?? c.is_connected ?? c.initialized ?? true;
              return (
                <div key={name} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ textTransform: 'capitalize' }}>{name.replace(/_/g, ' ')}</span>
                  <span style={{ color: up ? theme.colors.primary : theme.colors.danger }}>{up ? 'ONLINE' : 'OFFLINE'}</span>
                </div>
              );
            });
          })()}
        </div>
      </div>
      <div style={glassCard}>
        <SectionHeader title="Agent Log" icon={Terminal} />
        <div style={{ backgroundColor: '#000', padding: '15px', borderRadius: '8px', height: '400px', overflowY: 'auto', fontFamily: 'monospace', fontSize: '11px' }}>
          {data.agentActivity.length > 0 ? (
            data.agentActivity.map((log, i) => (
              <div key={i} style={{ marginBottom: '8px' }}><span style={{ color: theme.colors.textMuted }}>[{new Date(log.timestamp).toLocaleTimeString()}]</span> <span style={{ color: theme.colors.primary }}>{log.component}</span>: {log.message}</div>
            ))
          ) : (
            <div style={{ color: theme.colors.textMuted }}>Waiting for logs...</div>
          )}
        </div>
      </div>
    </div>
  );

  const renderAnalytics = () => (
    <AdvancedAnalytics data={data} />
  );

  const tabs = [
    { id: 'portfolio', label: 'PORTFOLIO & WEALTH', icon: Globe, component: () => <UnifiedPortfolio onDrill={setDrilldownSymbol} /> },
    { id: 'overview', label: 'COMMAND DESK', icon: Activity, component: renderOverview },
    { id: 'research', label: 'RESEARCH & AUTO-PILOT', icon: FileText, component: () => <ResearchView activeTab={activeTab} fetchData={fetchData} onDrill={setDrilldownSymbol} /> },
    { id: 'nse', label: 'NSE KENYA', icon: Flag, component: renderKenyaNSE },
    { id: 'analytics', label: 'ANALYTICS', icon: TrendingUp, component: renderAnalytics },
    { id: 'risk', label: 'RISK & CONTROLS', icon: Shield, component: renderRiskView },
    { id: 'market', label: 'MARKET FEEDS', icon: Globe, component: renderMarketView },
    { id: 'system', label: 'SYSTEM PULSE', icon: Cpu, component: renderPulseView }
  ];

  return (
    <div style={{ minHeight: '100vh', backgroundColor: theme.colors.bg, color: theme.colors.text, fontFamily: 'Outfit, sans-serif' }}>
      {/* EMPIRE SLY VAULT Executive Header */}
      <header style={{ 
        ...theme.glass, borderRadius: 0, padding: '16px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        borderBottom: `1px solid ${theme.colors.border}`, background: 'rgba(2, 6, 23, 0.95)', backdropFilter: 'blur(16px)',
        flexWrap: 'wrap', gap: '16px'
      }}>
        {/* Brand Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ 
            background: 'linear-gradient(135deg, #10b981 0%, #3b82f6 100%)', 
            padding: '10px 12px', borderRadius: '12px', boxShadow: '0 0 15px rgba(16, 185, 129, 0.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <Shield color="#000" size={22} strokeWidth={2.5} />
          </div>
          <div>
            <h1 style={{ fontSize: '20px', fontWeight: '900', margin: 0, letterSpacing: '1.5px', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
              EMPIRE SLY VAULT
              <span style={{ fontSize: '10px', backgroundColor: 'rgba(16,185,129,0.15)', color: theme.colors.primary, padding: '2px 8px', borderRadius: '4px', border: `1px solid ${theme.colors.primary}40`, fontWeight: '800' }}>
                QUANTUM AI
              </span>
            </h1>
            <div style={{ fontSize: '11px', color: theme.colors.textMuted, fontWeight: '700', letterSpacing: '0.5px' }}>
              MULTI-REGION WEALTH & AUTONOMOUS DESK
            </div>
          </div>
        </div>

        {/* Grouped Sleek Navigation Pills */}
        <nav style={{ 
          display: 'flex', gap: '4px', backgroundColor: 'rgba(0, 0, 0, 0.4)', padding: '5px', 
          borderRadius: '12px', border: `1px solid ${theme.colors.border}60`
        }}>
          {tabs.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button 
                key={tab.id} 
                onClick={() => setActiveTab(tab.id)} 
                style={{ 
                  background: isActive ? 'linear-gradient(135deg, rgba(16,185,129,0.2) 0%, rgba(59,130,246,0.2) 100%)' : 'transparent', 
                  border: isActive ? `1px solid ${theme.colors.primary}60` : '1px solid transparent', 
                  color: isActive ? '#fff' : theme.colors.textSecondary, 
                  fontSize: '11px', fontWeight: '800', cursor: 'pointer', 
                  borderRadius: '8px', padding: '8px 14px', display: 'flex', alignItems: 'center', gap: '6px',
                  transition: 'all 0.2s ease', letterSpacing: '0.5px'
                }}
              >
                <Icon size={14} color={isActive ? theme.colors.primary : theme.colors.textMuted} />
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* Controls & Quick Drill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <MarketClock />
          <input
            placeholder="Drill symbol…"
            title="Type a ticker + Enter to open its drill-down"
            onKeyDown={(e) => { if (e.key === 'Enter' && e.target.value.trim()) { setDrilldownSymbol(e.target.value.trim().toUpperCase()); e.target.value = ''; } }}
            style={{ background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border}`, color: theme.colors.text, padding: '7px 12px', borderRadius: '8px', fontSize: '11px', width: '110px', outline: 'none' }}
          />
          {isOperator && (
            <button
              onClick={toggleHalt}
              title={isHalted ? 'Resume automated trading' : 'Stop all new orders immediately'}
              style={{
                background: isHalted ? theme.colors.warning : 'rgba(244, 63, 94, 0.12)',
                border: `1px solid ${isHalted ? theme.colors.warning : theme.colors.danger}`,
                color: isHalted ? '#000' : theme.colors.danger,
                padding: '7px 14px', borderRadius: '8px', cursor: 'pointer',
                fontSize: '11px', fontWeight: 800, letterSpacing: '0.5px',
              }}
            >
              {isHalted ? 'RESUME' : 'HALT'}
            </button>
          )}
          <button onClick={() => setShowHelp(true)} title="Getting started guide" style={{ background: 'transparent', border: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, width: '32px', height: '32px', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: 800 }}>?</button>
          <button onClick={onLogout} style={{ background: 'transparent', border: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, padding: '7px 12px', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: '800' }}><LogOut size={14} /> EXIT</button>
        </div>
      </header>
      {isHalted && (
        <div style={{
          backgroundColor: theme.colors.warning, color: '#000', textAlign: 'center',
          padding: '8px', fontSize: '13px', fontWeight: 800, letterSpacing: '1px',
        }}>
          ⚠ TRADING HALTED — the agent is not submitting new orders. Protective stop-losses remain active.
        </div>
      )}
      {isOperator && anomalies.length > 0 && (
        <div style={{ padding: '16px 40px 0' }}>
          <div style={{ ...glassCard, padding: '16px 20px', borderLeft: `3px solid ${theme.colors.warning}` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
              <AlertTriangle size={16} color={theme.colors.warning} />
              <span style={{ fontSize: '12px', fontWeight: 800, textTransform: 'uppercase', color: theme.colors.textSecondary }}>
                {anomalies.length} thing{anomalies.length > 1 ? 's' : ''} worth a look
              </span>
            </div>
            {anomalies.slice(0, 5).map((a, i) => {
              const c = a.severity === 'high' ? theme.colors.danger : a.severity === 'medium' ? theme.colors.warning : theme.colors.textMuted;
              return (
                <div key={i}
                  onClick={() => a.symbol && setDrilldownSymbol(a.symbol)}
                  style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '6px 0', fontSize: '13px', cursor: a.symbol ? 'pointer' : 'default' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: c, flexShrink: 0 }} />
                  <span style={{ color: theme.colors.text }}>{a.message}</span>
                  {a.symbol && <span style={{ color: theme.colors.textMuted, fontSize: '10px' }}>↗</span>}
                </div>
              );
            })}
          </div>
        </div>
      )}
      <main style={{ padding: '40px' }}>{tabs.find(t => t.id === activeTab)?.component()}</main>
      {showHelp && <HelpPanel onClose={() => setShowHelp(false)} />}
      {drilldownSymbol && <SymbolDrilldown symbol={drilldownSymbol} onClose={() => setDrilldownSymbol(null)} />}
      {drilldownSector && <SectorDrilldown sector={drilldownSector} onClose={() => setDrilldownSector(null)} />}
    </div>
  );
};

export default AdvancedDashboard;