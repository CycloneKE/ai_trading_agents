// Research: broker PDF ingestion, the operator approval queue and the
// symbol watchlist the research feeds.
import { useState, useEffect, useCallback } from 'react';
import { Upload, Clock, AlertTriangle, Layers, FileText, CheckCircle, XCircle } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { getApiBase } from '../../utils/apiBase';
import { card, SectionHeader } from '../ui';

const ResearchView = ({ active, onChanged, onDrill, mobile }) => {
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
    if (active) loadResearchData();
  }, [active, loadResearchData]);

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
        onChanged();
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? '16px' : '24px' }}>
      <div style={{ display: 'flex', gap: mobile ? '16px' : '24px', flexWrap: 'wrap' }}>
        <div style={card(mobile, { flex: '1 1 340px', minWidth: 0 })}>
          <SectionHeader title="Research PDF Ingestion" icon={Upload} />
          <div style={{ border: `2px dashed ${theme.colors.border}`, borderRadius: '12px', padding: mobile ? '20px 12px' : '30px', textAlign: 'center', backgroundColor: 'rgba(255,255,255,0.01)', position: 'relative' }}>
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

        <div style={card(mobile, { flex: '1 1 340px', minWidth: 0 })}>
          <SectionHeader title="Ingest Archives" icon={Clock} />
          <div style={{ maxHeight: '190px', overflowY: 'auto', overflowX: 'auto' }}>
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

      <div style={card(mobile)}>
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

      <div style={card(mobile)}>
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

export default ResearchView;
