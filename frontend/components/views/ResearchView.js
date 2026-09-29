// Research: broker PDF ingestion, the operator approval queue and the
// symbol watchlist the research feeds.
import { useState, useEffect, useCallback } from 'react';
import { Upload, Clock, AlertTriangle, Layers, FileText, CheckCircle, XCircle } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { getApiBase } from '../../utils/apiBase';
import { card, SectionHeader, localTime } from '../ui';

const DOC_LABEL = {
  market_pulse: 'Market Pulse',
  recommendation_sheet: 'Rating sheet',
  analyst_note: 'Analyst note',
  bond_auction: 'Bond auction',
};

// Prices on a rating sheet are in shillings for a Nairobi stock and dollars for a US one.
const priceText = (market, v) => (v == null ? 'N/A' : `${market === 'international' ? '$' : 'KES '}${Number(v).toFixed(2)}`);
const RATING_COLOR = (r) => (r === 'SELL' || r === 'REDUCE' ? theme.colors.danger : (r === 'HOLD' ? theme.colors.warning : theme.colors.primary));

const ResearchView = ({ active, onChanged, onDrill, mobile }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [escalations, setEscalations] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [uploads, setUploads] = useState([]);
  const [pulse, setPulse] = useState(null);
  const [ratings, setRatings] = useState([]);
  const [bonds, setBonds] = useState([]);

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

      const mpRes = await fetch(`${getApiBase()}/api/research/market-pulse`, { headers: { Authorization: `Bearer ${token}` } });
      if (mpRes.ok) setPulse(await mpRes.json());

      const rtRes = await fetch(`${getApiBase()}/api/research/ratings`, { headers: { Authorization: `Bearer ${token}` } });
      if (rtRes.ok) setRatings((await rtRes.json()).ratings || []);

      const bdRes = await fetch(`${getApiBase()}/api/research/bond-auctions`, { headers: { Authorization: `Bearer ${token}` } });
      if (bdRes.ok) setBonds((await bdRes.json()).auctions || []);
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

  const handleFollow = async (symbol) => {
    try {
      const res = await fetch(`${getApiBase()}/api/research/ratings/${symbol}/follow`, {
        method: 'POST', headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) loadResearchData();
    } catch (e) {
      console.error("Follow request failed", e);
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
          <SectionHeader title="Upload Research" icon={Upload} />
          <div style={{ border: `2px dashed ${theme.colors.border}`, borderRadius: '12px', padding: mobile ? '20px 12px' : '30px', textAlign: 'center', backgroundColor: 'rgba(255,255,255,0.01)', position: 'relative' }}>
            {!selectedFile && (
              <input type="file" onChange={handleFileChange} accept=".pdf,.jpg,.jpeg,.png,.webp" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }} />
            )}
            <FileText size={40} color={selectedFile ? theme.colors.primary : theme.colors.textMuted} style={{ marginBottom: '12px' }} />
            {selectedFile ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                <p style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: 'bold' }}>{selectedFile.name}</p>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button onClick={(e) => { e.stopPropagation(); handleUpload(); }} disabled={uploading} style={{ backgroundColor: theme.colors.primary, color: '#000', border: 'none', padding: '8px 20px', borderRadius: '8px', fontSize: '13px', fontWeight: '800', cursor: 'pointer' }}>
                    {uploading ? 'Reading…' : 'Read Document'}
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }} style={{ backgroundColor: 'transparent', color: theme.colors.danger, border: `1px solid ${theme.colors.danger}`, padding: '8px 20px', borderRadius: '8px', fontSize: '13px', fontWeight: '800', cursor: 'pointer' }}>
                    Clear
                  </button>
                </div>
              </div>
            ) : (
              <div>
                <p style={{ margin: 0, fontSize: '13px', color: theme.colors.textSecondary }}>Drag & drop or click to select an AIB-AXYS report (PDF) or rating sheet (JPG, PNG)</p>
                <p style={{ margin: '5px 0 0 0', fontSize: '11px', color: theme.colors.textMuted }}>
                  Market Pulse PDF: every stock's figures and closing price, T-bill rates and announcements, read exactly.
                  Daily Whispers picture (Kenyan or Global Equity sheet): each BUY/HOLD rating and target, read by AI and then checked row by row.
                  Bond Auction Note PDF: the papers on offer, sale dates and AXYS's bidding ranges, read exactly.
                  After reading, the agent lists below exactly what it did with the document.
                </p>
              </div>
            )}
          </div>
          
          {uploadResult && (
            <div style={{ marginTop: '20px', padding: '15px', borderRadius: '8px', border: `1px solid ${uploadResult.status === 'completed' ? theme.colors.primary : theme.colors.danger}`, backgroundColor: `${uploadResult.status === 'completed' ? theme.colors.primary : theme.colors.danger}10` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '700', fontSize: '14px', color: uploadResult.status === 'completed' ? theme.colors.primary : theme.colors.danger }}>
                {uploadResult.status === 'completed' ? <CheckCircle size={16} /> : <XCircle size={16} />}
                <span>{uploadResult.status === 'completed' ? `${DOC_LABEL[uploadResult.document_type] || 'Document'} read: what the agent did` : 'Could not be read'}</span>
              </div>
              {(uploadResult.actions || []).length > 0 ? (
                <ul style={{ fontSize: '12px', margin: '8px 0 0 0', paddingLeft: '18px', color: theme.colors.textSecondary, display: 'grid', gap: '4px' }}>
                  {uploadResult.actions.map((a, i) => <li key={i}>{a}</li>)}
                </ul>
              ) : (
                <p style={{ fontSize: '12px', margin: '5px 0 0 0', color: theme.colors.textMuted }}>{uploadResult.error || 'Check server logs for details'}</p>
              )}
            </div>
          )}
        </div>

        {pulse?.latest && (
          <div style={card(mobile, { flex: '1 1 340px', minWidth: 0 })}>
            <SectionHeader title="Latest Market Pulse" icon={FileText} />
            <div style={{ fontSize: '12px', color: theme.colors.textSecondary, display: 'grid', gap: '6px' }}>
              <div>Report of <strong>{pulse.latest.as_of}</strong>; {pulse.reports} read so far. Fundamentals held for {pulse.stocks_with_fundamentals} stocks: they feed the dividend sleeve, the Market Scan and the AI's review of NSE trades.</div>
              {pulse.rates?.tbill_91 && (
                <div>T-bills: 91-day {(pulse.rates.tbill_91 * 100).toFixed(2)}%, 182-day {((pulse.rates.tbill_182 || 0) * 100).toFixed(2)}%, 364-day {((pulse.rates.tbill_364 || 0) * 100).toFixed(2)}%{pulse.rates.usd_kes ? `; USD ${pulse.rates.usd_kes} KES` : ''}.</div>
              )}
              {(pulse.announcements || []).length > 0 && (
                <div style={{ maxHeight: '140px', overflowY: 'auto', borderTop: `1px solid ${theme.colors.border}`, paddingTop: '6px' }}>
                  {pulse.announcements.map((a, i) => (
                    <div key={i} style={{ padding: '3px 0' }}>
                      <span style={{ color: theme.colors.textMuted }}>{a.date}</span>{' '}
                      {a.symbol && <strong>{a.symbol}</strong>}{' '}
                      {a.link
                        ? <a href={a.link} target="_blank" rel="noopener noreferrer" style={{ color: theme.colors.textSecondary }}>{a.text} ↗</a>
                        : a.text}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        <div style={card(mobile, { flex: '1 1 340px', minWidth: 0 })}>
          <SectionHeader title="Documents Read" icon={Clock} />
          <div style={{ maxHeight: '320px', overflowY: 'auto', overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, textAlign: 'left' }}>
                  <th style={{ padding: '8px 0' }}>DOCUMENT AND WHAT THE AGENT DID</th>
                  <th style={{ padding: '8px 0' }}>UPLOADED</th>
                  <th style={{ padding: '8px 0' }}>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {uploads.length > 0 ? (
                  uploads.map(u => (
                    <tr key={u.id} style={{ borderBottom: `1px solid ${theme.colors.border}20`, verticalAlign: 'top' }}>
                      <td style={{ padding: '10px 8px 10px 0' }}>
                        <div style={{ fontWeight: 'bold', wordBreak: 'break-all' }}>
                          {u.document_type && <span style={{ fontSize: '9px', fontWeight: 800, padding: '1px 5px', borderRadius: '4px', marginRight: '6px',
                            background: `${theme.colors.accent}25`, color: theme.colors.accent }}>{(DOC_LABEL[u.document_type] || u.document_type).toUpperCase()}</span>}
                          {u.filename}
                        </div>
                        {(u.summary || []).length > 0 ? (
                          <details style={{ marginTop: '4px', color: theme.colors.textSecondary }}>
                            <summary style={{ cursor: 'pointer' }}>{u.summary[0]}</summary>
                            <ul style={{ margin: '4px 0 0 0', paddingLeft: '16px' }}>{u.summary.slice(1).map((a, i) => <li key={i}>{a}</li>)}</ul>
                          </details>
                        ) : (
                          <div style={{ marginTop: '4px', color: theme.colors.textMuted }}>No record of what was done (uploaded before this was kept; upload it again to see).</div>
                        )}
                      </td>
                      <td style={{ padding: '10px 8px 10px 0', color: theme.colors.textMuted, whiteSpace: 'nowrap' }}>{localTime(u.uploaded_at)}</td>
                      <td style={{ padding: '10px 0', color: u.status === 'completed' ? theme.colors.primary : theme.colors.danger }}>{u.status.toUpperCase()}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan="3" style={{ padding: '20px 0', textAlign: 'center', color: theme.colors.textMuted }}>No documents read yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {ratings.length > 0 && (
        <div style={card(mobile)}>
          <SectionHeader title="Broker Ratings" icon={FileText} />
          <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '10px' }}>
            The latest rating of each stock from the rating sheets of the last 30 days. A rating never places a trade by itself.
            For a stock the agent trades, the AI sees its rating when it reviews a trade. Stocks it does not trade are kept here for reference; press Follow to ask for one.
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, textAlign: 'left' }}>
                  <th style={{ padding: '8px 0' }}>STOCK</th>
                  <th style={{ padding: '8px 0' }}>RATING</th>
                  <th style={{ padding: '8px 0' }}>PRICE</th>
                  <th style={{ padding: '8px 0' }}>TARGET (UPSIDE)</th>
                  <th style={{ padding: '8px 0' }}>DATE</th>
                  <th style={{ padding: '8px 0' }}>AGENT</th>
                  <th style={{ padding: '8px 0', textAlign: 'right' }}></th>
                </tr>
              </thead>
              <tbody>
                {ratings.map((r) => (
                  <tr key={r.symbol} style={{ borderBottom: `1px solid ${theme.colors.border}20`, verticalAlign: 'top' }}>
                    <td style={{ padding: '10px 8px 10px 0' }}>
                      <div style={{ fontWeight: 'bold' }}>{r.symbol} <span style={{ fontSize: '10px', color: theme.colors.textMuted }}>{r.market === 'international' ? (r.exchange || 'US') : 'NSE'}</span></div>
                      <div style={{ fontSize: '11px', color: theme.colors.textMuted }}>{r.name}</div>
                      {r.rationale && (
                        <details style={{ marginTop: '4px', fontSize: '11px', color: theme.colors.textSecondary, maxWidth: '420px' }}>
                          <summary style={{ cursor: 'pointer' }}>Why</summary>
                          <div style={{ marginTop: '4px', lineHeight: 1.4 }}>{r.rationale}</div>
                        </details>
                      )}
                    </td>
                    <td style={{ padding: '10px 8px 10px 0', color: RATING_COLOR(r.recommendation), fontWeight: 700 }}>{r.recommendation}</td>
                    <td style={{ padding: '10px 8px 10px 0' }}>{priceText(r.market, r.current_price)}</td>
                    <td style={{ padding: '10px 8px 10px 0' }}>{priceText(r.market, r.target_price)} ({r.upside_pct != null ? `${r.upside_pct > 0 ? '+' : ''}${r.upside_pct.toFixed(1)}%` : 'n/a'})</td>
                    <td style={{ padding: '10px 8px 10px 0', color: theme.colors.textMuted, whiteSpace: 'nowrap' }}>{r.date}</td>
                    <td style={{ padding: '10px 8px 10px 0' }}>
                      <span style={{ fontSize: '11px', fontWeight: 800, padding: '2px 8px', borderRadius: '4px', whiteSpace: 'nowrap',
                        background: r.tracked ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255,255,255,0.05)',
                        color: r.tracked ? theme.colors.primary : theme.colors.textMuted }}>{r.tracked ? 'TRADED' : 'NOT TRADED'}</span>
                    </td>
                    <td style={{ padding: '10px 0', textAlign: 'right' }}>
                      {!r.tracked && (r.follow_pending
                        ? <span style={{ fontSize: '11px', color: theme.colors.warning }}>Waiting in the queue</span>
                        : <button onClick={() => handleFollow(r.symbol)} style={{ backgroundColor: 'transparent', color: theme.colors.primary, border: `1px solid ${theme.colors.primary}`, padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: '800', cursor: 'pointer' }}>FOLLOW</button>)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {bonds.length > 0 && (
        <div style={card(mobile)}>
          <SectionHeader title="Bond Auctions" icon={FileText} />
          <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '10px' }}>
            Read from the uploaded AIB-AXYS auction notes. You bid through your broker; the agent trades no bonds and nothing here is queued for approval.
          </div>
          <div style={{ display: 'grid', gap: '16px' }}>
            {bonds.map((b) => {
              const badge = { open: ['OPEN', theme.colors.primary], upcoming: ['UPCOMING', theme.colors.warning], closed: ['CLOSED', theme.colors.textMuted] }[b.status] || ['', theme.colors.textMuted];
              return (
                <div key={`${b.title}-${b.sale_from}`} style={{ border: `1px solid ${theme.colors.border}`, borderRadius: '10px', padding: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap', alignItems: 'baseline' }}>
                    <strong>{b.title}</strong>
                    <span style={{ fontSize: '11px', fontWeight: 800, color: badge[1] }}>{badge[0]} · sale {b.sale_from} to {b.sale_to}</span>
                  </div>
                  <div style={{ fontSize: '12px', color: theme.colors.textSecondary, margin: '6px 0 10px' }}>
                    {b.issuer || 'Issuer not given'}{b.total_kes_bn != null ? `: KES ${b.total_kes_bn} billion` : ''}{b.purpose ? `, ${b.purpose.toLowerCase()}` : ''}.
                    {b.min_bid_kes != null ? ` Minimum bid KES ${Number(b.min_bid_kes).toLocaleString()}.` : ''}{b.tax_pct != null ? ` Tax on interest ${b.tax_pct}%.` : ''}
                  </div>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                      <thead>
                        <tr style={{ color: theme.colors.textSecondary, textAlign: 'left', borderBottom: `1px solid ${theme.colors.border}` }}>
                          <th style={{ padding: '6px 0' }}>PAPER</th><th>YEARS TO MATURITY</th><th>COUPON</th><th>MATURES</th><th>AXYS BIDDING RANGE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(b.papers || []).map((pp) => (
                          <tr key={pp.paper} style={{ borderBottom: `1px solid ${theme.colors.border}20` }}>
                            <td style={{ padding: '8px 0', fontWeight: 700 }}>{pp.paper}{pp.reopened ? <span style={{ fontWeight: 400, color: theme.colors.textMuted }}> re-opened</span> : null}</td>
                            <td>{pp.tenor_years ?? '—'}</td>
                            <td>{pp.coupon_pct != null ? `${pp.coupon_pct}%` : '—'}</td>
                            <td>{pp.maturity || '—'}</td>
                            <td style={{ fontWeight: 700 }}>{pp.bid_low_pct != null && pp.bid_high_pct != null ? `${pp.bid_low_pct} to ${pp.bid_high_pct}%` : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {b.market && Object.keys(b.market).length > 0 && (
                    <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '8px' }}>
                      {b.market.inflation_pct != null && <>Inflation {b.market.inflation_pct}% ({b.market.inflation_month}). </>}
                      {b.market.kesonia_pct != null && <>Interbank rate (KESONIA) {b.market.kesonia_pct}%. </>}
                      {b.market.borrowing_vs_target_pct != null && <>Domestic borrowing at {b.market.borrowing_vs_target_pct}% of target. </>}
                      {b.market.last_accepted_rate_pct != null && <>Last accepted rate {b.market.last_accepted_rate_pct}%.</>}
                    </div>
                  )}
                  {(b.warnings || []).length > 0 && (
                    <div style={{ fontSize: '11px', color: theme.colors.warning, marginTop: '6px' }}>Check: {b.warnings.join('; ')}.</div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

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
                        <span>{priceText(esc.market, esc.target_price)} ({esc.upside_pct ? `${esc.upside_pct.toFixed(1)}%` : '-%'})</span>
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
