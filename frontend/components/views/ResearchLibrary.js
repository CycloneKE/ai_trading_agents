// The research library (/api/research/library, src/agent/research_notes.py):
// documents with no ratings (IPO briefings, results, policy and sector notes)
// that the agent has read and kept. Every figure carries the exact words it
// came from, so it can be checked against the document.
import { useState } from 'react';
import { BookOpen } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { getApiBase } from '../../utils/apiBase';
import { card, SectionHeader } from '../ui';

const TYPE = {
  ipo_briefing: 'IPO briefing', company_results: 'Company results', economic_report: 'Economic report',
  policy_notice: 'Policy or regulatory notice', sector_report: 'Sector report',
  commodity_report: 'Commodity report', market_commentary: 'Market commentary', other: 'Research document',
};

export const NoteCard = ({ n, onRemove, busy }) => (
  <div style={{ border: `1px solid ${theme.colors.border}`, borderRadius: '10px', padding: '14px', opacity: n.expired ? 0.6 : 1 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap', alignItems: 'baseline' }}>
      <strong>{n.title || n.filename}</strong>
      <span style={{ fontSize: '11px', fontWeight: 800, color: n.expired ? theme.colors.textMuted : theme.colors.primary }}>
        {n.expired ? 'NO LONGER SHOWN TO THE AI' : `SHOWN TO THE AI UNTIL ${n.expires_on}`}
      </span>
    </div>
    <div style={{ fontSize: '12px', color: theme.colors.textMuted, margin: '4px 0 8px' }}>
      {TYPE[n.document_type] || 'Document'} · {n.as_of}{n.date_read === false ? ' (date not printed; upload date used)' : ''}
      {n.symbols && n.symbols.length > 0 ? ` · ${n.symbols.join(', ')}` : ''}
      {n.markets && n.markets.length > 0 ? ` · ${n.markets.join(', ')}` : ''}
    </div>
    {n.read_by_ai === false && (
      <div style={{ fontSize: '12px', color: theme.colors.warning, marginBottom: '8px' }}>
        No AI was available, so only the text was kept. Add GROQ_API_KEY or GEMINI_API_KEY in Coolify and upload it again.
      </div>
    )}
    {n.summary && <div style={{ fontSize: '13px', color: theme.colors.textSecondary, lineHeight: 1.5 }}>{n.summary}</div>}
    {(n.facts || []).length > 0 && (
      <div style={{ overflowX: 'auto', marginTop: '10px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
          <tbody>
            {n.facts.map((f, i) => (
              <tr key={i} style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary }}>
                <td style={{ padding: '5px 8px 5px 0', whiteSpace: 'nowrap' }}>{f.label}</td>
                <td style={{ padding: '5px 8px', fontWeight: 700 }}>{f.value}</td>
                <td style={{ padding: '5px 0', color: theme.colors.textMuted }}>&ldquo;{f.quote}&rdquo;</td>
              </tr>
            ))}
            {(n.dates || []).map((d, i) => (
              <tr key={`d${i}`} style={{ borderBottom: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary }}>
                <td style={{ padding: '5px 8px 5px 0', whiteSpace: 'nowrap' }}>{d.label}</td>
                <td style={{ padding: '5px 8px', fontWeight: 700 }}>{d.date}</td>
                <td style={{ padding: '5px 0', color: theme.colors.textMuted }}>&ldquo;{d.quote}&rdquo;</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )}
    {(n.risks || []).length > 0 && (
      <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginTop: '8px' }}>Risks it names: {n.risks.join('; ')}</div>
    )}
    <div style={{ marginTop: '10px' }}>
      <button onClick={() => onRemove(n.id)} disabled={busy} style={{ background: 'transparent', border: `1px solid ${theme.colors.border}`, color: theme.colors.textSecondary, borderRadius: '6px', padding: '4px 10px', fontSize: '11px', cursor: 'pointer' }}>
        Remove from the library
      </button>
    </div>
  </div>
);

export default function ResearchLibrary({ notes, mobile, token, onChanged }) {
  const [busy, setBusy] = useState(false);
  if (!notes || notes.length === 0) return null;

  const remove = async (id) => {
    if (typeof window !== 'undefined' && !window.confirm('Remove this document? The AI will no longer see it.')) return;
    setBusy(true);
    try {
      await fetch(`${getApiBase()}/api/operator/research-library/${id}`, { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } });
      if (onChanged) await onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={card(mobile)}>
      <SectionHeader title="Research Library" icon={BookOpen} />
      <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '10px', lineHeight: 1.5 }}>
        Documents with no buy or sell ratings, such as IPO briefings, results and policy notices. Each figure shows the
        exact words it came from. For about three months the AI sees a short line from each document that names a stock
        when it reviews a trade in it. A document never places a trade or changes a setting by itself.
      </div>
      <div style={{ display: 'grid', gap: '16px' }}>
        {notes.map((n) => <NoteCard key={n.id} n={n} onRemove={remove} busy={busy} />)}
      </div>
    </div>
  );
}
