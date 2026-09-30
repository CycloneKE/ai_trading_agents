// Where the agent is now, against where the paper run needs it to be
// (/api/scorecard, src/agent/scorecard.py). Every measure has a target and a
// plain status; "too early" means there is not yet enough evidence to judge.
import { useEffect, useState } from 'react';
import { Target } from 'lucide-react';
import { theme } from '../DashboardStyles';
import { apiGet, card, SectionHeader, Empty, Note } from '../ui';

const STATUS = {
  pass: { label: 'PASS', color: theme.colors.primary },
  watch: { label: 'WATCH', color: theme.colors.warning },
  fail: { label: 'FAIL', color: theme.colors.danger },
  too_early: { label: 'TOO EARLY', color: theme.colors.textMuted },
  info: { label: 'INFO', color: theme.colors.textSecondary },
};

const LEVEL_COLOR = {
  needs_attention: theme.colors.danger,
  off_track: theme.colors.warning,
  too_early: theme.colors.accent,
  on_track: theme.colors.primary,
  meets_bar: theme.colors.primary,
};

const Pill = ({ status }) => {
  const s = STATUS[status] || STATUS.info;
  return (
    <span style={{
      flexShrink: 0, minWidth: '74px', textAlign: 'center', padding: '3px 8px', borderRadius: '999px',
      fontSize: '10px', fontWeight: 800, letterSpacing: '0.4px', color: s.color,
      background: `${s.color}22`, border: `1px solid ${s.color}66`,
    }}>
      {s.label}
    </span>
  );
};

const MetricRow = ({ m, mobile }) => (
  <div style={{ padding: '12px 0', borderBottom: `1px solid ${theme.colors.border}` }}>
    <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', flexDirection: mobile ? 'column' : 'row' }}>
      <Pill status={m.status} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 700, fontSize: '13px' }}>{m.label}</span>
          <span style={{ fontSize: '13px', color: (STATUS[m.status] || STATUS.info).color, overflowWrap: 'anywhere' }}>{m.display}</span>
        </div>
        {m.target && <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '2px' }}>Target: {m.target}</div>}
        {m.note && <div style={{ fontSize: '12px', color: theme.colors.textSecondary, marginTop: '4px', lineHeight: 1.5 }}>{m.note}</div>}
      </div>
    </div>
  </div>
);

export default function Scorecard({ mobile }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = () => apiGet('/api/scorecard')
      .then(({ ok, status, json }) => {
        if (!alive) return;
        if (ok && json && json.sections) { setData(json); setError(null); }
        else setError((json && json.error) || `HTTP ${status}`);
      })
      .catch((e) => alive && setError(e.message));
    load();
    const id = setInterval(load, 300000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!data) {
    return (
      <div style={card(mobile)}>
        <SectionHeader title="Agent scorecard" icon={Target} />
        {error ? <Empty>The scorecard could not load ({error}).</Empty> : <Empty>Scoring the agent… the first load can take a little while.</Empty>}
      </div>
    );
  }

  const { verdict, run, counts } = data;
  const tone = LEVEL_COLOR[verdict.level] || theme.colors.accent;
  return (
    <div style={{ display: 'grid', gap: mobile ? '16px' : '24px' }}>
      <div style={card(mobile)}>
        <SectionHeader title="Agent scorecard: now against where it needs to be" icon={Target} />
        <Note tone={tone}>
          <strong style={{ fontSize: '14px' }}>{verdict.title}.</strong> {verdict.text}
        </Note>
        <div style={{ marginTop: '14px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: theme.colors.textSecondary, marginBottom: '6px' }}>
            <span>Paper run: day {run.day} of {run.planned_days}</span>
            <span>{run.percent_through}% through</span>
          </div>
          <div style={{ height: '6px', borderRadius: '999px', background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
            <div style={{ width: `${Math.min(run.percent_through || 0, 100)}%`, height: '100%', background: theme.colors.accent }} />
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '14px' }}>
          {['pass', 'watch', 'fail', 'too_early', 'info'].map((k) => (
            <span key={k} style={{ fontSize: '12px', color: STATUS[k].color }}>
              <strong>{counts[k] || 0}</strong> {STATUS[k].label.toLowerCase()}
            </span>
          ))}
        </div>
        <div style={{ fontSize: '11px', color: theme.colors.textMuted, marginTop: '12px', lineHeight: 1.5 }}>
          Targets are proposals kept in config.json (scorecard.targets). Nothing here predicts the future: every figure is
          measured from what the agent did and what the market then did.
        </div>
      </div>
      {data.sections.map((sec) => (
        <div key={sec.id} style={card(mobile)}>
          <SectionHeader title={sec.title} icon={Target} />
          <div style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '4px' }}>{sec.question}</div>
          {sec.metrics.map((m) => <MetricRow key={m.id} m={m} mobile={mobile} />)}
        </div>
      ))}
    </div>
  );
}
