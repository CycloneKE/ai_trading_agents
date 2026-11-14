// Keeps one broken card from blanking the whole dashboard.
//
// React unmounts everything when a component throws while drawing, so a
// single endpoint answering in an unexpected shape used to leave an empty
// page. Wrapped in this, only that card shows a short note (and a retry);
// the rest of the page keeps working.
import React from 'react';
import { theme } from './DashboardStyles';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error(`${this.props.name || 'A panel'} could not be drawn:`, error, info?.componentStack);
  }

  componentDidUpdate(prev) {
    // Moving to another page gives the panel a fresh start.
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" style={{
        padding: '16px 18px', borderRadius: '14px', fontSize: '13px',
        background: 'rgba(244, 63, 94, 0.06)', border: '1px solid rgba(244, 63, 94, 0.25)',
        color: theme.colors.textSecondary, display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap',
      }}>
        <span>
          <strong style={{ color: '#fff' }}>{this.props.name || 'This panel'}</strong> could not be shown.
          The rest of the page is unaffected; the details are in the browser console.
        </span>
        <button
          onClick={() => this.setState({ error: null })}
          style={{
            background: 'transparent', color: theme.colors.text, border: `1px solid ${theme.colors.border}`,
            borderRadius: '8px', padding: '5px 12px', fontSize: '12px', fontWeight: 700, cursor: 'pointer',
          }}
        >
          Try again
        </button>
      </div>
    );
  }
}
