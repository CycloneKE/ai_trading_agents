import { useState } from 'react';
import { Shield, Zap, Lock, User, AlertCircle, ArrowRight } from 'lucide-react';
import { theme, glassCard } from './DashboardStyles';
import { getApiBase } from '../utils/apiBase';

const Login = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch(`${getApiBase()}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem('trading_token', data.token);
        localStorage.setItem('trading_user', data.username);
        onLoginSuccess(data.token);
      } else {
        setError(data.error || 'Invalid credentials. Access denied.');
      }
    } catch (err) {
      setError('System offline. Connection to authentication server failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center', 
      backgroundColor: theme.colors.bg,
      backgroundImage: 'radial-gradient(circle at 50% 50%, #1e293b 0%, #020617 100%)',
      padding: '20px'
    }}>
      <div style={{ ...glassCard, width: '100%', maxWidth: '420px', padding: '48px', position: 'relative' }}>
        {/* Glow effect */}
        <div style={{ 
          position: 'absolute', top: '-20px', left: '50%', transform: 'translateX(-50%)', 
          width: '100px', height: '100px', backgroundColor: theme.colors.primary, 
          filter: 'blur(60px)', opacity: 0.15, borderRadius: '50%', zIndex: 0 
        }} />

        <div style={{ textAlign: 'center', marginBottom: '40px', position: 'relative', zIndex: 1 }}>
          <div style={{ 
            backgroundColor: theme.colors.primary, width: '64px', height: '64px', 
            borderRadius: '16px', display: 'flex', alignItems: 'center', 
            justifyContent: 'center', margin: '0 auto 20px auto',
            boxShadow: '0 0 30px rgba(16, 185, 129, 0.4)'
          }}>
            <Shield color="#000" size={32} />
          </div>
          <h1 style={{ fontSize: '24px', fontWeight: '800', margin: '0 0 8px 0', letterSpacing: '-1px' }}>
            AEGIS-TRADER <span style={{ color: theme.colors.primary }}>AI</span>
          </h1>
          <p style={{ fontSize: '11px', color: theme.colors.textSecondary, textTransform: 'uppercase', letterSpacing: '2px', fontWeight: '700' }}>
            Institutional Access Required
          </p>
        </div>

        <form onSubmit={handleLogin} style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '12px', color: theme.colors.textSecondary, marginBottom: '8px', fontWeight: '800' }}>
              OPERATOR IDENTITY
            </label>
            <div style={{ position: 'relative' }}>
              <span style={{ position: 'absolute', left: '12px', top: '12px', color: theme.colors.textMuted }}>
                <User size={18} />
              </span>
              <input 
                type="text" 
                value={username} 
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Username"
                style={{ 
                  width: '100%', padding: '12px 12px 12px 40px', borderRadius: '10px', 
                  backgroundColor: 'rgba(2, 6, 23, 0.5)', border: `1px solid ${theme.colors.border}`,
                  color: '#fff', fontSize: '14px' 
                }}
                required
              />
            </div>
          </div>

          <div style={{ marginBottom: '24px' }}>
            <label style={{ display: 'block', fontSize: '12px', color: theme.colors.textSecondary, marginBottom: '8px', fontWeight: '800' }}>
              ACCESS CODE
            </label>
            <div style={{ position: 'relative' }}>
              <span style={{ position: 'absolute', left: '12px', top: '12px', color: theme.colors.textMuted }}>
                <Lock size={18} />
              </span>
              <input 
                type="password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                style={{ 
                  width: '100%', padding: '12px 12px 12px 40px', borderRadius: '10px', 
                  backgroundColor: 'rgba(2, 6, 23, 0.5)', border: `1px solid ${theme.colors.border}`,
                  color: '#fff', fontSize: '14px' 
                }}
                required
              />
            </div>
          </div>

          {error && (
            <div style={{ 
              backgroundColor: `${theme.colors.danger}15`, border: `1px solid ${theme.colors.danger}40`, 
              padding: '12px', borderRadius: '8px', color: theme.colors.danger, 
              fontSize: '12px', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' 
            }}>
              <AlertCircle size={14} />
              {error}
            </div>
          )}

          <button 
            type="submit" 
            disabled={isLoading}
            style={{ 
              width: '100%', padding: '16px', borderRadius: '12px', 
              backgroundColor: theme.colors.primary, color: '#000', 
              fontWeight: '800', border: 'none', cursor: isLoading ? 'wait' : 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
              transition: 'all 0.2s', boxShadow: '0 4px 20px rgba(16, 185, 129, 0.2)'
            }}
          >
            {isLoading ? 'ESTABLISHING SECURE LINK...' : (
              <>
                INITIALIZE TERMINAL <ArrowRight size={18} />
              </>
            )}
          </button>
        </form>

        <div style={{ marginTop: '32px', textAlign: 'center' }}>
          <p style={{ fontSize: '10px', color: theme.colors.textMuted }}>
            BY PROCEEDING, YOU AGREE TO INSTITUTIONAL <br /> COMPLIANCE AND SECURITY PROTOCOLS.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
