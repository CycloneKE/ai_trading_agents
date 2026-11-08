import { useState, useEffect } from 'react';
import Head from 'next/head';
import AdvancedDashboard from '../components/AdvancedDashboard';
import Login from '../components/Login';

export default function App() {
  const [mounted, setMounted] = useState(false);
  const [authToken, setAuthToken] = useState(null);

  useEffect(() => {
    setMounted(true);
    const token = localStorage.getItem('trading_token');
    if (token) setAuthToken(token);
  }, []);

  const handleLoginSuccess = (token) => {
    setAuthToken(token);
  };

  const handleLogout = () => {
    localStorage.removeItem('trading_token');
    localStorage.removeItem('trading_user');
    setAuthToken(null);
  };

  if (!mounted) return null;

  return (
    <>
      <Head>
        <title>AEGIS TRADER AI | Secure Terminal</title>
        {/* Lay the page out at the phone's own width, using the full screen
            on phones with rounded corners or a notch. */}
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta name="theme-color" content="#030712" />
        <meta name="description" content="Institutional AI-driven trading terminal with multi-region intelligence, real-time risk management, and automated order routing." />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
      </Head>
      {!authToken ? (
        <Login onLoginSuccess={handleLoginSuccess} />
      ) : (
        <AdvancedDashboard onLogout={handleLogout} />
      )}
      <style jsx global>{`
        body {
          margin: 0;
          padding: 0;
          background-color: #030712;
          color: #f8fafc;
          font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          background-image: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.04) 0%, transparent 45%),
                            radial-gradient(circle at 100% 100%, rgba(14, 165, 233, 0.03) 0%, transparent 45%);
          background-attachment: fixed;
        }
        * {
          box-sizing: border-box;
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
        ::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        ::-webkit-scrollbar-track {
          background: #030712;
        }
        ::-webkit-scrollbar-thumb {
          background: #1e293b;
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: #334155;
        }
      `}</style>
    </>
  );
}
