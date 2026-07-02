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
        <meta name="description" content="Premium AI-driven trading terminal with real-time market intelligence and risk management." />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet" />
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
          background-color: #020617;
          color: #f8fafc;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
        * {
          box-sizing: border-box;
        }
        ::-webkit-scrollbar {
          width: 8px;
        }
        ::-webkit-scrollbar-track {
          background: #020617;
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