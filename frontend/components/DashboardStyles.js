/**
 * Premium Dashboard Design System
 * Focuses on Glassmorphism, Sophisticated Dark Tones, and High-Contrast Accents.
 */

export const theme = {
  colors: {
    bg: '#020617',                   // Deepest Slate
    bgSecondary: '#0f172a',          // Slate-900 
    surface: 'rgba(30, 41, 59, 0.5)', // Slate-800 with transparency
    border: 'rgba(51, 65, 85, 0.5)',  // Slate-700 with transparency
    
    // Semantic Accents
    primary: '#10b981',   // Emerald (Growth/Success)
    secondary: '#0ea5e9', // Sky (Info/System)
    accent: '#8b5cf6',    // Violet (AI/Intelligence)
    warning: '#f59e0b',   // Amber
    danger: '#f43f5e',    // Rose
    
    text: '#f8fafc',          // Slate-50
    textSecondary: '#94a3b8', // Slate-400
    textMuted: '#64748b',     // Slate-500
  },
  
  glass: {
    backdropFilter: 'blur(12px) saturate(180%)',
    WebkitBackdropFilter: 'blur(12px) saturate(180%)',
    backgroundColor: 'rgba(17, 25, 40, 0.75)',
    border: '1px solid rgba(255, 255, 255, 0.125)',
    borderRadius: '16px',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
  },
  
  animations: {
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    hover: 'translateY(-2px) scale(1.01)',
  },
  
  shadows: {
    glow: '0 0 15px rgba(16, 185, 129, 0.2)',
    inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
  }
};

export const glassCard = {
  ...theme.glass,
  padding: '24px',
  transition: theme.animations.transition,
};

export const interactiveGlassCard = {
  ...glassCard,
  cursor: 'pointer',
  ':hover': {
    backgroundColor: 'rgba(30, 41, 59, 0.8)',
    transform: theme.animations.hover,
    borderColor: theme.colors.primary,
  }
};
