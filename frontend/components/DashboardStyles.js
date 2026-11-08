/**
 * Premium Dashboard Design System
 * Focuses on Glassmorphism, Sophisticated Slate Tones, and High-Contrast Accents.
 */

export const theme = {
  colors: {
    bg: '#030712',                     // Deepest Obsidian Slate
    bgSecondary: '#0b1120',            // Slate-950
    surface: 'rgba(15, 23, 42, 0.65)',  // Slate-900 with glass transparency
    surfaceElevated: 'rgba(30, 41, 59, 0.8)', // Slate-800 elevated glass
    border: 'rgba(255, 255, 255, 0.08)', // Crisp translucent border
    borderHover: 'rgba(255, 255, 255, 0.18)',
    borderActive: 'rgba(16, 185, 129, 0.5)',
    
    // Semantic Accents
    primary: '#10b981',   // Emerald (Growth / Profit / Healthy)
    primaryLight: '#34d399',
    secondary: '#0ea5e9', // Sky (System / Telemetry)
    accent: '#8b5cf6',    // Violet (AI Models / Intelligence)
    warning: '#f59e0b',   // Amber (Pre-market / Warnings)
    danger: '#f43f5e',    // Rose (Loss / Risk / Halted)
    
    text: '#f8fafc',          // Slate-50
    textSecondary: '#94a3b8', // Slate-400
    textMuted: '#64748b',     // Slate-500
  },
  
  radius: {
    xs: '4px',
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '20px',
    full: '9999px',
  },

  glass: {
    backdropFilter: 'blur(16px) saturate(180%)',
    WebkitBackdropFilter: 'blur(16px) saturate(180%)',
    backgroundColor: 'rgba(11, 17, 32, 0.75)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '16px',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 0 rgba(255, 255, 255, 0.05)',
  },
  
  animations: {
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    hover: 'translateY(-2px)',
  },
  
  shadows: {
    glowEmerald: '0 0 20px rgba(16, 185, 129, 0.25)',
    glowCyan: '0 0 20px rgba(14, 165, 233, 0.25)',
    glowRose: '0 0 20px rgba(244, 63, 94, 0.25)',
    inner: 'inset 0 1px 2px 0 rgba(0, 0, 0, 0.25)',
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
    backgroundColor: 'rgba(30, 41, 59, 0.85)',
    transform: theme.animations.hover,
    borderColor: theme.colors.borderHover,
  }
};
