"""
Sector Specialist Agent:
Maintains progressive knowledge profiles and outlooks for individual market sectors
by analyzing news feeds and persisting summaries into SQLite.
"""
import os
import json
import sqlite3
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

# Default mapping of assets to sectors
_DEFAULT_SECTOR_MAP = {
    # US Stocks
    "AAPL": "technology",
    "GOOGL": "technology",
    "MSFT": "technology",
    "NVDA": "technology",
    "QQQ": "technology",
    "TSLA": "automotive",
    "XLE": "energy",
    "XLF": "finance",
    "DIA": "finance",
    "SPY": "finance",
    "XLV": "healthcare",
    
    # Kenya NSE Stocks
    "SCOM": "telecommunications",
    "EQTY": "finance",
    "KCB": "finance",
    "COOP": "finance",
    "SCBK": "finance",
    "SBIC": "finance",
    "ABSA": "finance",
    "NCBA": "finance",
    "BAT": "agriculture_consumer",
    "EABL": "agriculture_consumer",
    "KEGN": "energy",
    "KNRE": "energy",
    "BAMB": "materials_construction",
    "TOTL": "energy",
    "CTUM": "investments",
    "NMG": "media",
    "BRIT": "finance",
    "CIC": "finance"
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sector_knowledge_profiles (
    sector          TEXT PRIMARY KEY,
    last_updated    TEXT NOT NULL,
    knowledge_json  TEXT NOT NULL
);
"""

_SPECIALIST_PROMPT = """
You are an Elite Sector Specialist Analyst representing the trading system's Swarm.
Your responsibility is to analyze all news, events, and metrics affecting your assigned sector: '{sector}'.

You will be given:
1. The PREVIOUS persistent knowledge profile for this sector (what you learned in past iterations).
2. The LATEST news headlines and sentiment snippets collected for this sector during this cycle.

Your task is to merge the new insights into your existing profile to form a progressively smarter, updated knowledge base. 
Analyze the information carefully. Then, provide an updated outlook score, a revised profile description, key risks, and near-term catalysts.

Output your response STRICTLY as a valid JSON object matching this schema:
{{
  "outlook_score": float_value_between_neg_1_and_pos_1,
  "updated_profile_text": "A concise paragraph summarizing the current state of the sector, cumulative triggers, earnings results, and general trend. Overwrite or append to the previous profile as appropriate.",
  "risk_factors": ["risk factor 1", "risk factor 2"],
  "catalysts": ["catalyst 1", "catalyst 2"]
}}

Ensure the output is strictly valid raw JSON. Do not wrap in markdown blocks, do not add trailing text or comments.
"""


class SectorSpecialistManager:
    def __init__(self, llm_orchestrator, db_path: str = str(DATA_DIR / 'escalations.db')):
        self.llm = llm_orchestrator
        self.db_path = db_path
        self._lock = threading.Lock()
        
        # Initialize SQLite table inside escalations.db
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        
        # Load sector maps from config or fall back to defaults
        self.sector_map = dict(_DEFAULT_SECTOR_MAP)
        if self.llm and hasattr(self.llm, "config"):
            custom_map = self.llm.config.get("swarm", {}).get("sector_map", {})
            if custom_map:
                self.sector_map.update({k.upper(): v.lower() for k, v in custom_map.items()})
                logger.info(f"Loaded {len(custom_map)} custom sector mappings from config.")
        
        self._seed_default_profiles()
        logger.info("SectorSpecialistManager initialized.")

    def _seed_default_profiles(self):
        """Seed default realistic profiles for sectors if the table is empty."""
        default_profiles = {
            "technology": {
                "outlook_score": 0.65,
                "updated_profile_text": "The technology sector remains highly bullish, driven by strong cloud software earnings, accelerating AI infrastructure demand, and robust hardware cycles. Margins are expanding due to efficiency improvements, though regulatory scrutiny on big tech remains a minor tail risk.",
                "risk_factors": ["High valuation multiples", "Antitrust and regulatory scrutiny", "Supply chain constraints in semiconductor packaging"],
                "catalysts": ["AI infrastructure capital expenditure expansion", "Strong software-as-a-service (SaaS) adoption", "Upcoming semiconductor earnings beats"]
            },
            "energy": {
                "outlook_score": -0.15,
                "updated_profile_text": "The energy sector faces headwinds due to localized demand slowdowns, high global production rates, and growing inventory levels. Strategic transitions to renewable sources are slowly diluting fossil fuel dominance, keeping price gains capped.",
                "risk_factors": ["Overproduction by non-OPEC members", "Global demand contraction fears", "Regulatory policies targeting carbon emissions"],
                "catalysts": ["Geopolitical production risks", "Seasonal winter heating demand spikes", "Strategic reserve replenishments"]
            },
            "finance": {
                "outlook_score": 0.35,
                "updated_profile_text": "The financial sector exhibits steady performance. A stable interest rate environment maintains high net interest margins for major banks. Credit quality remains resilient despite slight increases in provisions for credit losses. Investment banking fees are showing signs of cyclical recovery.",
                "risk_factors": ["Increasing regulatory capital requirements", "Commercial real estate loan exposure", "Inverted yield curve duration risk"],
                "catalysts": ["Federal Reserve monetary easing clarity", "M&A advisory activity rebound", "Consumer credit resilience updates"]
            },
            "healthcare": {
                "outlook_score": 0.20,
                "updated_profile_text": "Healthcare shows defensive growth characteristics. Strong innovation in therapeutic pipelines (e.g. GLP-1 weight loss drugs) offsets patent expirations for older blockbusters. Medtech equipment demand is recovering as surgical volumes normalize.",
                "risk_factors": ["Drug price negotiation provisions", "Patent expiration cliffs", "High R&D cost amortization"],
                "catalysts": ["FDA product approvals", "Strategic biotech acquisitions", "Favorable clinical trial results"]
            },
            "telecommunications": {
                "outlook_score": 0.45,
                "updated_profile_text": "Telecommunications is positioned strongly, especially in emerging markets. Growing data consumption, expansion of 5G network coverage, and mobile money ecosystems (like M-Pesa on Safaricom) provide stable recurring revenues and strong free cash flow generation.",
                "risk_factors": ["High capital expenditure requirements", "Pricing competition in mobile data packages", "Foreign exchange devaluation impacts on dollar debt"],
                "catalysts": ["Mobile financial services transaction growth", "Favorable regulatory spectrum allocations", "Fiber infrastructure rollouts"]
            },
            "utilities": {
                "outlook_score": 0.10,
                "updated_profile_text": "Utilities are showing stable income profiles. Grid modernization projects support equity returns. Growth in power demand from data centers provides a new volume-driven tailwind.",
                "risk_factors": ["High debt load sensitivity to rates", "Severe weather events", "Grid modernization capital expenditure pressures"],
                "catalysts": ["Lower interest rates decreasing financing costs", "Data center grid connection agreements", "Renewable generation integration"]
            },
            "consumer_staples": {
                "outlook_score": 0.05,
                "updated_profile_text": "Consumer staples remain stable, with companies showing pricing power to defend margins. Input cost inflation is easing, though consumers are becoming more price-sensitive and shifting toward private-label alternatives.",
                "risk_factors": ["Brand volume erosion to private labels", "Input cost volatility", "Retailer inventory destocking"],
                "catalysts": ["Slowing inflation supporting real wages", "Defensive rotation during market uncertainty", "Margin expansion from cost-cutting initiatives"]
            },
            "real_estate": {
                "outlook_score": -0.25,
                "updated_profile_text": "Real estate remains pressured, particularly in commercial office space, due to structurally higher remote work adoption and elevated refinancing costs for maturing commercial debt. Residential markets remain supported by low inventory.",
                "risk_factors": ["Commercial mortgage maturities refinancing risk", "Low office occupancy rates", "High borrowing costs for developers"],
                "catalysts": ["Interest rate cuts stabilizing cap rates", "Residential inventory expansion", "Logistics/industrial property demand growth"]
            },
            "materials": {
                "outlook_score": 0.0,
                "updated_profile_text": "Materials sector performance is neutral. Demand for industrial metals is balanced by slower construction activity, while chemical margins are recovering as raw material energy inputs stabilize.",
                "risk_factors": ["Global industrial manufacturing slowdown", "Fluctuating commodity prices", "Energy input cost spikes"],
                "catalysts": ["Infrastructure spending programs", "Supply constraints in copper and lithium", "Inventory restocking cycle launch"]
            }
        }
        
        with self._lock:
            for sector, profile in default_profiles.items():
                cur = self._conn.execute("SELECT 1 FROM sector_knowledge_profiles WHERE sector = ?", (sector,))
                if not cur.fetchone():
                    self._conn.execute(
                        "INSERT INTO sector_knowledge_profiles (sector, last_updated, knowledge_json) VALUES (?, ?, ?)",
                        (sector, datetime.utcnow().isoformat(), json.dumps(profile))
                    )
            self._conn.commit()

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Returns the assigned sector for a symbol. Defaults to 'general'."""
        return self.sector_map.get(symbol.upper(), "general")

    def load_sector_profile(self, sector: str) -> Dict[str, Any]:
        """Loads the persisted profile from SQLite database."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT knowledge_json FROM sector_knowledge_profiles WHERE sector = ?",
                (sector.lower(),)
            )
            row = cur.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except Exception as e:
                    logger.error(f"Failed to parse sector profile JSON for {sector}: {e}")
            
            # Default profile if not found
            return {
                "outlook_score": 0.0,
                "updated_profile_text": f"Initial profiling for the {sector} sector.",
                "risk_factors": [],
                "catalysts": []
            }

    def save_sector_profile(self, sector: str, profile: Dict[str, Any]) -> None:
        """Saves/overwrites the sector knowledge profile to SQLite."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sector_knowledge_profiles (sector, last_updated, knowledge_json) "
                "VALUES (?, ?, ?)",
                (
                    sector.lower(),
                    datetime.utcnow().isoformat(),
                    json.dumps(profile)
                )
            )
            self._conn.commit()

    def run_sector_analysis(self, sector: str, news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs sector analysis using the swarm's configured sector specialist model.
        Fuses new headlines with the previous persistent knowledge.
        """
        previous_profile = self.load_sector_profile(sector)
        
        # Build prompt variables
        news_summary = ""
        if news_list:
            news_summary = "\n".join([f"- [{n.get('time', 'recent')}] {n.get('title', '')} (Sentiment: {n.get('sentiment', 'N/A')})" for n in news_list[:15]])
        else:
            news_summary = "No new headlines collected in this cycle."
            
        system_prompt = _SPECIALIST_PROMPT.format(sector=sector)
        user_prompt = (
            f"--- PREVIOUS SECTOR PROFILE ---\n"
            f"{json.dumps(previous_profile, indent=2)}\n\n"
            f"--- LATEST NEWS HEADLINES ---\n"
            f"{news_summary}\n\n"
            f"Analyze and output the updated sector profile JSON now."
        )
        
        updated_profile = previous_profile
        if self.llm and getattr(self.llm, "enabled", False):
            try:
                # Direct Claude 3.5 Sonnet call
                model_to_use = self.llm.config.get("swarm", {}).get("agents", {}).get("sector_specialist", "anthropic/claude-3.5-sonnet")
                
                # Propose JSON using the dynamic model
                logger.info(f"Running Sector Specialist analysis for '{sector}' using {model_to_use}...")
                proposal = self.llm.propose_json(system_prompt, user_prompt, model_override=model_to_use)
                
                if proposal and isinstance(proposal, dict) and "outlook_score" in proposal:
                    updated_profile = proposal
                    # Save updated state to SQLite
                    self.save_sector_profile(sector, updated_profile)
                    logger.info(f"Sector Specialist: Updated profile saved for {sector}. Outlook score: {updated_profile.get('outlook_score')}")
            except Exception as e:
                logger.error(f"Sector Specialist agent failed for {sector}: {e}")
                
        return updated_profile

    def close(self):
        """Close connection."""
        with self._lock:
            self._conn.close()
