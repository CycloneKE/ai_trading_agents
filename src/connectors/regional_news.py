"""
East Africa / Kenya regional news connector.

No API key required. Business Daily Africa, The East African, Nation, and
Capital Business don't offer public APIs, so rather than building a scraper
per outlet this pulls from Google News RSS, which already indexes them —
free and keyless.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

# Broad enough to surface NSE-listed companies, the Nairobi bourse, and
# macro stories (CBK rate moves, KES fx) without drowning in noise.
DEFAULT_QUERY = (
    '("Nairobi Securities Exchange" OR "NSE Kenya" OR "Central Bank of Kenya" '
    'OR "Kenya shilling" OR "Kenya economy") when:2d'
)

_TAG_RE = re.compile(r'<[^>]+>')


def _strip_html(text: str) -> str:
    return _TAG_RE.sub('', text or '').strip()


def get_east_africa_news(query: str = DEFAULT_QUERY, limit: int = 10,
                          timeout: int = 8) -> List[Dict[str, Any]]:
    """Fetch recent East Africa business/markets news via Google News RSS."""
    try:
        params = {'q': query, 'hl': 'en-KE', 'gl': 'KE', 'ceid': 'KE:en'}
        resp = requests.get(GOOGLE_NEWS_RSS, params=params, timeout=timeout,
                             headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        items = []
        for item in root.findall('.//item')[:limit]:
            title = (item.findtext('title') or '').strip()
            link = (item.findtext('link') or '').strip()
            pub_date = item.findtext('pubDate')
            source_el = item.find('source')
            source = source_el.text if source_el is not None else 'Google News'

            try:
                published = parsedate_to_datetime(pub_date) if pub_date else datetime.utcnow()
            except (TypeError, ValueError):
                published = datetime.utcnow()

            items.append({
                'title': title,
                'summary': _strip_html(item.findtext('description')),
                'source': source,
                'url': link,
                'time': published.isoformat(),
                'region': 'east_africa',
            })
        return items
    except Exception as e:
        logger.warning(f"Failed to fetch East Africa news: {e}")
        return []
