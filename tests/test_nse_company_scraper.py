"""Regression fixture is the real HTML snippet from
https://afx.kwayisi.org/nse/scom.html (fetched 2026-07-11) — verifies the
parser against afx's actual "Growth & Valuation" table markup, not a
guessed structure."""
from src.connectors.nse_scraper import _parse_company_page

REAL_SCOM_SNIPPET = (
    '<table><thead><tr><th colspan="2">Growth &amp; Valuation</th></tr></thead>'
    '<tbody><tr><td>Earnings Per Share</td><td class="hi">2.3863</td></tr>'
    '<tr><td>Price/Earning Ratio</td><td class="hi">14.69</td></tr>'
    '<tr><td>Dividend Per Share</td><td>2.30</td></tr>'
    '<tr><td>Dividend Yield</td><td>6.56%</td></tr>'
    '<tr><td>Shares Outstanding</td><td>40.1B</td></tr>'
    '<tr><td>Market Capitalization</td><td>1.4T</td></tr></tbody></table>'
)


def test_parses_real_afx_company_page_snippet():
    result = _parse_company_page(REAL_SCOM_SNIPPET, 'scom')
    assert result == {
        'symbol': 'SCOM', 'eps': 2.3863, 'pe_ratio': 14.69,
        'dividend_per_share': 2.30, 'dividend_yield_pct': 6.56,
    }


def test_missing_growth_valuation_table_returns_none():
    assert _parse_company_page('<html><body>no data here</body></html>', 'XXXX') is None


def test_partial_fields_missing_yield_returns_none():
    html = ('<table><thead><tr><th colspan="2">Growth &amp; Valuation</th></tr></thead>'
            '<tbody><tr><td>Earnings Per Share</td><td>1.0</td></tr></tbody></table>')
    assert _parse_company_page(html, 'XXXX') is None
