"""Regression: NSE table parser must handle implicit (unclosed) HTML tags.

afx.kwayisi.org serves minified HTML5 without </td>/</tr>; the parser
previously extracted 0 rows from it, which silently froze all NSE data
at the March synthetic seed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.connectors.nse_scraper import _NSETableParser


MINIFIED = ('<table><tr><td><a href=x>SCOM</a><td><a href=x>Safaricom Plc</a>'
            '<td>2,785,507<td>34.20<td class=hi>+0.15'
            '<tr><td>EQTY<td>Equity Group<td>1,000<td>50.25<td>-0.50</table>')


def test_implicit_close_html_parses():
    p = _NSETableParser()
    p.feed(MINIFIED)
    assert p.rows == [
        ['SCOM', 'Safaricom Plc', '2,785,507', '34.20', '+0.15'],
        ['EQTY', 'Equity Group', '1,000', '50.25', '-0.50'],
    ]


def test_explicit_close_html_still_parses():
    p = _NSETableParser()
    p.feed('<table><tr><td>X</td><td>9</td></tr><tr><td>Y</td><td>8</td></tr></table>')
    assert p.rows == [['X', '9'], ['Y', '8']]


def test_dangling_row_flushed_at_table_end():
    p = _NSETableParser()
    p.feed('<table><tr><td>only<td>row</table>')
    assert p.rows == [['only', 'row']]
