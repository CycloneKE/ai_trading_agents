"""The Research page's readers for the 29 September 2026 uploads: AIB-AXYS's
Daily Whispers as two pictures (a Global Equity sheet of US-listed stocks and
a Kenyan Equity sheet, most of whose stocks the agent does not trade) and a
Primary Bond Auction Note. Before, the Global sheet was rejected whole (its
names are not NSE stocks), the Kenyan stocks it did not trade flooded the
approval queue on every upload, and the bond note matched no reader. The
bond note text below is written to the note's layout and labels."""
import json
import os
from datetime import date
from types import SimpleNamespace

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import bcrypt
import pytest

import src.utils.paths as paths
from src.agent import bond_auctions as ba
from src.agent import daily_whispers as dw
from src.agent import market_pulse as mp
from src.agent.broker_research_ingest import BrokerResearchIngest, describe
from src.agent.escalation_manager import EscalationManager

ON = date(2026, 9, 29)
no_close = lambda symbol, on: None

GLOBAL = {'report_title': 'The Daily Whispers: Global Equity Market', 'report_date': '2026-09-29', 'rows': [
    {'security_name': 'Boeing Co', 'ticker': 'BA', 'exchange': 'NYSE', 'ytd_pct': -3.4, 'current_price': 184.39,
     'target_price': 273.12, 'upside_pct': 48.1, 'recommendation': 'BUY', 'rationale': ['Deliveries are recovering.']},
    {'security_name': 'Western Digital Corporation', 'ticker': 'WDC', 'exchange': 'NASDAQ', 'ytd_pct': 163.1,
     'current_price': 453.23, 'target_price': 664.92, 'upside_pct': 46.7, 'recommendation': 'BUY', 'rationale': []},
    {'security_name': 'NVIDIA Corporation', 'ticker': 'NVDA', 'exchange': 'NASDAQ', 'ytd_pct': 22.7,
     'current_price': 228.86, 'target_price': 327.70, 'upside_pct': 43.2, 'recommendation': 'BUY', 'rationale': []},
    {'security_name': 'MongoDB', 'ticker': 'MDB', 'exchange': 'NASDAQ', 'ytd_pct': -20.3, 'current_price': 334.68,
     'target_price': 454.15, 'upside_pct': 35.7, 'recommendation': 'BUY', 'rationale': []},
    {'security_name': 'Medtronic PLC', 'ticker': 'MDT', 'exchange': 'NYSE', 'ytd_pct': -6.8, 'current_price': 89.50,
     'target_price': 104.83, 'upside_pct': 17.1, 'recommendation': 'BUY', 'rationale': []},
]}
KENYA = {'report_title': 'The Daily Whispers: Kenyan Equity Market', 'report_date': '2026-09-29', 'rows': [
    {'security_name': 'KCB Group', 'ytd_pct': 41.8, 'current_price': 93.25, 'target_price': 119.15,
     'upside_pct': 27.8, 'recommendation': 'BUY', 'rationale': []},
    {'security_name': 'Diamond Trust Bank', 'ytd_pct': 64.4, 'current_price': 188.25, 'target_price': 224.40,
     'upside_pct': 19.2, 'recommendation': 'BUY', 'rationale': []},
    {'security_name': 'Jubilee Holdings', 'ytd_pct': 15.8, 'current_price': 388.00, 'target_price': 429.00,
     'upside_pct': 10.6, 'recommendation': 'BUY', 'rationale': []},
    {'security_name': 'Williamson Tea Kenya', 'ytd_pct': 6.0, 'current_price': 158.50, 'target_price': 173.67,
     'upside_pct': 9.6, 'recommendation': 'BUY', 'rationale': []},
    {'security_name': 'Kenya Re-insurance Corporation', 'ytd_pct': 46.2, 'current_price': 4.40,
     'target_price': 4.81, 'upside_pct': 9.3, 'recommendation': 'BUY', 'rationale': []},
]}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA_DIR', tmp_path)
    return tmp_path


# ---- the Global Equity sheet

def test_a_global_sheet_is_read_row_by_row_as_us_listed_stocks():
    accepted, rejected = dw.verify(GLOBAL, ON, no_close, intl_price=no_close)
    assert rejected == []
    assert [(r['symbol'], r['exchange'], r['market']) for r in accepted] == [
        ('BA', 'NYSE', 'international'), ('WDC', 'NASDAQ', 'international'), ('NVDA', 'NASDAQ', 'international'),
        ('MDB', 'NASDAQ', 'international'), ('MDT', 'NYSE', 'international')]
    assert accepted[0]['name'] == 'Boeing Co' and accepted[0]['upside_pct'] == 48.1


def test_the_code_and_exchange_are_found_in_the_printed_name_when_not_split_out():
    row = {**GLOBAL['rows'][0], 'security_name': 'Boeing Co (BA) EX: NYSE', 'ticker': None, 'exchange': None}
    [got], rejected = dw.verify({'rows': [row]}, ON, no_close, intl_price=no_close)
    assert rejected == [] and got['symbol'] == 'BA' and got['exchange'] == 'NYSE' and got['name'] == 'Boeing Co'


def test_global_rows_face_the_same_checks_as_any_other():
    rows = [dict(GLOBAL['rows'][0], target_price=2731.2),                              # a digit too many
            dict(GLOBAL['rows'][1], exchange='LSE'),                                    # a market the agent cannot read yet
            dict(GLOBAL['rows'][2], ticker='NVIDIA CORP'),                              # not a stock code
            dict(GLOBAL['rows'][3], ticker=None, exchange=None),                        # nothing to go on
            GLOBAL['rows'][4]]
    accepted, rejected = dw.verify({'rows': rows}, ON, no_close, intl_price=no_close)
    assert [r['symbol'] for r in accepted] == ['MDT']
    reasons = [r['reason'] for r in rejected]
    assert 'numbers disagree' in reasons[0] and 'only US-listed' in reasons[1]
    assert 'not one the agent can use' in reasons[2] and 'no US stock code' in reasons[3]


def test_a_us_price_far_from_the_current_one_is_rejected():
    accepted, rejected = dw.verify({'rows': [GLOBAL['rows'][2]]}, ON, no_close, intl_price=lambda s, on: 120.0)
    assert accepted == [] and 'far from the current price' in rejected[0]['reason']
    accepted, _ = dw.verify({'rows': [GLOBAL['rows'][2]]}, ON, no_close, intl_price=lambda s, on: 230.32)
    assert [r['symbol'] for r in accepted] == ['NVDA'] and accepted[0]['price_checked'] is True


def test_the_kenyan_sheet_still_reads_and_never_looks_at_the_us_check():
    accepted, rejected = dw.verify(KENYA, ON, no_close, intl_price=lambda s, on: pytest.fail('NSE rows use the NSE check'))
    assert rejected == []
    assert [r['symbol'] for r in accepted] == ['KCB', 'DTK', 'JUB', 'WTK', 'KNRE']
    assert all(r['market'] == 'kenyan' for r in accepted)


# ---- what happens to stocks the agent does not trade

CONFIG = {'data_manager': {'symbols': ['NVDA', 'AAPL'], 'nse_symbols': ['KCB', 'SCOM']},
          'research_ingest': {'auto_follow_rules': {'require_existing_symbol': True,
                                                    'allowed_recommendations': ['BUY', 'HOLD', 'ACCUMULATE'],
                                                    'min_upside_pct': 5.0}}}


@pytest.fixture
def em(tmp_path):
    manager = EscalationManager(str(tmp_path / 'esc.db'))
    yield manager
    manager.close()


def _signals(sheet):
    accepted, _ = dw.verify(sheet, ON, no_close, intl_price=no_close)
    return [{'symbol': r['symbol'], 'market': r['market'], 'current_price': r['current_price'],
             'target_price': r['target_price'], 'upside_pct': r['upside_pct'],
             'recommendation': r['recommendation'], 'rationale': r['rationale'], 'risk_factors': [],
             'time_horizon': 'medium_term', 'confidence': 1.0} for r in accepted]


def test_stocks_the_agent_does_not_trade_are_kept_for_reference_not_queued(em):
    ingest = BrokerResearchIngest(None, em, CONFIG)
    upload = em.record_upload('kenya.jpg', 'aib_axys')
    result = ingest._handle_signals(upload, _signals(KENYA), queue_untracked=False)
    assert result['status'] == 'completed' and result['signals_processed'] == 5
    assert result['reference_only'] == ['DTK', 'JUB', 'WTK', 'KNRE']
    assert [e[0] for e in result['escalated']] == []                                    # KCB is traded: auto-followed or queued by the rules
    queued = [e['symbol'] for e in em.get_pending_escalations()]
    assert not set(queued) & {'DTK', 'JUB', 'WTK', 'KNRE'}
    assert em.latest_signal('DTK')['recommendation'] == 'BUY'                            # the rating itself is on record


def test_a_traded_us_stock_from_the_global_sheet_follows_the_normal_rules(em):
    ingest = BrokerResearchIngest(None, em, CONFIG)
    result = ingest._handle_signals(em.record_upload('g.jpg', 'aib_axys'), _signals(GLOBAL), queue_untracked=False)
    assert result['reference_only'] == ['BA', 'WDC', 'MDB', 'MDT']
    assert result['auto_followed'] == ['NVDA']                                           # traded, BUY, 43% upside


def test_a_request_already_waiting_is_not_queued_again(em):
    ingest = BrokerResearchIngest(None, em, CONFIG)
    upload = em.record_upload('n.pdf', 'aib_axys')
    signals = [{'symbol': 'ABSA', 'market': 'kenyan', 'current_price': 10, 'target_price': 11,
                'recommendation': 'BUY', 'rationale': '', 'risk_factors': [], 'confidence': 1.0}]
    first = ingest._handle_signals(upload, signals)                                       # an analyst note still queues
    second = ingest._handle_signals(upload, signals)
    assert [e[0] for e in first['escalated']] == ['ABSA'] and second['already_queued'] == ['ABSA']
    assert [e['symbol'] for e in em.get_pending_escalations()] == ['ABSA']


def test_the_queue_can_be_switched_back_on_by_configuration(em):
    ingest = BrokerResearchIngest(None, em, {**CONFIG, 'research_ingest': {'queue_untracked_ratings': True}})
    assert ingest.queue_untracked_ratings is True
    assert BrokerResearchIngest(None, em, CONFIG).queue_untracked_ratings is False


def test_the_upload_summary_says_what_was_kept_and_why(em):
    ingest = BrokerResearchIngest(None, em, CONFIG)
    result = ingest._handle_signals(em.record_upload('k.jpg', 'aib_axys'), _signals(KENYA), queue_untracked=False)
    text = ' '.join(describe({'document_type': 'recommendation_sheet', **result, 'accepted': [], 'rejected': [],
                              'as_of': '2026-09-29'}))
    assert 'Kept for reference' in text and 'DTK, JUB, WTK, KNRE' in text and 'press Follow' in text


def test_a_sheet_picture_goes_through_read_check_keep_and_list(em, data_dir, monkeypatch, tmp_path):
    """The whole path for a Global sheet with a stand-in for the AI's reading."""
    llm = SimpleNamespace(enabled=True, read_image_json=lambda *a, **k: GLOBAL)
    monkeypatch.setattr(dw, 'last_intl_price', lambda s, on: None)
    picture = tmp_path / 'global.jpg'
    picture.write_bytes(b'\xff\xd8\xff')
    result = BrokerResearchIngest(llm, em, CONFIG).process_image(str(picture))
    assert result['status'] == 'completed' and result['document_type'] == 'recommendation_sheet'
    assert result['reference_only'] == ['BA', 'WDC', 'MDB', 'MDT'] and result['auto_followed'] == ['NVDA']
    assert {r['symbol'] for r in dw.recent_ratings(today=ON)} == {'BA', 'WDC', 'NVDA', 'MDB', 'MDT'}
    assert dw.latest('NVDA', today=ON)['market'] == 'international'                      # the AI's trade review sees it


# ---- the bond auction note

NOTE = """Page | 2
Summary
The Exchequer is seeking to raise KES 50.00 Bn through the reopening of two treasury bonds. Domestic borrowing
is well ahead of the borrowing target, with a 177.5% performance rate. The Weighted Average Rate of Accepted Bids was 13.611%  for
the former.
AXYS September 2026 Primary Bond Auction Note III
 Table 1: Key Auction Highlights
FXD3/2019/015 & FXD1/2019/020
Issuer: Republic of Kenya
Total Amount: KES 50.0 billion
Purpose: For budgetary support
Tenor: FXD3/2019/015 – 7.8 Yrs)– Re-opened
FXD1/2019/020 - (12.5 Yrs)– Re-opened
Coupon Rate: FXD3/2019/015 – 12.3400%
FXD1/2019/020 – 12.8730%
Price Quote: Discounted/Premium/Par
Period of sale: 24th Sep 2026 to 30th Sep 2026
Minimum
Amount: KES 50,000.00
Taxation: 10.0%
Maturity Dates: FXD3/2019/015 – 10th -Jul - 2034
FXD1/2019/020 – 21st -Mar -2039
Non-competitive
bids per CSD A/C: Maximum KES 50 million per CDS A/c
AXYS
Competitive
Bidding Range
Recommendation:
FXD3/2019/015 – 12.59-12.79%
FXD1/2019/020 – 13.41-13.61%
Source: CBK, AXYS Research
Page | 3
Kenya's headline inflation inched higher to 6.6% y/y in August 2026 from 6.5% in July. In August, the Kenya Shilling
Overnight Interbank Average (KESONIA) remained relatively stable at 8.75%, largely unchanged from July.
"""


def test_the_bond_note_is_recognised_and_its_figures_read_exactly():
    texts = [NOTE]
    assert ba.is_bond_auction(texts) and not mp.is_market_pulse(texts)
    note = ba.parse(texts)
    assert note['warnings'] == [] and ba.usable(note)
    assert (note['issuer'], note['total_kes_bn'], note['sale_from'], note['sale_to']) == (
        'Republic of Kenya', 50.0, '2026-09-24', '2026-09-30')
    assert (note['min_bid_kes'], note['tax_pct'], note['purpose']) == (50000.0, 10.0, 'For budgetary support')
    a, b = note['papers']
    assert (a['paper'], a['tenor_years'], a['reopened'], a['coupon_pct'], a['maturity'], a['bid_low_pct'], a['bid_high_pct']) == (
        'FXD3/2019/015', 7.8, True, 12.34, '2034-07-10', 12.59, 12.79)
    assert (b['paper'], b['tenor_years'], b['coupon_pct'], b['maturity'], b['bid_low_pct'], b['bid_high_pct']) == (
        'FXD1/2019/020', 12.5, 12.873, '2039-03-21', 13.41, 13.61)
    assert note['market'] == {'inflation_pct': 6.6, 'inflation_month': 'August 2026', 'kesonia_pct': 8.75,
                              'borrowing_vs_target_pct': 177.5, 'last_accepted_rate_pct': 13.611}


def test_what_the_note_does_not_say_stays_empty_and_what_cannot_be_right_is_dropped():
    note = ba.parse([NOTE.replace('Taxation: 10.0%', '').replace('13.41-13.61%', '13.61-13.41%')
                     .replace('KES 50.0 billion', 'KES 5000.0 billion')])
    assert note['tax_pct'] is None and note['total_kes_bn'] is None
    assert note['papers'][1]['bid_low_pct'] is None and note['papers'][1]['bid_high_pct'] is None
    assert any('upside down' in w for w in note['warnings']) and any('implausible' in w for w in note['warnings'])
    assert ba.parse(['a page about something else'])['warnings'] == ['the key auction highlights table was not found']


def test_the_sale_period_says_whether_the_auction_is_open():
    note = ba.parse([NOTE])
    assert [ba.status(note, date(2026, 9, d)) for d in (23, 24, 30)] == ['upcoming', 'open', 'open']
    assert ba.status(note, date(2026, 10, 1)) == 'closed' and ba.status({}, ON) == 'unknown'


def test_the_note_is_kept_once_and_listed(data_dir):
    first = ba.ingest('/x/AXYS_Note_III.pdf', [NOTE], ON)
    assert first['stored'] and first['source_file'] == 'AXYS_Note_III.pdf'
    ba.ingest('/x/again.pdf', [NOTE], ON)                                                  # the same note again replaces it
    [listed] = ba.recent(today=ON)
    assert listed['status'] == 'open' and listed['source_file'] == 'again.pdf'
    assert not ba.ingest('/x/junk.pdf', ['Table 1: Key Auction Highlights\nsource: nothing'], ON)['stored']


def test_a_bond_pdf_is_read_not_sent_to_the_stock_rating_reader(em, data_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(mp, 'page_texts', lambda path: [NOTE])
    llm = SimpleNamespace(enabled=True)                                                     # asked for nothing
    result = BrokerResearchIngest(llm, em, CONFIG).process_pdf(str(tmp_path / 'AXYS_Bond_Note.pdf'))
    assert result['status'] == 'completed' and result['document_type'] == 'bond_auction'
    assert result['signals_processed'] == 0 and em.get_pending_escalations() == []
    assert em.get_upload_history()[0]['status'] == 'completed'
    text = ' '.join(result['actions'])
    assert 'KES 50 billion' in text and 'FXD3/2019/015' in text and '12.59 to 12.79%' in text
    assert 'nothing was queued' in text


def test_a_bond_note_whose_figures_cannot_be_read_says_so(em, data_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(mp, 'page_texts', lambda path: ['Primary Bond Auction Note\nKey Auction Highlights\nTable 1: Key Auction Highlights\nnothing readable'])
    result = BrokerResearchIngest(None, em, CONFIG).process_pdf(str(tmp_path / 'x.pdf'))
    assert result['status'] == 'failed' and 'could not be read' in result['error']


# ---- the Research page's data

def _client(tmp_path, monkeypatch, em):
    import src.api.auth as auth
    from src.api.api_server import TradingAPI
    from src.api.auth import create_token
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {
        'password': bcrypt.hashpw(b'unused', bcrypt.gensalt()).decode(), 'role': 'operator'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    agent = SimpleNamespace(components={'escalation_manager': em, 'research_ingest': BrokerResearchIngest(None, em, CONFIG),
                                        'data_manager': None}, config={'data_manager': {'symbols': [], 'nse_symbols': []}})
    api = TradingAPI(agent, {})
    return api.app.test_client(), {'Authorization': f"Bearer {create_token('tester', 'operator')}"}


def test_the_page_lists_ratings_with_whether_each_is_traded_and_lets_the_operator_follow_one(tmp_path, monkeypatch, em, data_dir):
    accepted, _ = dw.verify(GLOBAL, date.today(), no_close, intl_price=no_close)
    dw.remember(accepted, 'g.jpg')
    ingest = BrokerResearchIngest(None, em, CONFIG)
    ingest._handle_signals(em.record_upload('g.jpg', 'aib_axys'), _signals(GLOBAL), queue_untracked=False)
    client, headers = _client(tmp_path, monkeypatch, em)

    rows = {r['symbol']: r for r in client.get('/api/research/ratings', headers=headers).get_json()['ratings']}
    assert set(rows) == {'BA', 'WDC', 'NVDA', 'MDB', 'MDT'}
    assert rows['NVDA']['tracked'] is True and rows['BA']['tracked'] is False and rows['BA']['follow_pending'] is False

    first = client.post('/api/research/ratings/BA/follow', headers=headers).get_json()
    again = client.post('/api/research/ratings/BA/follow', headers=headers).get_json()
    assert first == {'success': True, 'already_queued': False} and again['already_queued'] is True
    [queued] = [e for e in em.get_pending_escalations() if e['symbol'] == 'BA']
    assert queued['recommendation'] == 'BUY' and queued['market'] == 'international' and queued['target_price'] == 273.12
    rows = {r['symbol']: r for r in client.get('/api/research/ratings', headers=headers).get_json()['ratings']}
    assert rows['BA']['follow_pending'] is True
    assert client.post('/api/research/ratings/ZZZZ/follow', headers=headers).status_code == 404


def test_the_page_lists_bond_auctions(tmp_path, monkeypatch, em, data_dir):
    ba.ingest('/x/note.pdf', [NOTE], date.today())
    client, headers = _client(tmp_path, monkeypatch, em)
    [auction] = client.get('/api/research/bond-auctions', headers=headers).get_json()['auctions']
    assert auction['papers'][0]['paper'] == 'FXD3/2019/015' and auction['status'] in ('open', 'closed', 'upcoming')
    assert client.get('/api/research/bond-auctions').status_code in (401, 403)
