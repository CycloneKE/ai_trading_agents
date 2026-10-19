"""The research library (src/agent/research_notes.py): documents with no
ratings are kept, every figure is checked against the document's own words
before it is trusted, and only live, relevant ones reach the AI's trade
review. Also the rating reader's fix: an AI that wraps its list under any key
is no longer read as "no ratings"."""
import json
import os
from datetime import date, timedelta
from types import SimpleNamespace

os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-dashboard-honesty-tests')

import bcrypt
import pytest

import src.api.auth as auth
import src.utils.paths as paths
from src.agent import research_notes as rn
from src.agent.broker_research_ingest import BrokerResearchIngest, describe, has_rating_words, items_from
from src.agent.escalation_manager import EscalationManager
from src.api.auth import create_token

TODAY = date(2026, 10, 18)
CONFIG = {'data_manager': {'symbols': ['AAPL'], 'nse_symbols': ['SCOM', 'KCB']}}

TEXT = """Acme Telecom Ltd (ACME) Initial Public Offering Briefing
Date: 14 October 2026
The offer opens on 20 October 2026 and closes on 3 November 2026.
The offer price is KES 20.00 per share for 1,000,000,000 shares.
Listing is expected on 17 November 2026 on the Nairobi Securities Exchange.
Net profit grew 18 percent to KES 7.4 billion in the last financial year.
Key risks include regulatory tariff changes and foreign currency exposure."""
EXTRACTED = {'text': TEXT, 'pages': [{'combined': TEXT, 'text': TEXT}], 'metadata': {'pages_count': 1}}

GOOD_DIGEST = {
    'document_type': 'ipo_briefing', 'title': 'Acme Telecom IPO Briefing', 'as_of': '2026-10-14',
    'summary': 'Acme Telecom is selling shares to the public at KES 20.00.',
    'symbols': ['ACME', 'SCOM', 'FAKE'], 'markets': ['kenya', 'mars'],
    'facts': [
        {'label': 'Offer price', 'value': 'KES 20.00 per share', 'quote': 'The offer price is KES 20.00 per share'},
        {'label': 'Shares offered', 'value': '1,000,000,000', 'quote': 'for 1,000,000,000 shares'},
        {'label': 'Profit growth', 'value': '18%', 'quote': 'Net profit grew 18 percent'},
        {'label': 'Dividend yield', 'value': '9%', 'quote': 'a dividend yield of 9 percent'},      # invented
        {'label': 'Offer price', 'value': 'KES 25.00', 'quote': 'The offer price is KES 20.00 per share'},  # wrong number
        {'label': 'Profit', 'value': 'KES 7.4 billion'},                                         # no quote
    ],
    'dates': [
        {'label': 'Offer closes', 'date': '2026-11-03', 'quote': 'closes on 3 November 2026'},
        {'label': 'Listing', 'date': '2026-11-17', 'quote': 'Listing is expected on 17 November 2026'},
        {'label': 'Imaginary', 'date': '2026-12-01', 'quote': 'something never said in the text'},
        {'label': 'Bad', 'date': 'soon', 'quote': 'Listing is expected on 17 November 2026'},
    ],
    'risks': ['regulatory tariff changes', 'foreign currency exposure']}


class FakeAI:
    enabled = True

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def propose_json(self, system, user, model_override=None):
        self.calls.append(user)
        return self.reply


@pytest.fixture(autouse=True)
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA_DIR', tmp_path)
    return tmp_path


# ------------------------------------------------------------------- checks

def test_a_figure_is_kept_only_if_its_quote_is_in_the_document_and_its_number_is_in_the_quote():
    text = rn._norm(TEXT)
    ok, why = rn.check_fact(GOOD_DIGEST['facts'][0], text)
    assert ok['value'] == 'KES 20.00 per share' and why is None
    assert rn.check_fact(GOOD_DIGEST['facts'][3], text)[1].endswith('the quote is not in the document')
    assert 'is not in the quoted words' in rn.check_fact(GOOD_DIGEST['facts'][4], text)[1]
    assert 'no quote' in rn.check_fact(GOOD_DIGEST['facts'][5], text)[1]


def test_the_same_number_written_differently_is_the_same_number():
    text = rn._norm("The offer is 1,250.00 shillings and growth was 20.0 percent")
    assert rn.check_fact({'label': 'Offer', 'value': '1250', 'quote': 'The offer is 1,250.00 shillings'}, text)[0]
    assert rn.check_fact({'label': 'Growth', 'value': '20%', 'quote': 'growth was 20.0 percent'}, text)[0]
    assert rn.check_fact({'label': 'Growth', 'value': '21%', 'quote': 'growth was 20.0 percent'}, text)[0] is None


def test_a_date_needs_a_real_date_and_words_from_the_document():
    text = rn._norm(TEXT)
    assert rn.check_date(GOOD_DIGEST['dates'][0], text)[0]['date'] == '2026-11-03'
    assert rn.check_date(GOOD_DIGEST['dates'][2], text)[0] is None
    assert rn.check_date(GOOD_DIGEST['dates'][3], text)[0] is None
    assert rn.check_date({'label': 'x', 'date': '1850-01-01', 'quote': 'Listing is expected on 17 November 2026'}, text)[0] is None


def test_symbols_must_be_printed_in_the_document_or_be_ones_the_agent_trades():
    assert rn.clean_symbols(['ACME', 'SCOM', 'FAKE', 'acme'], TEXT, ['SCOM']) == ['ACME', 'SCOM']


def test_validate_drops_what_fails_and_says_why():
    out = rn.validate(GOOD_DIGEST, TEXT, ['SCOM'], TODAY)
    assert [f['label'] for f in out['facts']] == ['Offer price', 'Shares offered', 'Profit growth']
    assert [d['label'] for d in out['dates']] == ['Offer closes', 'Listing']
    assert len(out['rejected']) == 5 and out['markets'] == ['kenya']        # 'mars' is not a market
    assert out['document_type'] == 'ipo_briefing' and out['date_read'] and out['as_of'] == '2026-10-14'


def test_an_unreadable_or_future_date_falls_back_to_today_and_says_so():
    for bad in (None, 'sometime', '2031-01-01'):
        out = rn.validate({**GOOD_DIGEST, 'as_of': bad}, TEXT, [], TODAY)
        assert out['as_of'] == TODAY.isoformat() and out['date_read'] is False
    assert rn.validate({**GOOD_DIGEST, 'document_type': 'wizardry'}, TEXT, [], TODAY)['document_type'] == 'other'


def test_a_document_stays_in_view_for_ninety_days_or_two_weeks_past_its_last_key_date():
    assert rn.expiry('2026-10-14', []) == '2027-01-12'
    assert rn.expiry('2026-10-14', [{'date': '2027-03-01'}]) == '2027-03-15'
    assert rn.expiry('2026-10-14', [{'date': '2026-11-17'}], keep_days=30) == '2026-12-01'


# --------------------------------------------------------------- reading it

def test_a_long_document_is_read_in_pieces_and_never_more_than_the_cap():
    pages = [{'combined': 'x' * 2500} for _ in range(40)]
    pieces = rn.chunks({'pages': pages})
    assert len(pieces) == rn.MAX_CHUNKS and all(len(p) <= rn.CHUNK_CHARS for p in pieces)
    assert rn.chunks({'text': 'short'}) == ['short']


def test_ingest_keeps_the_checked_digest_and_reports_what_it_dropped():
    ai = FakeAI(GOOD_DIGEST)
    result = rn.ingest('/tmp/acme_ipo.pdf', EXTRACTED, ai, CONFIG, TODAY)
    assert result['document_type'] == 'research_document' and result['ai_read'] and len(ai.calls) == 1
    note = result['note']
    assert note['symbols'] == ['ACME', 'SCOM'] and len(note['facts']) == 3 and note['filename'] == 'acme_ipo.pdf'
    assert note['expires_on'] == '2027-01-12'                                # the later of 90 days and last key date + 14
    assert len(result['rejected']) == 5
    assert rn.notes(TODAY)[0]['id'] == note['id']
    assert (rn._dir() / 'texts' / f"{note['id']}.txt").read_text() == TEXT


def test_uploading_the_same_document_again_replaces_it():
    rn.ingest('/tmp/acme_ipo.pdf', EXTRACTED, FakeAI(GOOD_DIGEST), CONFIG, TODAY)
    rn.ingest('/tmp/acme_ipo.pdf', EXTRACTED, FakeAI(GOOD_DIGEST), CONFIG, TODAY)
    assert len(rn.notes(TODAY)) == 1


def test_with_no_ai_the_text_is_kept_and_the_upload_says_how_to_fix_it():
    result = rn.ingest('/tmp/acme_ipo.pdf', EXTRACTED, None, CONFIG, TODAY)
    assert result['ai_read'] is False and result['note']['facts'] == [] and result['note']['read_by_ai'] is False
    lines = rn.describe(result)
    assert len(lines) == 1 and 'GROQ_API_KEY' in lines[0] and 'only its text was kept' in lines[0]


def test_an_ai_that_gives_no_answer_keeps_nothing_unchecked():
    result = rn.ingest('/tmp/acme_ipo.pdf', EXTRACTED, FakeAI(None), CONFIG, TODAY)
    assert result['ai_read'] is False and result['note']['summary'] == ''


def test_the_upload_describes_what_was_kept_in_plain_words():
    result = rn.ingest('/tmp/acme_ipo.pdf', EXTRACTED, FakeAI(GOOD_DIGEST), CONFIG, TODAY)
    text = '\n'.join(describe(result))
    assert 'No buy or sell ratings' in text and 'IPO briefing' in text and 'Acme Telecom IPO Briefing' in text
    assert 'Kept 3 checked figures and 2 dates' in text and 'ACME, SCOM' in text
    assert 'Not used: Dividend yield: the quote is not in the document' in text
    assert 'never places a trade' in text


# ------------------------------------------------------------ using it later

def kept(digest=GOOD_DIGEST, name='acme.pdf'):
    return rn.ingest(f'/tmp/{name}', EXTRACTED, FakeAI(digest), CONFIG, TODAY)['note']


def test_the_ai_sees_documents_that_name_the_stock_and_not_others():
    kept()
    line = rn.context_for('ACME', CONFIG, TODAY)
    assert 'IPO briefing of 2026-10-14' in line and 'Offer price KES 20.00 per share' in line
    assert 'Offer closes 2026-11-03' in line
    assert rn.context_for('SCOM', CONFIG, TODAY)                              # named too
    assert rn.context_for('KCB', CONFIG, TODAY) is None                       # not named, and the document names stocks
    assert rn.context_for('AAPL', CONFIG, TODAY) is None


def test_a_document_about_a_whole_market_is_background_for_stocks_in_it_only():
    kept({**GOOD_DIGEST, 'symbols': [], 'markets': ['kenya'], 'document_type': 'policy_notice'}, 'policy.pdf')
    assert 'policy or regulatory notice' in rn.context_for('KCB', CONFIG, TODAY)
    assert rn.context_for('AAPL', CONFIG, TODAY) is None                      # a US stock is not in Kenya
    assert rn.market_of('KCB', CONFIG) == 'kenya' and rn.market_of('BTC-USD', CONFIG) == 'crypto'


def test_a_document_stops_reaching_the_ai_when_it_is_stale():
    kept()
    assert rn.context_for('ACME', CONFIG, date(2027, 1, 12))
    assert rn.context_for('ACME', CONFIG, date(2027, 1, 13)) is None
    old = rn.notes(date(2027, 2, 1))[0]
    assert old['expired'] is True                                              # still listed for the operator
    assert rn.notes(date(2027, 2, 1), include_expired=False) == []


def test_at_most_three_documents_and_a_bounded_amount_of_text_reach_the_ai():
    for i in range(6):
        kept({**GOOD_DIGEST, 'title': f'Briefing {i}', 'as_of': f'2026-10-{10 + i:02d}'}, f'b{i}.pdf')
    line = rn.context_for('ACME', CONFIG, TODAY)
    assert line.count(' || ') <= 2 and len(line) <= 700


def test_a_document_can_be_removed_so_the_ai_stops_seeing_it():
    note = kept()
    assert rn.remove(note['id']) is True and rn.remove(note['id']) is False
    assert rn.notes(TODAY) == [] and rn.context_for('ACME', CONFIG, TODAY) is None


def test_the_trade_review_prompt_carries_the_documents_as_background(monkeypatch):
    from src.agent.llm_orchestrator import LLMOrchestrator
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    orch = LLMOrchestrator({'primary_llm_provider': 'gemini'})
    seen = {}

    def fake(system, user, fallback, model_override=None, purpose='volume'):
        seen['user'] = user
        return {'action': 'buy', 'confidence': 0.6, 'reasoning': 'ok'}

    monkeypatch.setattr(orch, '_complete', fake)
    orch.validate_trade('ACME', {'action': 'buy', 'confidence': 0.7}, {'close': 20.0},
                        research_context={'documents': 'IPO briefing of 2026-10-14: Acme sells shares at KES 20.00'})
    assert 'Documents on file: IPO briefing of 2026-10-14' in seen['user'] and 'Broker research' not in seen['user']
    orch.validate_trade('ACME', {'action': 'sell', 'confidence': 0.7}, {'close': 20.0},
                        research_context={'recommendation': 'BUY', 'target_price': 25, 'rationale': 'cheap'})
    assert 'Broker research: BUY' in seen['user'] and 'Documents on file' not in seen['user']


# --------------------------------------------- through the upload pipeline

@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    em = EscalationManager(str(tmp_path / 'esc.db'))
    from src.agent import market_pulse
    monkeypatch.setattr(market_pulse, 'page_texts', lambda path: [TEXT])
    import src.agent.broker_research_ingest as bri
    monkeypatch.setattr(bri.pdf_parser, 'extract_all', lambda path: EXTRACTED)
    yield em
    em.close()


def test_a_pdf_with_no_ratings_is_kept_as_a_document_not_thrown_away(pipeline):
    ing = BrokerResearchIngest(FakeAI(GOOD_DIGEST), pipeline, CONFIG)
    out = ing.process_pdf('/tmp/acme_ipo.pdf')
    assert out['status'] == 'completed' and out['document_type'] == 'research_document'
    assert any('Kept 3 checked figures' in a for a in out['actions'])
    assert not any('0 ratings' in a for a in out['actions'])
    assert len(rn.notes(TODAY)) == 1


def test_an_ai_is_not_asked_for_ratings_in_a_document_that_has_none(pipeline):
    ai = FakeAI(GOOD_DIGEST)
    assert has_rating_words(TEXT) is False
    BrokerResearchIngest(ai, pipeline, CONFIG).process_pdf('/tmp/acme_ipo.pdf')
    assert len(ai.calls) == 1                                                  # the digest only, not a rating hunt


def test_a_rating_note_whose_list_is_wrapped_under_any_key_is_read_not_dropped(pipeline, monkeypatch):
    note = "Safaricom: recommendation BUY, target price KES 40, current price KES 30."
    import src.agent.broker_research_ingest as bri
    monkeypatch.setattr(bri.pdf_parser, 'extract_all', lambda path: {'text': note, 'pages': [{'combined': note}],
                                                                      'metadata': {'pages_count': 1}})
    from src.agent import market_pulse
    monkeypatch.setattr(market_pulse, 'page_texts', lambda path: [note])
    row = {'symbol': 'SCOM', 'market': 'kenyan', 'current_price': 30.0, 'target_price': 40.0,
           'upside_pct': 33.3, 'recommendation': 'BUY', 'rationale': 'cheap', 'confidence': 0.9}
    for wrapper in ({'ratings': [row]}, {'results': [row]}, {'recommendations': [row]}, [row], row):
        out = BrokerResearchIngest(FakeAI(wrapper), pipeline, CONFIG).process_pdf('/tmp/note.pdf')
        assert out['document_type'] == 'analyst_note' and out['signals_processed'] == 1, wrapper


def test_items_from_reads_the_shapes_an_ai_gives_and_ignores_the_rest():
    row = {'symbol': 'KCB', 'recommendation': 'HOLD'}
    assert items_from({'recommendations': [row]}) == [row] and items_from([row, 'x']) == [row]
    assert items_from({'anything_at_all': [row]}) == [row] and items_from(row) == [row]
    assert items_from({'notes': ['a', 'b']}) == [] and items_from(None) == [] and items_from({'recommendations': []}) == []


# --------------------------------------------------------------------- API

@pytest.fixture
def api(tmp_path, monkeypatch):
    import src.api.api_server as api_server
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({role: {'password': bcrypt.hashpw(b'x', bcrypt.gensalt()).decode(), 'role': role}
                                 for role in ('viewer', 'operator')}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    client = api_server.TradingAPI(SimpleNamespace(components={}, config={}), {}).app.test_client()
    return client, lambda role: {'Authorization': f"Bearer {create_token(role, role)}"}


def test_the_library_is_listed_for_any_logged_in_user_and_removal_is_for_operators(api):
    client, header = api
    note = kept()
    data = client.get('/api/research/library', headers=header('viewer')).get_json()
    assert [n['id'] for n in data['notes']] == [note['id']] and data['notes'][0]['facts'][0]['quote']
    assert client.get('/api/research/library').status_code in (401, 403)
    assert client.delete(f"/api/operator/research-library/{note['id']}", headers=header('viewer')).status_code == 403
    assert client.delete(f"/api/operator/research-library/{note['id']}", headers=header('operator')).status_code == 200
    assert client.delete(f"/api/operator/research-library/{note['id']}", headers=header('operator')).status_code == 404
    assert client.get('/api/research/library', headers=header('viewer')).get_json()['notes'] == []


# ------------------------------------------------------------ linked pages

LINKED_TEXT = ("Acme Telecom audited results for the year. Net profit grew 18 percent to KES 7.4 billion. "
               "The board proposed a final dividend of KES 1.50 per share, payable on 12 December 2026. ") * 2
PAGE_DIGEST = {
    'document_type': 'company_results', 'title': 'Acme audited results', 'as_of': '2026-10-01',
    'summary': 'Profit rose and a dividend is proposed.', 'symbols': ['ACME'], 'markets': ['kenya'],
    'facts': [{'label': 'Dividend', 'value': 'KES 1.50 per share',
               'quote': 'final dividend of KES 1.50 per share'},
              {'label': 'Profit', 'value': 'KES 9.9 billion', 'quote': 'Net profit grew 18 percent to KES 7.4 billion'}],
    'dates': [{'label': 'Dividend paid', 'date': '2026-12-12', 'quote': 'payable on 12 December 2026'}], 'risks': []}
LINK = 'https://acme.example/results.pdf'


def fake_fetcher(url):
    return {'url': url, 'title': 'Acme results page', 'text': LINKED_TEXT, 'content_type': 'text/html'}


def with_links(links):
    text = TEXT + '\n' + '\n'.join(links)
    ex = {**EXTRACTED, 'text': text, 'links': links[:1]}
    return rn.ingest('/tmp/acme.pdf', ex, FakeAI(GOOD_DIGEST), CONFIG, TODAY)['note']


def test_the_links_a_document_carries_are_kept_without_duplicates_or_junk():
    ex = {'text': 'See https://a.example/x, and (https://b.example/y). Also http://c.example/z. https://a.example/x',
          'links': ['https://a.example/x', 'mailto:x@y.z', 'javascript:alert(1)']}
    assert rn.links_in(ex, ex['text']) == ['https://a.example/x', 'https://b.example/y', 'http://c.example/z']
    many = {'text': ' '.join(f'https://h.example/{i}' for i in range(90))}
    assert len(rn.links_in(many, many['text'])) == rn.MAX_LINKS


def test_a_new_document_lists_its_links_and_says_nothing_is_downloaded_yet():
    note = with_links([LINK, 'https://acme.example/agm'])
    assert note['links'] == [LINK, 'https://acme.example/agm'] and note['evidence'] == []
    lines = rn.describe({'note': note, 'ai_read': True})
    assert any('links to 2 web pages' in l and 'until you press Fetch' in l for l in lines)


def test_following_a_link_keeps_only_checked_figures_with_their_source():
    note = with_links([LINK])
    out = rn.fetch_link(note['id'], LINK, FakeAI(PAGE_DIGEST), CONFIG, TODAY, fake_fetcher)
    assert out['ok'] is True
    ev = out['evidence']
    assert ev['url'] == LINK and ev['status'] == 'read' and ev['title'] == 'Acme results page'
    assert [f['label'] for f in ev['facts']] == ['Dividend']                       # the wrong profit figure is dropped
    assert any('Profit' in r and 'not in the quoted words' in r for r in ev['rejected'])
    stored = rn.notes(TODAY)[0]
    assert stored['evidence'][0]['facts'][0]['value'] == 'KES 1.50 per share'
    assert any((rn._dir() / 'texts').glob(f"{note['id']}.*.txt"))


def test_only_a_link_the_document_carries_can_be_followed():
    note = with_links([LINK])
    asked = []

    def spy(url):
        asked.append(url)
        return fake_fetcher(url)

    out = rn.fetch_link(note['id'], 'https://evil.example/steal', FakeAI(PAGE_DIGEST), CONFIG, TODAY, spy)
    assert out['ok'] is False and out['status'] == 400 and asked == []
    assert rn.fetch_link('nope', LINK, FakeAI(PAGE_DIGEST), CONFIG, TODAY, spy)['status'] == 404


def test_a_refused_or_failed_page_is_reported_and_nothing_is_kept():
    from src.utils.safe_fetch import FetchError
    note = with_links([LINK])

    def refuse(url):
        raise FetchError('acme.example points to a private or internal address, so it is not followed')

    out = rn.fetch_link(note['id'], LINK, FakeAI(PAGE_DIGEST), CONFIG, TODAY, refuse)
    assert out['ok'] is False and out['status'] == 422 and 'private or internal' in out['error']
    assert rn.notes(TODAY)[0]['evidence'] == []
    assert rn.fetch_link(note['id'], LINK, None, CONFIG, TODAY, fake_fetcher)['status'] == 409     # no AI
    assert rn.fetch_link(note['id'], LINK, FakeAI(None), CONFIG, TODAY, fake_fetcher)['status'] == 502


def test_fetching_a_link_again_replaces_its_evidence_and_a_reupload_keeps_it():
    note = with_links([LINK])
    rn.fetch_link(note['id'], LINK, FakeAI(PAGE_DIGEST), CONFIG, TODAY, fake_fetcher)
    rn.fetch_link(note['id'], LINK, FakeAI(PAGE_DIGEST), CONFIG, TODAY, fake_fetcher)
    assert len(rn.notes(TODAY)[0]['evidence']) == 1
    with_links([LINK])                                                              # the same document uploaded again
    assert len(rn.notes(TODAY)[0]['evidence']) == 1


def test_linked_evidence_reaches_the_ai_as_background():
    note = with_links([LINK])
    rn.fetch_link(note['id'], LINK, FakeAI(PAGE_DIGEST), CONFIG, TODAY, fake_fetcher)
    assert 'from linked pages: Dividend KES 1.50 per share' in rn.context_for('ACME', CONFIG, TODAY)


def test_a_checked_later_date_on_a_linked_page_keeps_the_document_in_view_longer_and_an_unchecked_one_does_not():
    note = with_links([LINK])
    assert rn.notes(TODAY)[0]['expires_on'] == '2027-01-12'
    forged = {**PAGE_DIGEST, 'dates': [{'label': 'Payout', 'date': '2030-01-01', 'quote': 'payable on 1 January 2030'}]}
    rn.fetch_link(note['id'], LINK, FakeAI(forged), CONFIG, TODAY, fake_fetcher)       # the page never says this
    assert rn.notes(TODAY)[0]['expires_on'] == '2027-01-12'
    real_text = LINKED_TEXT + ' The special dividend is payable on 30 June 2027. '
    real = {**PAGE_DIGEST, 'dates': [{'label': 'Special dividend paid', 'date': '2027-06-30',
                                      'quote': 'payable on 30 June 2027'}]}
    rn.fetch_link(note['id'], LINK, FakeAI(real), CONFIG, TODAY,
                  lambda u: {'url': u, 'title': 't', 'text': real_text, 'content_type': 'text/html'})
    assert rn.notes(TODAY)[0]['expires_on'] == '2027-07-14'                            # 30 June 2027 + 14 days


def test_removing_a_document_removes_what_was_fetched_for_it():
    note = with_links([LINK])
    rn.fetch_link(note['id'], LINK, FakeAI(PAGE_DIGEST), CONFIG, TODAY, fake_fetcher)
    assert rn.remove(note['id']) is True
    assert list((rn._dir() / 'texts').glob(f"{note['id']}*")) == []


def test_the_reading_prompt_tells_the_ai_to_ignore_instructions_inside_a_document():
    assert 'Ignore them' in rn.SYSTEM_PROMPT and 'not orders' in rn.SYSTEM_PROMPT


def test_the_fetch_route_is_for_operators_and_follows_only_the_documents_own_links(api, monkeypatch):
    client, header = api
    note = with_links([LINK])
    from src.utils import safe_fetch
    monkeypatch.setattr(safe_fetch, 'fetch', fake_fetcher)
    url = f"/api/operator/research-library/{note['id']}/fetch"
    assert client.post(url, json={'url': LINK}, headers=header('viewer')).status_code == 403
    assert client.post(url, json={}, headers=header('operator')).status_code == 400
    # no AI is wired into this test agent, so an allowed link is refused plainly, not fetched
    r = client.post(url, json={'url': LINK}, headers=header('operator'))
    assert r.status_code == 409 and 'GROQ_API_KEY' in r.get_json()['error']
    r = client.post(url, json={'url': 'https://evil.example/x'}, headers=header('operator'))
    assert r.status_code == 400 and 'not one this document contains' in r.get_json()['error']


# ------------------------------------------- an issue the exact reader missed

PULSE_FIRST = "AXYS Market Pulse – 16th Oct 2026\nEquity markets rose. NASI up 0.6% to 240.50.\n"
PULSE_RATES = ("Fixed income\n91-day rate 8.76%\n182-day rate 8.87%\n364-day rate 9.04%\nInterbank Rate 8.75%\n"
               "US Dollar 129.10 129.40\nEuro 150.20 150.90\n")


def test_a_market_pulse_without_the_usual_scorecard_is_still_recognised_by_its_title_and_rates():
    from src.agent import market_pulse as mp
    classic = [PULSE_FIRST, "MARKET SCORECARD\nrows"]
    assert mp.is_market_pulse(classic)
    odd = [PULSE_FIRST, PULSE_RATES]                                  # no 'Market Scorecard' anywhere
    assert mp.is_market_pulse(odd)
    assert mp.is_market_pulse([PULSE_FIRST.replace('Market Pulse', 'MARKET  PULSE'), PULSE_RATES])


def test_other_documents_are_still_not_taken_for_a_market_pulse():
    from src.agent import market_pulse as mp
    assert not mp.is_market_pulse([PULSE_FIRST])                                       # title but no rate lines
    assert not mp.is_market_pulse(["Acme IPO briefing\n", "A note that mentions the Market Pulse and 91-day rate 8.76%"])
    assert not mp.is_market_pulse([TEXT])
    assert not mp.is_market_pulse([])
    assert not mp.is_market_pulse(["Weekly note\n", "AXYS Market Pulse – 16th Oct 2026\n" + PULSE_RATES])   # title not on page 1


def test_a_pulse_with_no_scorecard_stores_its_rates_exchange_rates_and_says_so(tmp_path, monkeypatch):
    from src.agent import market_pulse as mp
    monkeypatch.setattr(mp, '_dir', lambda: tmp_path / 'market_pulse')
    out = mp.ingest('pulse.pdf', [PULSE_FIRST, PULSE_RATES])
    assert out['document_type'] == 'market_pulse' and out['stocks_read'] == 0 and out['stocks_mapped'] == 0
    assert out['rates']['tbill_91'] == pytest.approx(0.0876) and out['rates']['interbank'] == pytest.approx(0.0875)
    text = '\n'.join(describe({**out, 'status': 'completed'}))        # the upload route adds the status
    assert 'no Market Scorecard table that could be read' in text and 'Stored price, earnings' not in text
    assert '91-day T-bill rate 8.76%' in text


def test_the_upload_route_gives_such_an_issue_to_the_exact_reader_not_the_library(pipeline, monkeypatch, tmp_path):
    from src.agent import market_pulse
    import src.agent.broker_research_ingest as bri
    monkeypatch.setattr(market_pulse, '_dir', lambda: tmp_path / 'market_pulse')
    monkeypatch.setattr(market_pulse, 'page_texts', lambda path: [PULSE_FIRST, PULSE_RATES])
    ai = FakeAI(GOOD_DIGEST)
    out = BrokerResearchIngest(ai, pipeline, CONFIG).process_pdf('/tmp/pulse.pdf')
    assert out['document_type'] == 'market_pulse' and ai.calls == [] and rn.notes(TODAY) == []


# ------------------------------------------------ figures rounded by the reader

def test_a_figure_rounded_from_the_quoted_one_is_the_same_figure_but_a_different_one_is_not():
    t = rn._norm("The 91-day rate was 8.7557% and the interbank rate 8.75%, KCB rose 89.50 points")
    ok = lambda value, quote: rn.check_fact({'label': 'x', 'value': value, 'quote': quote}, t)[0] is not None
    assert ok('8.76%', '91-day rate was 8.7557%')                  # rounded to two places
    assert ok('8.8%', '91-day rate was 8.7557%')                   # and to one
    assert ok('89.5', 'KCB rose 89.50 points')                     # a trailing zero
    assert not ok('8.9%', '91-day rate was 8.7557%')               # a different figure
    assert not ok('8.7558%', '91-day rate was 8.7557%')            # more places than were quoted, and wrong
    assert not ok('8.7557', 'interbank rate 8.75%')                # cannot invent precision the quote lacks


# ------------------------------------------------------------ the market costs

def test_the_market_costs_follow_the_sources_checked_and_the_config_agrees():
    from src.agent import commodity_watch as cw
    config = json.load(open('config/config.json', encoding='utf-8'))
    assert cw.costs({}) == {'uk_equities': 0.7, 'japan_equities': 0.3, 'france_equities': 0.6,
                            'south_africa_equities': 0.7}
    assert cw.costs(config) == cw.costs({})                                   # config.json and the defaults match
    note = config['watch_only']['equity_markets']['_comment']
    for needle in ('0.4%', 'GBP 1.50', '0.25%', 'NOT confirmed', 'assumption'):
        assert needle in note, needle
