"""AIB-AXYS's Daily Whispers rating sheet, sent as a picture
(src/agent/daily_whispers.py), and the Research page's account of what the
agent did with each upload. The rows are the 24 September 2026 sheet's."""
import json
from datetime import date
from types import SimpleNamespace

import pytest

import src.utils.paths as paths
from src.agent import daily_whispers as dw
from src.agent import market_pulse as mp

SHEET = {
    'report_title': 'The Daily Whispers', 'report_date': '2026-09-24',
    'rows': [
        {'security_name': 'NCBA Group', 'ytd_pct': 7.1, 'current_price': 90.00, 'target_price': 110.07,
         'upside_pct': 22.3, 'recommendation': 'BUY',
         'rationale': ['Strong earnings momentum, with NII growing 22.0% y/y.']},
        {'security_name': 'Equity Group Holdings', 'ytd_pct': 60.3, 'current_price': 107.00,
         'target_price': 126.32, 'upside_pct': 18.1, 'recommendation': 'BUY', 'rationale': []},
        {'security_name': 'KenGen Co Plc', 'ytd_pct': 17.6, 'current_price': 10.80, 'target_price': 12.40,
         'upside_pct': 14.8, 'recommendation': 'BUY', 'rationale': []},
        {'security_name': 'Williamson Tea Kenya', 'ytd_pct': 7.0, 'current_price': 160.00,
         'target_price': 173.67, 'upside_pct': 8.5, 'recommendation': 'BUY', 'rationale': []},
        {'security_name': 'Jubilee Holdings Ltd', 'ytd_pct': 21.6, 'current_price': 407.25,
         'target_price': 429.00, 'upside_pct': 5.3, 'recommendation': 'HOLD', 'rationale': []},
    ]}
ON = date(2026, 9, 24)
no_close = lambda symbol, on: None


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA_DIR', tmp_path)
    return tmp_path


def test_every_row_of_the_sheet_is_matched_and_checked():
    accepted, rejected = dw.verify(SHEET, ON, no_close)
    assert rejected == []
    assert [(r['symbol'], r['recommendation']) for r in accepted] == [
        ('NCBA', 'BUY'), ('EQTY', 'BUY'), ('KEGN', 'BUY'), ('WTK', 'BUY'), ('JUB', 'HOLD')]
    assert accepted[0]['target_price'] == 110.07 and accepted[0]['upside_pct'] == 22.3
    assert 'NII growing' in accepted[0]['rationale']


def test_a_misread_digit_or_a_stranger_is_rejected_with_the_reason():
    rows = [dict(SHEET['rows'][0], target_price=1100.07),                    # a digit too many
            dict(SHEET['rows'][1], security_name='Acme Holdings'),            # not listed
            dict(SHEET['rows'][2], recommendation='STRONG MAYBE'),
            dict(SHEET['rows'][3], current_price=None),
            SHEET['rows'][4], SHEET['rows'][4]]                               # twice
    accepted, rejected = dw.verify({'rows': rows}, ON, no_close)
    assert [r['symbol'] for r in accepted] == ['JUB']
    reasons = [r['reason'] for r in rejected]
    assert 'numbers disagree' in reasons[0] and 'not matched' in reasons[1]
    assert 'not recognised' in reasons[2] and 'unreadable' in reasons[3] and 'twice' in reasons[4]


def test_a_price_far_from_the_recorded_close_is_rejected():
    close = lambda symbol, on: 60.0 if symbol == 'NCBA' else None
    accepted, rejected = dw.verify({'rows': SHEET['rows'][:2]}, ON, close)
    assert [r['symbol'] for r in accepted] == ['EQTY'] and 'recorded close' in rejected[0]['reason']
    near = lambda symbol, on: 89.5
    accepted, _ = dw.verify({'rows': SHEET['rows'][:1]}, ON, near)
    assert accepted[0]['price_checked']


class _Eyes:
    enabled = True

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def read_image_json(self, system, user, image_b64, media_type, schema):
        self.calls.append(media_type)
        return self.reply


def test_the_picture_is_read_and_dated(tmp_path):
    img = tmp_path / 'whispers.jpg'
    img.write_bytes(b'\xff\xd8 not really a jpeg')
    eyes = _Eyes(SHEET)
    sheet = dw.read(str(img), eyes, today=date(2026, 9, 26), last_close=no_close)
    assert eyes.calls == ['image/jpeg'] and sheet['as_of'] == '2026-09-24' and sheet['date_read']
    assert len(sheet['accepted']) == 5
    undated = dw.read(str(img), _Eyes({**SHEET, 'report_date': '2026-12-01'}),
                      today=date(2026, 9, 26), last_close=no_close)
    assert undated['as_of'] == '2026-09-26' and not undated['date_read']      # a future date is not believed
    with pytest.raises(ValueError, match='no AI model'):
        dw.read(str(img), None)
    with pytest.raises(ValueError, match='no recommendation table'):
        dw.read(str(img), _Eyes({'rows': []}), last_close=no_close)
    with pytest.raises(ValueError, match='could read'):
        dw.read(str(img), _Eyes(None), last_close=no_close)


def test_the_rating_reaches_the_ai_trade_review_for_30_days(data_dir):
    accepted, _ = dw.verify(SHEET, ON, no_close)
    assert dw.remember(accepted, 'whispers.jpg') == 5
    assert dw.remember(accepted, 'whispers.jpg') == 0                         # the same day again
    ctx = mp.research_context('NCBA', today=date(2026, 9, 26))
    assert ctx['recommendation'] == 'BUY' and ctx['target_price'] == 110.07
    assert 'AIB-AXYS rating 2026-09-24: BUY at 90.0, target 110.07 (+22.3%)' in ctx['rationale']
    assert mp.research_context('NCBA', today=date(2026, 10, 30)) is None     # stale after 30 days


def _ingest(tmp_path, llm):
    from src.agent.broker_research_ingest import BrokerResearchIngest
    from src.agent.escalation_manager import EscalationManager
    em = EscalationManager(str(tmp_path / 'esc.db'))
    cfg = {'data_manager': {'nse_symbols': ['NCBA', 'EQTY']}}
    return BrokerResearchIngest(llm, em, cfg), em


def test_an_uploaded_sheet_says_what_the_agent_did(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(dw, 'last_real_close', no_close)
    ing, em = _ingest(tmp_path, _Eyes(SHEET))
    img = tmp_path / 'whispers.jpg'
    img.write_bytes(b'\xff\xd8')
    out = ing.process_image(str(img))
    assert out['status'] == 'completed' and out['document_type'] == 'recommendation_sheet'
    text = ' '.join(out['actions'])
    assert 'NCBA BUY at 90.0, target 110.07 (+22.3%)' in text
    assert 'never places a trade by itself' in text
    assert 'Added to the research watchlist' in text and 'approval queue' in text
    history = em.get_upload_history()
    assert history[0]['document_type'] == 'recommendation_sheet' and history[0]['summary'] == out['actions']


def test_a_market_pulse_upload_says_what_the_agent_did(data_dir, tmp_path, monkeypatch):
    from tests.test_market_pulse import TEXTS
    import src.connectors.nse_connector as nse_connector
    csvs = tmp_path / 'csv'
    csvs.mkdir()
    import src.connectors.nse_scraper as nse_scraper
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', csvs)
    monkeypatch.setattr(nse_scraper, 'DATA_DIR', csvs)
    monkeypatch.setattr(mp, 'page_texts', lambda path: TEXTS)
    ing, em = _ingest(tmp_path, None)
    out = ing.process_pdf('Daily_Market_Watch.pdf')
    text = ' '.join(out['actions'])
    assert 'Read as the AIB-AXYS Market Pulse of 2026-09-23' in text
    assert '91-day T-bill rate 8.78%' in text and 'nothing was sent to the approval queue' in text
    assert em.get_upload_history()[0]['summary'] == out['actions']


def test_a_failed_upload_says_nothing_was_changed(tmp_path):
    ing, _ = _ingest(tmp_path, None)
    img = tmp_path / 'whispers.png'
    img.write_bytes(b'\x89PNG')
    out = ing.process_image(str(img))
    assert out['status'] == 'failed' and out['actions'][0].startswith('Nothing was changed: no AI model')


def test_the_old_readers_false_ratings_are_withdrawn(tmp_path):
    from src.agent.escalation_manager import EscalationManager
    em = EscalationManager(str(tmp_path / 'esc.db'))
    old = em.record_upload('Daily_Market_Watch_23rd_September_2026_1.pdf')
    note = em.record_upload('KCB_initiation.pdf')
    for up, sym in ((old, 'BRIT'), (old, 'KCB'), (note, 'EQTY')):
        sid = em.record_signal(up, {'symbol': sym, 'recommendation': 'HOLD',
                                    'rationale': '[Extracted Analyst Rationale]: ...'})
        em.create_escalation(sid, sym, 'follow', 'not configured', 'high')
    em.add_to_watchlist('KCB', 'kenyan', f'upload_{old}', 'HOLD', 34.8)
    em.add_to_watchlist('SCOM', 'kenyan', f'upload_{note}', 'BUY', 45)
    reopened = EscalationManager(str(tmp_path / 'esc.db'))                    # runs on every start
    assert [e['symbol'] for e in reopened.get_pending_escalations()] == ['EQTY']
    assert [w['symbol'] for w in reopened.get_active_watchlist()] == ['SCOM']
    assert reopened.withdraw_misread_market_reports()['escalations'] == 0     # nothing left to do
    hist = {h['filename']: h for h in reopened.get_upload_history()}
    assert 'withdrawn' in hist['Daily_Market_Watch_23rd_September_2026_1.pdf']['summary'][0]


def test_claude_reads_the_picture_under_the_budget(tmp_path):
    from src.agent.ai_budget import AiBudget
    from src.agent.claude_provider import ClaudeProvider
    sent = {}

    class _Messages:
        def create(self, **kw):
            sent.update(kw)
            return SimpleNamespace(stop_reason='end_turn', usage={'input_tokens': 1800, 'output_tokens': 600},
                                   content=[SimpleNamespace(type='text', text=json.dumps(SHEET))])

    budget = AiBudget(tmp_path / 'spend.json', 20)
    claude = ClaudeProvider('k', budget, {}, client=SimpleNamespace(messages=_Messages()))
    out = claude.read_image_json('sys', 'user', 'AAAA', 'image/jpeg', dw.SCHEMA)
    assert out['rows'][0]['security_name'] == 'NCBA Group'
    image = sent['messages'][0]['content'][0]
    assert image['type'] == 'image' and image['source']['media_type'] == 'image/jpeg'
    assert sent['output_config']['format']['schema'] is dw.SCHEMA
    assert budget.summary()['spent_usd'] > 0


def test_without_claude_the_free_gemini_model_reads_it(monkeypatch):
    from src.agent import llm_orchestrator as lo
    monkeypatch.setenv('GEMINI_API_KEY', 'g')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    posted = {}

    def post(url, headers=None, json=None, timeout=None):
        posted.update(json)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'candidates': [{'content': {'parts': [{'text': __import__('json').dumps(SHEET)}]}}]})
    monkeypatch.setattr(lo.requests, 'post', post)
    llm = lo.LLMOrchestrator({})
    out = llm.read_image_json('sys', 'user', 'AAAA', 'image/png', dw.SCHEMA)
    assert out['report_date'] == '2026-09-24'
    assert posted['contents'][0]['parts'][0]['inline_data']['mime_type'] == 'image/png'
