import pytest
from src.connectors.nse_universe import nse_tick_size, quantize_nse_price
from src.agent.nse_order_queue import NseOrderQueue
from src.agent.alerts import EmailAlerter


def test_nse_tick_size_bands():
    # Under 20: tick is 0.05
    assert nse_tick_size(5.0) == 0.05
    assert nse_tick_size(19.95) == 0.05

    # 20 to 50: tick is 0.25
    assert nse_tick_size(20.0) == 0.25
    assert nse_tick_size(42.5) == 0.25
    assert nse_tick_size(49.75) == 0.25

    # 50 to 100: tick is 0.50
    assert nse_tick_size(50.0) == 0.50
    assert nse_tick_size(75.25) == 0.50
    assert nse_tick_size(99.50) == 0.50

    # 100 and above: tick is 1.00
    assert nse_tick_size(100.0) == 1.00
    assert nse_tick_size(152.3) == 1.00
    assert nse_tick_size(350.0) == 1.00


def test_quantize_nse_price():
    # Under 20: quantized to nearest 0.05
    assert quantize_nse_price(15.22) == 15.20
    assert quantize_nse_price(15.23) == 15.25
    assert quantize_nse_price(4.01) == 4.00

    # 20 to 50: quantized to nearest 0.25
    assert quantize_nse_price(42.60) == 42.50
    assert quantize_nse_price(42.65) == 42.75
    assert quantize_nse_price(20.10) == 20.00
    assert quantize_nse_price(20.15) == 20.25

    # 50 to 100: quantized to nearest 0.50
    assert quantize_nse_price(67.20) == 67.00
    assert quantize_nse_price(67.35) == 67.50

    # 100 and above: quantized to nearest 1.00
    assert quantize_nse_price(142.30) == 142.00
    assert quantize_nse_price(142.70) == 143.00


def test_order_queue_quantizes_ticket_price(tmp_path):
    db_file = tmp_path / 'test_queue.db'
    queue = NseOrderQueue(str(db_file))

    # Pass unquantized price (42.60 on a 20-50 KES stock)
    tid = queue.create_ticket('EQTY', 'buy', 500, suggested_limit_price=42.60)
    assert tid is not None

    ticket = queue._get(tid)
    # Must be quantized to 42.50 (nearest multiple of 0.25)
    assert ticket['suggested_limit_price'] == 42.50


def test_order_queue_triggers_alerter(tmp_path, monkeypatch):
    dispatched = []

    class DummyAlerter:
        def notify_nse_ticket(self, t):
            dispatched.append(t)

    db_file = tmp_path / 'test_alert_queue.db'
    queue = NseOrderQueue(str(db_file), alerter=DummyAlerter())

    tid = queue.create_ticket('SCOM', 'buy', 1000, suggested_limit_price=15.22,
                              rationale='Dividend yield + Momentum')
    assert tid is not None
    assert len(dispatched) == 1
    assert dispatched[0]['id'] == tid
    assert dispatched[0]['symbol'] == 'SCOM'
    assert dispatched[0]['suggested_limit_price'] == 15.20  # quantized to 0.05
    assert dispatched[0]['quantity'] == 1000


def test_telegram_alert_delivery(monkeypatch):
    sent_payloads = []

    class FakeResponse:
        status_code = 200

    def fake_post(url, json=None, timeout=None):
        sent_payloads.append({'url': url, 'json': json})
        return FakeResponse()

    monkeypatch.setattr('requests.post', fake_post)

    env = {
        'TELEGRAM_BOT_TOKEN': 'test-bot-token-12345',
        'TELEGRAM_CHAT_ID': '12345678'
    }
    alerter = EmailAlerter(config={}, env=env, background=False)
    assert alerter.telegram_enabled is True

    alerter.notify_nse_ticket({
        'id': 42,
        'symbol': 'BAT',
        'side': 'buy',
        'quantity': 200,
        'suggested_limit_price': 380.0,
        'book': 'long_term',
        'ensemble_confidence': 0.85,
        'rationale': 'High dividend yield',
        'llm_reasoning': 'Sustainable dividend cover'
    })

    assert len(sent_payloads) == 1
    assert 'test-bot-token-12345' in sent_payloads[0]['url']
    card_text = sent_payloads[0]['json']['text']
    assert 'NSE Order Ticket #42' in card_text
    assert 'BAT' in card_text
    assert 'KES 380.00' in card_text
