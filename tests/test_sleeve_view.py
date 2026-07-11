from src.api.sleeve_view import build_sleeve_view


def test_filters_to_long_term_book():
    tickets = [
        {'symbol': 'SCOM', 'quantity': 100, 'book': 'trading', 'rationale': '', 'llm_reasoning': ''},
        {'symbol': 'EQTY', 'quantity': 50, 'book': 'long_term',
         'rationale': 'Sleeve accumulation: combined=0.7', 'llm_reasoning': ''},
    ]
    view = build_sleeve_view({}, tickets, [], nse_capital_kes=100000, capital_split_pct=0.5)
    assert len(view['pending_tickets']) == 1
    assert view['pending_tickets'][0]['symbol'] == 'EQTY'


def test_veto_flag_detected_from_rationale():
    tickets = [{'symbol': 'SCOM', 'quantity': 10, 'book': 'long_term',
               'rationale': 'Sleeve accumulation: combined=0.7 | VETO_FLAG',
               'llm_reasoning': 'profit warning'}]
    view = build_sleeve_view({}, tickets, [], nse_capital_kes=100000, capital_split_pct=0.5)
    assert view['pending_tickets'][0]['veto_flagged'] is True
    assert view['pending_tickets'][0]['llm_reasoning'] == 'profit warning'


def test_veto_unavailable_detected_from_rationale():
    tickets = [{'symbol': 'SCOM', 'quantity': 10, 'book': 'long_term',
               'rationale': 'Sleeve accumulation: combined=0.7 | VETO_UNAVAILABLE',
               'llm_reasoning': ''}]
    view = build_sleeve_view({}, tickets, [], nse_capital_kes=100000, capital_split_pct=0.5)
    assert view['pending_tickets'][0]['veto_unavailable'] is True
    assert view['pending_tickets'][0]['veto_flagged'] is False


def test_capital_target_computed():
    view = build_sleeve_view({}, [], [], nse_capital_kes=200000, capital_split_pct=0.4)
    assert view['capital_target_kes'] == 80000.0


def test_cumulative_dividends_excludes_declared_only():
    history = [
        {'amount_kes': 100.0, 'status': 'received'},
        {'amount_kes': 50.0, 'status': 'swept'},
        {'amount_kes': 30.0, 'status': 'declared'},
    ]
    view = build_sleeve_view({}, [], history, nse_capital_kes=0, capital_split_pct=0)
    assert view['cumulative_dividends_kes'] == 150.0


def test_holdings_passed_through():
    holdings = {'SCOM': {'quantity': 200, 'avg_entry_price_kes': 15.0}}
    view = build_sleeve_view(holdings, [], [], nse_capital_kes=0, capital_split_pct=0)
    assert view['holdings'] == [{'symbol': 'SCOM', 'quantity': 200, 'avg_entry_price_kes': 15.0}]
