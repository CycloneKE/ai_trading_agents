"""Alpaca broker integration for live and paper trading.

This talks to Alpaca's documented REST API directly with `requests`, the same
way src/connectors/coinbase_broker.py and src/connectors/oanda_broker.py do.
It used to use the `alpaca-trade-api` SDK, which was never installed in the
runtime image, so BrokerManager's guarded import failed, the 'alpaca' broker
type was never registered, and the broker the config marks `primary` silently
did not exist. The agent then traded against the internal simulator instead.

Installing that SDK would have fixed the symptom and introduced a worse
problem: alpaca-trade-api 0.48 pins `urllib3 < 1.25`, which forces urllib3
1.24.3 (2019) underneath `requests`. That is the HTTP stack carrying the API
credentials, and it also drags in alpha-vantage, asyncio-nats-client and
peewee for endpoints this project never calls.

The seven calls the connector actually made map one-to-one onto REST
endpoints, so the dependency bought nothing:

    connect / get_account_info  -> GET    /v2/account
    place_order                 -> POST   /v2/orders
    cancel_order                -> DELETE /v2/orders/{id}
    get_positions               -> GET    /v2/positions
    get_orders                  -> GET    /v2/orders
    get_portfolio_history       -> GET    /v2/account/portfolio/history

Docs: https://docs.alpaca.markets/reference/
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import requests

from .base_broker import BaseBroker, OrderRequest, OrderResponse, Position, AccountInfo

logger = logging.getLogger(__name__)

PAPER_BASE_URL = 'https://paper-api.alpaca.markets'
LIVE_BASE_URL = 'https://api.alpaca.markets'

# Every request carries a deadline. The SDK applied its own; plain requests
# does not, and a socket that never answers would hang the trading loop
# indefinitely rather than failing the cycle and moving on.
DEFAULT_TIMEOUT = 15.0

# The fractional-seconds group of an RFC 3339 timestamp.
_FRACTIONAL_SECONDS = re.compile(r'\.(\d+)')


class AlpacaAPIError(RuntimeError):
    """A non-2xx response from Alpaca, carrying their error message."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


# Crypto pairs are 'BTC-USD' here (the yfinance form every other part of the
# system uses) but 'BTC/USD' to Alpaca's order API, and positions come back
# as 'BTCUSD'. Without translation every crypto order was rejected as an
# unknown asset, and a held crypto position never matched its own symbol, so
# the agent could not tell it already owned it.
_CRYPTO_PAIR = re.compile(r'^([A-Z0-9]+)-(USDT|USDC|USD)$')
_CRYPTO_QUOTES = ('USDT', 'USDC', 'USD')


def is_crypto_symbol(symbol: Optional[str]) -> bool:
    return bool(_CRYPTO_PAIR.match(symbol or ''))


def to_alpaca_symbol(symbol: str) -> str:
    """'BTC-USD' -> 'BTC/USD'. Stock tickers pass through unchanged."""
    m = _CRYPTO_PAIR.match(symbol or '')
    return f"{m.group(1)}/{m.group(2)}" if m else symbol


def from_alpaca_symbol(symbol: Optional[str], asset_class: Optional[str] = None) -> Optional[str]:
    """'BTC/USD', or 'BTCUSD' for a crypto asset, -> 'BTC-USD'. Stocks unchanged."""
    if not symbol:
        return symbol
    if '/' in symbol:
        base, quote = symbol.split('/', 1)
        return f"{base}-{quote}"
    if asset_class == 'crypto':
        for quote in _CRYPTO_QUOTES:
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return f"{symbol[:-len(quote)]}-{quote}"
    return symbol


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse Alpaca's RFC 3339 timestamps into aware datetimes.

    Alpaca returns e.g. '2021-03-16T18:38:01.942282Z', sometimes with
    nanosecond precision, which datetime.fromisoformat cannot take. Normalise
    the fractional part to exactly six digits rather than lose the timestamp.
    """
    if value in (None, ''):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    text = _FRACTIONAL_SECONDS.sub(
        lambda m: '.' + m.group(1)[:6].ljust(6, '0'), text, count=1)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        logger.warning("Could not parse Alpaca timestamp %r", value)
        return None


def _as_float(value: Any) -> Optional[float]:
    """Alpaca sends numbers as strings, and omitted fields as null."""
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float_or_zero(value: Any) -> float:
    parsed = _as_float(value)
    return 0.0 if parsed is None else parsed


class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation over the REST API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.broker_name = "alpaca"

        self.api_key = config.get('api_key') or os.getenv('TRADING_ALPACA_API_KEY')
        self.api_secret = config.get('api_secret') or os.getenv('TRADING_ALPACA_API_SECRET')
        self.is_paper_trading = config.get('paper', True)
        self.timeout = float(config.get('timeout', DEFAULT_TIMEOUT))

        self.base_url = (PAPER_BASE_URL if self.is_paper_trading else LIVE_BASE_URL).rstrip('/')
        self.session: Optional[requests.Session] = None

    # -- plumbing ---------------------------------------------------------

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'APCA-API-KEY-ID': self.api_key or '',
            'APCA-API-SECRET-KEY': self.api_secret or '',
            'Content-Type': 'application/json',
        })
        return session

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """Issue one authenticated request and return the decoded body.

        Raises AlpacaAPIError on a non-2xx response so callers log Alpaca's
        own message ('insufficient buying power', 'asset not tradable')
        instead of a bare status code.
        """
        if self.session is None:
            self.session = self._build_session()
        url = f"{self.base_url}/v2/{path.lstrip('/')}"
        response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        if not response.ok:
            detail = response.text[:500]
            try:
                payload = response.json()
                detail = payload.get('message', detail) if isinstance(payload, dict) else detail
            except ValueError:
                pass
            raise AlpacaAPIError(
                f"{method} /v2/{path.lstrip('/')} returned {response.status_code}: {detail}",
                status_code=response.status_code)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # -- lifecycle --------------------------------------------------------

    def connect(self) -> bool:
        """Verify the credentials by fetching the account."""
        try:
            if not self.api_key or not self.api_secret:
                logger.error(
                    "Alpaca API keys not provided. Set TRADING_ALPACA_API_KEY and "
                    "TRADING_ALPACA_API_SECRET, or api_key/api_secret in the broker config.")
                self.is_connected = False
                return False

            self.session = self._build_session()
            account = self._request('GET', 'account')

            # Say which endpoint was reached. A paper run that silently
            # authenticated against the live endpoint is the one mistake here
            # that costs real money, and the URL is the only thing that
            # decides it.
            logger.info("Successfully connected to Alpaca at %s (paper=%s, account status=%s)",
                        self.base_url, self.is_paper_trading,
                        (account or {}).get('status', 'unknown'))
            self.is_connected = True
            self.last_connection_time = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from Alpaca API."""
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
        self.is_connected = False
        self.session = None
        return True

    # -- trading ----------------------------------------------------------

    def _to_order_response(self, o: Dict[str, Any]) -> OrderResponse:
        return OrderResponse(
            order_id=o.get('id'),
            client_order_id=o.get('client_order_id'),
            symbol=from_alpaca_symbol(o.get('symbol'), o.get('asset_class')),
            quantity=_as_float_or_zero(o.get('qty')),
            filled_quantity=_as_float_or_zero(o.get('filled_qty')),
            side=o.get('side'),
            order_type=o.get('type'),
            status=o.get('status'),
            created_at=_parse_timestamp(o.get('created_at')),
            updated_at=_parse_timestamp(o.get('updated_at')),
            limit_price=_as_float(o.get('limit_price')),
            stop_price=_as_float(o.get('stop_price')),
            filled_avg_price=_as_float(o.get('filled_avg_price')),
            broker_name=self.broker_name,
        )

    def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        """Place an order through Alpaca."""
        try:
            if not self.is_connected or self.session is None:
                if not self.connect():
                    return None

            tif = order.time_in_force
            if isinstance(order.quantity, (int, float)) and (order.quantity % 1 != 0):
                tif = "day"

            payload: Dict[str, Any] = {
                'symbol': to_alpaca_symbol(order.symbol),
                'qty': str(order.quantity),
                'side': order.side,
                'type': order.order_type,
                'time_in_force': tif,
                # Alpaca only fills pre/after-market orders when this is set
                # (requires a DAY limit order per their extended-hours rules)
                'extended_hours': bool(order.extended_hours),
            }
            if is_crypto_symbol(order.symbol):
                # Crypto trades around the clock: Alpaca accepts only gtc or
                # ioc for it and rejects the 'day' used for fractional
                # stock orders. Extended hours does not apply.
                if tif not in ('gtc', 'ioc'):
                    payload['time_in_force'] = 'gtc'
                payload.pop('extended_hours', None)
            if order.limit_price is not None:
                payload['limit_price'] = str(order.limit_price)
            if order.stop_price is not None:
                payload['stop_price'] = str(order.stop_price)
            if order.client_order_id:
                payload['client_order_id'] = order.client_order_id

            return self._to_order_response(self._request('POST', 'orders', json=payload))
        except Exception as e:
            logger.error(f"Error placing Alpaca order: {str(e)}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an Alpaca order."""
        try:
            self._request('DELETE', f'orders/{order_id}')
            return True
        except Exception as e:
            logger.error(f"Error canceling Alpaca order: {str(e)}")
            return False

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get Alpaca account details."""
        try:
            acc = self._request('GET', 'account') or {}
            return AccountInfo(
                account_id=acc.get('id'),
                cash=_as_float_or_zero(acc.get('cash')),
                equity=_as_float_or_zero(acc.get('equity')),
                buying_power=_as_float_or_zero(acc.get('buying_power')),
                initial_margin=_as_float_or_zero(acc.get('initial_margin')),
                maintenance_margin=_as_float_or_zero(acc.get('maintenance_margin')),
                day_trade_count=int(acc.get('daytrade_count') or 0),
                last_updated=datetime.now(timezone.utc),
                broker_name=self.broker_name,
            )
        except Exception as e:
            logger.error(f"Error getting Alpaca account info: {str(e)}")
            return None

    def fetch_positions(self) -> List[Position]:
        """Current positions. Raises on failure.

        The trading loop's position gate uses this rather than
        get_positions(), because get_positions() returns [] on an API error
        and "could not check" must never read as "holds nothing".
        """
        return [
            Position(
                symbol=from_alpaca_symbol(p.get('symbol'), p.get('asset_class')),
                quantity=_as_float_or_zero(p.get('qty')),
                avg_entry_price=_as_float_or_zero(p.get('avg_entry_price')),
                current_price=_as_float_or_zero(p.get('current_price')),
                market_value=_as_float_or_zero(p.get('market_value')),
                unrealized_pl=_as_float_or_zero(p.get('unrealized_pl')),
                unrealized_pl_percent=_as_float_or_zero(p.get('unrealized_plpc')),
                cost_basis=_as_float_or_zero(p.get('cost_basis')),
                broker_name=self.broker_name,
            )
            for p in (self._request('GET', 'positions') or [])
        ]

    def get_positions(self) -> List[Position]:
        """Get current Alpaca positions; [] on error (see fetch_positions)."""
        try:
            return self.fetch_positions()
        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {str(e)}")
            return []

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Open orders, optionally for one symbol (our form). Raises on failure.

        Filtered here rather than with Alpaca's `symbols` parameter, which
        would need the slash form for crypto; the open-order list is short.
        """
        orders = [self._to_order_response(o)
                  for o in (self._request('GET', 'orders',
                                          params={'status': 'open', 'limit': 500}) or [])]
        return [o for o in orders if symbol is None or o.symbol == symbol]

    def get_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get Alpaca open orders; [] on error (see fetch_open_orders)."""
        try:
            return self.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error getting Alpaca orders: {str(e)}")
            return []

    def get_order_by_client_order_id(self, client_order_id: str) -> Optional[OrderResponse]:
        """The order carrying our client_order_id, or None if Alpaca has none.

        Raises on any other failure. The order journal needs "not found" and
        "could not ask" kept apart: the first means the order never reached
        the broker, the second means nothing is known yet.
        """
        try:
            o = self._request('GET', 'orders:by_client_order_id',
                              params={'client_order_id': client_order_id})
        except AlpacaAPIError as e:
            if e.status_code == 404:
                return None
            raise
        return self._to_order_response(o) if o else None

    def get_portfolio_history(self, period: str = '1D', timeframe: str = '1Min') -> Dict[str, Any]:
        """Get Alpaca portfolio history."""
        try:
            history = self._request('GET', 'account/portfolio/history',
                                    params={'period': period, 'timeframe': timeframe}) or {}
            return {
                'timestamp': history.get('timestamp') or [],
                'equity': [_as_float_or_zero(e) for e in (history.get('equity') or [])],
                'profit_loss': [_as_float_or_zero(pl) for pl in (history.get('profit_loss') or [])],
                'profit_loss_pct': [_as_float_or_zero(p) for p in (history.get('profit_loss_pct') or [])],
            }
        except Exception as e:
            logger.error(f"Error getting Alpaca portfolio history: {str(e)}")
            return {}
