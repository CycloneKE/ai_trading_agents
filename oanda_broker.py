"""
OANDA Broker for Forex Trading
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from base_broker import BaseBroker, OrderRequest, OrderResponse, Position, AccountInfo
import requests

class OandaBroker(BaseBroker):
    def get_status(self) -> Dict[str, Any]:
        """Return health/status info for OandaBroker."""
        return {
            'broker_name': self.broker_name,
            'is_connected': self.is_connected,
            'api_key_set': bool(self.api_key),
            'account_id_set': bool(self.account_id),
            'timestamp': datetime.now().isoformat(),
            'status': 'ok' if self.is_connected else 'not_connected'
        }
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.broker_name = "oanda"
        self.api_key = config.get('api_key', '')
        self.account_id = config.get('account_id', '')
        self.base_url = "https://api-fxpractice.oanda.com/v3/"

    def connect(self) -> bool:
        self.is_connected = True  # Placeholder: Add real auth if needed
        return True

    def disconnect(self) -> bool:
        self.is_connected = False
        return True

    def place_order(self, order: OrderRequest) -> Optional[OrderResponse]:
        # Placeholder: Implement real OANDA order placement
        return None

    def cancel_order(self, order_id: str) -> bool:
        # Placeholder: Implement real OANDA order cancel
        return False

    def get_account_info(self) -> Optional[AccountInfo]:
        """Fetch account info from OANDA REST API."""
        if not self.api_key or not self.account_id:
            return None
        url = f"{self.base_url}accounts/{self.account_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            data = resp.json()["account"]
            return AccountInfo(
                account_id=data["id"],
                cash=float(data.get("balance", 0)),
                equity=float(data.get("NAV", 0)),
                buying_power=float(data.get("marginAvailable", 0)),
                initial_margin=float(data.get("marginUsed", 0)),
                maintenance_margin=float(data.get("marginRate", 0)),
                day_trade_count=int(data.get("openTradeCount", 0)),
                last_updated=datetime.now(),
                broker_name=self.broker_name
            )
        except Exception as e:
            return None
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch latest price for a symbol from OANDA REST API."""
        if not self.api_key:
            return None
        url = f"{self.base_url}instruments/{symbol}/candles"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"count": 1, "granularity": "M1", "price": "M"}
        try:
            resp = requests.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                return None
            candles = resp.json().get("candles", [])
            if not candles:
                return None
            price = float(candles[0]["mid"]["c"])
            return {"symbol": symbol, "price": price, "timestamp": candles[0]["time"]}
        except Exception as e:
            return None

    def get_positions(self) -> List[Position]:
        # Placeholder: Implement real OANDA positions
        return []

    def get_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        # Placeholder: Implement real OANDA orders
        return []
