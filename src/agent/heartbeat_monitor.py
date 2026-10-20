"""Dead-Man's Switch Heartbeat Monitor."""
import time
import threading
import logging

logger = logging.getLogger(__name__)

class HeartbeatMonitor:
    def __init__(self, risk_manager, broker_manager, audit_journal, timeout_seconds=180):
        self.risk_manager = risk_manager
        self.broker_manager = broker_manager
        self.audit_journal = audit_journal
        self._timeout = timeout_seconds
        self.last_ping = time.time()
        self._triggered = False
        self._daemon_thread = None

    def ping(self):
        self.last_ping = time.time()

    def elapsed(self) -> float:
        return time.time() - self.last_ping

    def is_healthy(self) -> bool:
        return self.elapsed() < self._timeout

    def status(self) -> dict:
        return {
            'healthy': self.is_healthy(),
            'last_ping': self.last_ping,
            'elapsed_seconds': round(self.elapsed(), 1),
            'timeout_seconds': self._timeout,
            'triggered': self._triggered
        }

    def start_watchdog(self):
        self._daemon_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._daemon_thread.start()

    def _check_timeout(self):
        if self.elapsed() > self._timeout and not self._triggered:
            self._triggered = True
            logger.critical("HEARTBEAT_TIMEOUT: Agent loop unresponsive for >{}s".format(self._timeout))
            if hasattr(self.risk_manager, 'set_persistent_kill_switch'):
                self.risk_manager.set_persistent_kill_switch(True, "HEARTBEAT_TIMEOUT: Agent loop unresponsive for >{}s".format(self._timeout))
            
            try:
                if hasattr(self.broker_manager, 'get_broker'):
                    broker = self.broker_manager.get_broker()
                    if hasattr(broker, 'get_orders'):
                        orders = broker.get_orders()
                        for order in orders:
                            try:
                                broker.cancel_order(order.id)
                            except Exception as e:
                                logger.error(f"Failed to cancel order {order.id}: {e}")
            except Exception as e:
                logger.error(f"Failed to cancel orders during heartbeat timeout: {e}")

            if hasattr(self.audit_journal, 'log_event'):
                self.audit_journal.log_event(
                    'system', 
                    'heartbeat_timeout', 
                    {'elapsed': self.elapsed(), 'timeout': self._timeout}, 
                    {'action': 'kill_switch_activated', 'orders_cancelled': True}
                )

    def _watchdog_loop(self):
        while True:
            self._check_timeout()
            time.sleep(5)
