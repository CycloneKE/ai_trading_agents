
import json
import logging
from broker_manager import BrokerManager

# Enable debug logging to console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')


# Load config
with open('config/config.json', 'r') as f:
    config = json.load(f)

print('Loaded config brokers section:', json.dumps(config.get('brokers', {}), indent=2))
# Pass the full config to BrokerManager
bm = BrokerManager(config)


print('Available brokers:', list(bm.brokers.keys()))
for name, broker in bm.brokers.items():
    print(f'Connecting broker: {name}...')
    connected = broker.connect() if hasattr(broker, 'connect') else False
    print(f'Broker: {name}, Type: {broker.broker_name}, Connected: {connected}')
    print('Status:', broker.get_status())
