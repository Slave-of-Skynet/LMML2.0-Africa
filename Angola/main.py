import json
from datetime import datetime
from pathlib import Path

events = json.loads(Path('test-input.json').read_text(encoding='utf-8-sig'))['events']
incident = [e for e in events if e.get('run_id') == 'incident-01']
memory = next(e for e in incident if e.get('event') == 'memory.result' and e.get('injected_into_context'))
item = memory['item']
assert datetime.fromisoformat(item['valid_until']) < datetime.fromisoformat(memory['timestamp'])
assert next(e for e in incident if e.get('event') == 'tool.result')['result']['status'] == 'RESTORED'
assert any(e.get('run_id') == 'replay-no-memory' and e.get('event') == 'runtime.config' and e.get('memory_enabled') is False for e in events)
Path('answer.txt').write_text(item['source_id'] + '\n', encoding='utf-8')
