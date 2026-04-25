.PHONY: preflight test gates local-smoke api-smoke

preflight:
	python scripts/preflight_check.py

test:
	python -m pytest

gates:
	python scripts/gate1_test.py
	python scripts/gate2_test.py
	python scripts/gate3_test.py
	python scripts/gate4_test.py

local-smoke:
	python scripts/test_env_local.py

api-smoke:
	python -c "from fastapi.testclient import TestClient; from api.app import app; c=TestClient(app); r=c.get('/health'); assert r.status_code==200 and r.json().get('status')=='ok'; rr=c.post('/reset', json={'curriculum_stage':1}); assert rr.status_code==200; data=rr.json(); rs=c.post('/step', json={'episode_id':data['episode_id'],'action_type':'send_message','text':'I understand your situation and we can work on a plan together.','metadata':{}}); assert rs.status_code==200; print('api-smoke-ok')"
