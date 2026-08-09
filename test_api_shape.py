import urllib.request, json

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/verify',
    data=json.dumps({'question': 'Who invented the telephone?'}).encode(),
    headers={'Content-Type': 'application/json'}
)
resp = urllib.request.urlopen(req, timeout=180)
data = json.loads(resp.read())
r = data['result']
cv = r['claim_verifications'][0] if r['claim_verifications'] else None

print('HTTP:', resp.status)
print('label:', r['label'])
print('confidence_pct:', r['confidence_pct'])
print('claims count:', len(r['claims']))
print('claim_verifications count:', len(r['claim_verifications']))

if cv:
    print('cv[0].claim type:', type(cv['claim']).__name__)
    print('cv[0].claim.text:', cv['claim']['text'][:80])
    print('cv[0].claim.is_negated:', cv['claim']['is_negated'])
    print('cv[0].claim.is_relevant_to_question:', cv['claim']['is_relevant_to_question'])
    print('cv[0].verdict:', cv['verdict'])
    print('cv[0].best_nli_entailment:', cv['best_nli_entailment'])
    print('cv[0].best_relevance_score:', cv['best_relevance_score'])
    print('cv[0].supporting_count:', cv['supporting_count'])
    
print('evidence count:', len(r['evidence']))
print('logic_trace count:', len(r['logic_trace']))
print('entity_drift_detected:', r['entity_drift_detected'])

# Confirm every value is JSON-serializable (no Python objects remain)
json.dumps(data)
print('\nFull JSON serialization: OK')
