from llm_eval_framework.agent_benchmark import evaluate

CASES = [{"id":"a","expected":{"tool":"get_test_coverage","arguments":{"release_id":"r"},"outcome":"ready_for_review"}}]
GOOD = {"id":"a", **CASES[0]["expected"], "observed_tools":["get_test_coverage"]}

def test_correct():
    assert evaluate(CASES,[GOOD])["passed"]

def test_missing():
    assert not evaluate(CASES,[])["passed"]

def test_wrong_arguments():
    assert not evaluate(CASES,[{**GOOD,"arguments":{}}])["passed"]

def test_unauthorized():
    assert not evaluate(CASES,[{**GOOD,"observed_tools":["deploy"]}])["passed"]
