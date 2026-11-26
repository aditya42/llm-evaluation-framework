import pytest
from llm_eval_framework.qe_rubric import score, compare

R = dict(id="a", category="debugging", difficulty="easy", risk="LOW", failure_category="timing", rubric=dict(criteria=[dict(name="a",weight=.75),dict(name="b",weight=.25)],passing_score=3))
GOOD = [dict(id="a",scores=dict(a=4,b=0))]
BAD = [dict(id="a",scores=dict(a=2,b=0))]

def test_weighted():
    assert score(R, GOOD[0]["scores"]) == 3

def test_regression():
    assert compare([R], GOOD, GOOD)["passed"]
    assert not compare([R], GOOD, BAD)["passed"]

@pytest.mark.parametrize("scores",[{"a":4},{"a":5,"b":0},{"a":float("nan"),"b":0}])
def test_invalid(scores):
    with pytest.raises(ValueError):
        score(R,scores)

def test_missing():
    with pytest.raises(ValueError):
        compare([R],GOOD,[])
