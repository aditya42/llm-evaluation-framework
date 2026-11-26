import pytest
from llm_eval_framework.llm_judge import validate

R={"id":"a","rubric":{"criteria":[{"name":"quality","weight":1}],"passing_score":3}}
BASE={"scores":{"quality":3},"rationale":{"quality":"Adequate"}}
def test_grounded():
    result=validate(R,{**BASE,"claims":[{"claim":"API failed","verdict":"supported","evidence":"503"}]},"API returned 503")
    assert result["groundedness"]==1
def test_no_context():
    assert validate(R,{**BASE,"claims":[]},"")["groundedness"] is None
def test_invented_evidence():
    with pytest.raises(ValueError):
        validate(R,{**BASE,"claims":[{"claim":"API failed","verdict":"supported","evidence":"500"}]},"503")
def test_unsupported():
    assert validate(R,{**BASE,"claims":[{"claim":"Database failed","verdict":"unsupported","evidence":""}]},"503")["unsupported_claim_rate"]==1
