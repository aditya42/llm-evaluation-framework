import pytest
from llm_eval_framework.live_benchmark import summarize

def test_cost():
    rows=[dict(input_tokens=1000,output_tokens=500,latency_ms=200)]
    result=summarize(rows,2,4)
    assert result["estimated_generation_cost_usd"]==pytest.approx(.004)
    assert result["mean_latency_ms"]==200

def test_invalid_price():
    with pytest.raises(ValueError):
        summarize([{}],-1,0)
