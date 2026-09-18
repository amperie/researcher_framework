import json
import pytest
from tests.test_platform_workflow import CODE, request, run, workflow


def test_build_uses_conversation_and_validates_exact_proposal():
    source = CODE.replace("return []", "return []  # agreed draft")
    engine, model = workflow([{"content": "Created the agreed algorithm", "sourceCode": source}])
    messages = [{"role": "user", "content": "Use only OHLCV and tick data."}]
    result = run(engine, request("build", messages=messages))
    assert model.ainvoke.await_count == 1
    assert json.loads(model.ainvoke.call_args.args[0][1].content)["conversation"] == messages
    assert result.proposal.changes[0].after == source
    assert result.validation.contentHash == result.proposal.contentHash
    assert result.validation.target == "proposal"
    assert result.validation.status == "passed"
    assert result.provenance["steps"] == ["code", "validate"]


@pytest.mark.parametrize("source", [None, "invalid python!"])
def test_build_does_not_claim_invalid_or_missing_source_passed(source):
    engine, _ = workflow([{"content": "Clarification or proposed source", "sourceCode": source}])
    result = run(engine, request("build"))
    if source is None:
        assert result.proposal is None and result.validation is None
    else:
        assert result.validation.status == "failed"
