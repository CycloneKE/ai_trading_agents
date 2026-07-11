from src.agent.sleeve.llm_veto import check_candidate


class FakeOrchestrator:
    def __init__(self, response, enabled=True):
        self.enabled = enabled
        self._response = response

    def propose_json(self, system_prompt, user_prompt, model_override=None):
        return self._response


def test_flag_true_passed_through():
    orch = FakeOrchestrator({'flag': True, 'reason': 'dividend cut reported'})
    result = check_candidate(orch, 'SCOM')
    assert result.flag is True
    assert result.reason == 'dividend cut reported'
    assert result.available is True


def test_flag_false_passed_through():
    orch = FakeOrchestrator({'flag': False, 'reason': ''})
    result = check_candidate(orch, 'SCOM')
    assert result.flag is False
    assert result.available is True


def test_disabled_orchestrator_marks_unavailable_never_flags():
    orch = FakeOrchestrator(None, enabled=False)
    result = check_candidate(orch, 'SCOM')
    assert result.available is False
    assert result.flag is False


def test_none_orchestrator_marks_unavailable():
    result = check_candidate(None, 'SCOM')
    assert result.available is False
    assert result.flag is False


def test_malformed_response_marks_unavailable_never_flags():
    orch = FakeOrchestrator({'unexpected': 'shape'})
    result = check_candidate(orch, 'SCOM')
    assert result.available is False
    assert result.flag is False


def test_non_dict_response_marks_unavailable():
    orch = FakeOrchestrator(None, enabled=True)
    result = check_candidate(orch, 'SCOM')
    assert result.available is False
