from app.agents.policy import PolicyAgent


def test_policy_threshold_mapping_low_score_approves():
    agent = PolicyAgent()
    record = {"TransactionAmt": 10.0}
    decision = agent.decide(record, probability=0.01, risk_score=1)
    assert decision == "APPROVE"


def test_policy_high_amount_and_score_blocks():
    agent = PolicyAgent()
    record = {"TransactionAmt": 6000.0}
    # High score representative of high probability
    decision = agent.decide(record, probability=0.9, risk_score=90)
    assert decision == "BLOCK"

