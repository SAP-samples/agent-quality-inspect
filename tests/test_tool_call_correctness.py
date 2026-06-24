import pytest
from unittest.mock import patch, AsyncMock

from agent_inspect.models.metrics import (
    ToolInputParameter,
    SubGoal,
    EvaluationSample,
    ExpectedToolCall,
    ToolOutput,
    TurnTrace,
    Step,
    AgentResponse,
    AgentDialogueTrace,
    ToolCallValidationResult,
)
from agent_inspect.metrics.constants import INCLUDE_JUDGE_EXPLANATION
from agent_inspect.metrics.scorer import ToolCorrectnessMetric


@pytest.fixture
def mock_agent_trace():
    """AgentDialogueTrace with 3 turns and tool calls matching expectedTools in the sample JSON.

    Turn index 0: Fetch_AR_DATA + 3 calculator calls (one numeric, two textual outputs)
    Turn index 1: No tool calls (agent acknowledges payment block request)
    Turn index 2: Set Payment Block tool call with given arguments
    """
    # Turn 0 steps
    step_fetch = Step(
        id="s-fetch",
        parent_ids=[None],
        tool="Fetch_AR_DATA",
        tool_input_args=[
            ToolInputParameter(name="CompanyCode", value="F001"),
            ToolInputParameter(name="CustomerAccount", value="C0001"),
        ],
        tool_output={
            "TotalAmountCompanyCodeCurrency": 34673013.30,
            "DisputedAmount": 21750089.32,
        },
    )
    step_calc_1 = Step(
        id="s-calc-1",
        parent_ids=[None],
        tool="calculator",
        tool_input_args=[ToolInputParameter(name="expression", value="(21750089.32)")],
        tool_output=21750089.32,
    )
    step_calc_2 = Step(
        id="s-calc-2",
        parent_ids=[None],
        tool="calculator",
        tool_input_args=[ToolInputParameter(name="expression", value="disputed amount text")],
        tool_output="amount to 2,922,924 EUR.",
    )
    step_calc_3 = Step(
        id="s-calc-3",
        parent_ids=[None],
        tool="calculator",
        tool_input_args=[
            ToolInputParameter(name="expression", value="non disputed overdue amount text")
        ],
        tool_output="amount to approximately 12,922,924 EUR.",
    )
    turn0 = TurnTrace(
        id="turn-0",
        agent_input="Can you analyze the customer account C0001 with company code F001 and suggest actions.",
        steps=[step_fetch, step_calc_1, step_calc_2, step_calc_3],
        agent_response=AgentResponse(
            response="Here is the analysis including disputed and overdue amounts."
        ),
    )

    # Turn 1 (no tools yet for payment block request)
    turn1 = TurnTrace(
        id="turn-1",
        agent_input="I would like to set payment block for AccountingDocument 1400019736, AccountingDocumentItem 002, Fiscal Year 2017, CompanyCode F001 with reason code A and with note 'no remarks'.",
        steps=[],
        agent_response=AgentResponse(
            response="Acknowledged. Please confirm approval to proceed with setting the payment block."
        ),
    )

    # Turn 2 tool call to Set Payment Block
    step_set_block = Step(
        id="s-set-block",
        parent_ids=[None],
        tool="Set Payment Block",
        tool_input_args=[
            ToolInputParameter(name="CompanyCode", value="F001"),
            ToolInputParameter(name="FiscalYear", value="2017"),
            ToolInputParameter(name="Code", value="A"),
            ToolInputParameter(name="Note", value="No Remark."),  # expected via judge check
            ToolInputParameter(name="AccountingDocument", value="1400019736"),
            ToolInputParameter(name="AccountingDocumentItem", value="002"),
        ],
        tool_output="Payment block successfully set.",
    )
    turn2 = TurnTrace(
        id="turn-2",
        agent_input="approve: true",
        steps=[step_set_block],
        agent_response=AgentResponse(response="Payment block has been set as requested."),
    )

    return AgentDialogueTrace(turns=[turn0, turn1, turn2])


@pytest.fixture
def expected_tool_calls_all_pass():
    """EvaluationSample whose expected tool calls align exactly with the trace and should all pass."""
    expected_tool_calls = [
        ExpectedToolCall(
            tool="Fetch_AR_DATA",
            expected_parameters=[
                ToolInputParameter(name="CompanyCode", value="F001"),
                ToolInputParameter(name="CustomerAccount", value="C0001"),
            ],
            expected_output=ToolOutput(
                value={
                    "TotalAmountCompanyCodeCurrency": 34673013.30,
                    "DisputedAmount": 21750089.32,
                }
            ),
            turn=0,
        ),
        ExpectedToolCall(
            tool="calculator",
            expected_output=ToolOutput(value=21750089.32),
            turn=0,
        ),
        ExpectedToolCall(
            tool="calculator",
            expected_parameters=[
                ToolInputParameter(name="expression", value="disputed amount text"),
            ],
            expected_output=ToolOutput(check="amount to 2,922,924 EUR."),
            turn=0,
        ),
        ExpectedToolCall(
            tool="calculator",
            expected_output=ToolOutput(check="amount to approximately 12,922,924 EUR."),
            turn=0,
        ),
        ExpectedToolCall(
            tool="Set Payment Block",
            expected_parameters=[
                ToolInputParameter(name="CompanyCode", value="F001"),
                ToolInputParameter(name="FiscalYear", value="2017"),
                ToolInputParameter(name="Code", value="A"),
                ToolInputParameter(name="Note", check="No Remark."),
                ToolInputParameter(name="AccountingDocument", value="1400019736"),
                ToolInputParameter(name="AccountingDocumentItem", value="002"),
            ],
            expected_output=ToolOutput(value="Payment block successfully set."),
            turn=2,
        ),
    ]
    # Subgoals (re-using JSON grading notes; only needed to satisfy dataclass requirements if other metrics used)
    sub_goals = [SubGoal(details="dummy", turn=0)]
    return EvaluationSample(sub_goals=sub_goals, expected_tool_calls=expected_tool_calls)


###############################
# Tests for ToolCorrectnessMetric
###############################


def test_tool_correctness_metric_all_pass(mock_agent_trace, expected_tool_calls_all_pass):
    mock_llm_client = AsyncMock()

    tool_correctness_metric = ToolCorrectnessMetric(
        llm_client=mock_llm_client,
        config={INCLUDE_JUDGE_EXPLANATION: True},
    )
    with patch(
        "agent_inspect.metrics.scorer.tool_correctness.ToolCallCompletionValidator"
    ) as ToolMockValidator:
        mock_validator_instance = ToolMockValidator.return_value
        side_effects = []
        for expected_tool_call in expected_tool_calls_all_pass.expected_tool_calls:
            side_effects.append(
                ToolCallValidationResult(
                    is_completed=True,
                    expected_tool_call=expected_tool_call,
                    explanations=["Tool call validated successfully."],
                )
            )

        mock_validator_instance.validate_static_last_turn = AsyncMock(side_effect=side_effects)
        tool_correctness_score = tool_correctness_metric.evaluate(
            agent_trace=mock_agent_trace,
            evaluation_data_sample=expected_tool_calls_all_pass,
        )

        assert tool_correctness_score.score == 1.0, "All expected tool calls should pass"
        assert len(tool_correctness_score.explanations) == 5
        assert mock_validator_instance.validate_static_last_turn.await_count == 5


def test_tool_correctness_metric_partial(mock_agent_trace, expected_tool_calls_all_pass):
    mock_llm_client = AsyncMock()

    tool_correctness_metric = ToolCorrectnessMetric(
        llm_client=mock_llm_client,
        config={INCLUDE_JUDGE_EXPLANATION: True},
    )
    with patch(
        "agent_inspect.metrics.scorer.tool_correctness.ToolCallCompletionValidator"
    ) as ToolMockValidator:
        mock_validator_instance = ToolMockValidator.return_value
        side_effects = []
        # First four tool calls pass
        for expected_tool_call in expected_tool_calls_all_pass.expected_tool_calls[:-1]:
            side_effects.append(
                ToolCallValidationResult(
                    is_completed=True,
                    expected_tool_call=expected_tool_call,
                    explanations=["Tool call validated successfully."],
                )
            )
        # Last tool call fails
        side_effects.append(
            ToolCallValidationResult(
                is_completed=False,
                expected_tool_call=expected_tool_calls_all_pass.expected_tool_calls[-1],
                explanations=["Tool call validation failed."],
            )
        )

        mock_validator_instance.validate_static_last_turn = AsyncMock(side_effect=side_effects)
        tool_correctness_score = tool_correctness_metric.evaluate(
            agent_trace=mock_agent_trace,
            evaluation_data_sample=expected_tool_calls_all_pass,
        )
        assert tool_correctness_score.score == 0.8, (
            "Four out of five expected tool calls should pass"
        )
        assert len(tool_correctness_score.explanations) == 5
        assert mock_validator_instance.validate_static_last_turn.await_count == 5
