from http import HTTPStatus


COMPLETE_INCOMPLETE_GRADE_PATTERN = r"(?i)GRADE\s*:\s*([CPI])(.*)$"
SUBGOAL_JSON_ARRAY_EXTRACTION_PATTERN = r"\[[\s\S]*\]"
COMPLETE_INCOMPLETE_PAIR = ["C", "I"]
TEMPLATE_SUBGOAL = "template_subgoal"
MAX_TURNS = "max_turns"
K_VALUE = "k_value"
NO_OF_TRIALS = "no_of_trials"
NUM_JUDGE_TRIALS = "num_judge_trials"
MAX_RETRY_JSON_DECODE_ERROR = 5
MAX_RETRY_JUDGE_TRIALS = "max_retry_judge_trials"
INCLUDE_JUDGE_EXPLANATION = "include_judge_explanation"
INCLUDE_VALIDATION_RESULTS = "include_validation_results"
INCLUDE_PROMPT_SENT_TO_LLMJ = "include_prompt_sent_to_llmj"
OPTIMIZE_JUDGE_TRIALS = "optimize_judge_trials"

OPTIMIZE_JUDGE_TRIALS_DEFAULT = False
MAX_TURNS_DEFAULT = 20

MAX_RETRY_JUDGE_TRIALS_DEFAULT = 5
NUM_JUDGE_TRIALS_DEFAULT = 5

AGENT_INPUT = "Agent Input"
AGENT_OUTPUT = "Agent Output"
AGENT_THOUGHT = "Agent Thought"
TOOL_CALL = "Tool Call"

STATUS_200 = HTTPStatus.OK
STATUS_429 = HTTPStatus.TOO_MANY_REQUESTS
STATUS_500 = HTTPStatus.INTERNAL_SERVER_ERROR
STATUS_404 = HTTPStatus.NOT_FOUND

MAX_RETRY_ATTEMPTS_EXCEEDED = "Maximum retry attempts exceeded."
COULD_NOT_REACH_MAJORITY_DECISION = (
    "Could not reach majority decision due to insufficient valid judge responses."
)

INPUT_TOKEN_CONSUMPTION = "input_token_consumption"
OUTPUT_TOKEN_CONSUMPTION = "output_token_consumption"
REASONING_TOKEN_CONSUMPTION = "reasoning_token_consumption"

MAX_WORKERS_KEY = "max_workers"
DEFAULT_MAX_WORKERS = 20

NUM_SUBGOALS_PER_CATEGORY = "num_subgoals_per_category"
DEFAULT_NUM_SUBGOALS_PER_CATEGORY = 3

DETAILS = "details"
