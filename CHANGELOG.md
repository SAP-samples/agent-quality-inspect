# 2.3.6a1 (2026-04-15)

### Bug Fixes

* **Token Consumption Metric:** Removed validation that required steps to be present in token consumption metric, allowing metric calculation even when no tool calls are made (#111)

* **Subgoal Validator:** Added filter to remove all invalid judge responses from subgoal validator explanations, ensuring only valid responses matching the expected grading pattern are included (#110)

### Improvements

* **Error Analysis:** Various improvements to error analysis module (#99):
  - Added missing type hints
  - Extracted tool validation explanations into constants
  - Handle invalid counts in trial judges for subgoal error analysis
  - Added predefined cluster `Unclassified` for uncategorized errors
  - Improved error clustering logic
  - Refactored `metrics_utils.py` into `core/utils.py` and `tools/utils.py` for better organization
  - Removed dead/unused methods

# 2.3.4a1 (2026-04-06)

### Bug Fixes

* **User Proxy:** Fixed overly strict termination condition check in `UserProxyAgent` that required the stop sequence to exactly match the entire message string. The check has been loosened from `==` (equality) to `in` (substring containment), allowing the agent to correctly detect stop sequences even when they appear as part of a larger message.


# 2.3.3a1 (2026-03-30)

### Improvements

* **Python Version:** Added support for Python 3.14 while maintaining backward compatibility with Python 3.12 and 3.10
  - openai version upgraded to 2.29.0

* **Documentation:** Removal of Documentation related python packages from main project dependencies and moving them to be in the `docs_requirements.txt` for better separation of concerns and lighter installation for users who don't need to build docs.

### Bug Fixes

* Fixed syntax warnings by adding escape characters in regex patterns
* Fixed missed input types handling in tool schema parsing
* Fixed unit test cases for updated pipeline models


# 2.2.0a1 (2026-03-16)

### Features

* **User Proxy:** Added support for multiple terminating conditions in user proxy implementations under `agent_inspect.user_proxy`. The user proxy can now be configured with either single or multiple terminating conditions to control when the simulated user should exit the conversation.

  **Configuration Options:**
  - `use_expert_agent` (bool): Controls user persona type. Default: `True` (expert persona)
  - `terminating_condition_mode` (str): Specifies the mode - either `"single"` or `"multiple"`. Default: `"single"`

  **Predefined Terminating Condition Classes:**
  User proxy now includes three specialized terminating condition types for multi-condition mode:

  - `TaskCompletedTerminatingCondition`: Terminates when the task goal is satisfied
    - Default stop sequence: `"##DONE##"`
    - Default check: Task is complete when instruction goal is satisfied

  - `TaskDelegatedTerminatingCondition`: Terminates when control is transferred to another agent
    - Default stop sequence: `"##DELEGATE##"`
    - Default check: Task terminates when agent transfers control to another system

  - `TaskBlockedTerminatingCondition`: Terminates when agent lacks required capabilities
    - Default stop sequence: `"##BLOCKED##"`
    - Default check: Task terminates when agent cannot proceed due to missing permissions/tools

  **Key Features:**
  - Automatic validation of terminating condition configurations
  - Support for both expert and non-expert user personas with different prompt templates
  - Comprehensive unit tests and acceptance tests for all terminating condition modes

* **Error Analysis:** Major refactoring and expansion of error analysis tools with new analysis methods and improved architecture:

  **New Error Analysis Classes:**
  - `SemisupervisedToolCallErrorAnalysis`: LLM-based classification of tool call errors into predefined error clusters
  - `DeterministicToolCallErrorAnalysis`: Fast, rule-based tool call error classification using regex patterns (no LLM required)
  - `SemisupervisedSubgoalErrorAnalysis`: Classifies subgoal errors into predefined categories using LLM
  - `UnsupervisedSubgoalErrorAnalysis`: Dynamically discovers error patterns through LLM-based clustering

  **Architecture Improvements:**
  - Introduced abstract base class hierarchy for better code organization:
    - `ErrorAnalysis` (ABC): Base class providing common functionality for all error analysis
    - `ToolCallErrorAnalysis` (ABC): Base for tool call-level error analysis
    - `SubgoalErrorAnalysis` (ABC): Base for subgoal-level error analysis
  - Added `StatisticAnalysis` utility class for computing judge expectations and variance
  - All error analysis classes now support both synchronous (`analyze_batch`) and asynchronous (`analyze_batch_async`) execution

  **Error Cluster Support:**
  - Support for custom user-defined error clusters via `ErrorCluster` dataclass
  - Default predefined error clusters for tool call errors:
    - Incorrect Tool Input
    - Incorrect Tool Output Handling
    - Wrong Tool Selection
  - Default predefined error clusters for subgoal errors:
    - Incorrect Tool Input
    - Incorrect Tool Output Handling
    - Wrong Tool Selection
    - Missed Tool Call
    - Instruction Following Error
    - Incomplete Communication
    - Faithfulness Error
    - Logical Reasoning Error

  **New Features:**
  - Support for `None` as user instruction in subgoal validation
  - Concurrent processing with configurable thread pool executors (default: 20 workers)
  - Retry logic for LLM requests with JSON decode errors
  - Improved error handling with detailed error codes and messages

* **LLM Templates:** Refactored LLM constants into separate template and schema modules:
  - Separated prompt templates into `llm_templates.py`
  - Separated output schemas into `llm_output_schemas.py`
  - Improved maintainability and reusability of LLM prompt templates

### Improvements

* **Error Analysis:** Complete refactoring of error analysis module structure:
  - Split monolithic `error_analysis.py` into focused, specialized classes
  - Removed `Base` prefix from base classes for cleaner naming
  - Moved error analysis unit tests into dedicated folder structure
  - Added comprehensive class diagram documentation in base classes
  - Extracted validation logic to utils with proper error handling
  - Consistent use of `InvalidInputValueError` for input validation errors

* **Testing:** Expanded test coverage for error analysis:
  - Added unit tests for deterministic tool call error analysis
  - Added unit tests for semi-supervised tool call error analysis
  - Added acceptance tests for semi-supervised subgoal error analysis
  - Added unit tests for statistic analysis with edge cases
  - Added unit tests for handling missing error IDs in clustering
  - Added unit tests for async error analysis execution

* **Documentation:** Enhanced documentation across error analysis modules:
  - Added comprehensive docstrings with usage examples
  - Updated module-level documentation
  - Added class hierarchy diagrams in base classes
  - Improved type annotations throughout

* **Python Version:** Upgraded minimum Python version to 3.12

* **Code Quality:** Applied formatting and linting improvements:
  - Applied ruff formatting across all changed files
  - Fixed various code quality issues identified by static analysis
  - Improved type hints and annotations


# 2.1.1a1 (2026-03-06)
### Bug Fixes
- **metrics/validator**: Fixed circular import between `validator` and `scorer` modules by moving the following templates from `scorer/templates.py` to a new `validator/templates.py` file:
  - `TOOL_CORRECTNESS_TEMPLATE` (used by `ToolCallCompletionValidator`)
  - `DEFAULT_MODEL_GRADED_FACT_SINGLE_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL`
  - `DEFAULT_MODEL_GRADED_FACT_MULTI_TURN_AT_CURRENT_TURN_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL`
  - `DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL`
  - `DEFAULT_MODEL_GRADED_FACT_DYNAMIC_SUMMARY_WITHOUT_INSTRUCT_REMOVE_HALLUCINATION_CHECK_TEMPLATE_ONE_SUBGOAL`

These templates are only used by validator classes, so moving them breaks the circular dependency between `scorer.progress` → `validator.subgoal_completion` → `scorer.templates`. This resolves the `ImportError: cannot import name 'ToolCallCompletionValidator' from partially initialized module` error when importing validator classes.

# 2.1.0a1 (2026-02-20)
### Features
* **Metric:** Added support for multi-turn validation for tool call completion under `agent_inspect.metrics.validator.tool_call_completion.ToolCallCompletionValidator` class, `validate_multi_turns` function.

### Improvements
* **Metric:** Renamed `validate` function to `validate_static_last_turn` in `ToolCallCompletionValidator` class for better clarity and distinction from the new `validate_multi_turns` function.
* **Metric:** Updated documentation and docstrings to reflect the new multi-turn validation support and the renaming of the original validate function.
* Added empty turn traces check for both subgoal completion validator and tool call completion validator.
* Added unit tests and acceptance tests for multi-turn validation in tool call completion validator.


# 2.0.0a1 (2026-02-13)

### Features
* Initial release of User Proxy and Unsupervised Error Analysis Tools.
* **User Proxy:** Added user proxy implementations under `agent_inspect.user_proxy` with single terminating condition.
* **User Proxy:** Added user proxy data classes under `agent_inspect.models.user_proxy`.
* **Error Analysis Tools:** Added error analysis tool implementation under `agent_inspect.tools.error_analysis`.
* **Error Analysis Tools:** Added error analysis tool data classes under `agent_inspect.models.tools`.
* **Metric:** Implemented Progress Per Turn (PPT) metric under `agent_inspect.metrics.scorer.ppt`.
* **Metric:** Added support for Subgoal validation without user instructions, in `SubGoalCompletionValidator` class, `validate_dynamic` function.
* **Metric:** Added PassAtK and PassHatK metric calculations under `agent_inspect.metrics.multi_samples`.
* **Metric:** Added unit tests and acceptance tests for new metrics.
* **Metric:** Added subgoal validation support when `List[Step]=[]` (i.e., no tool calls) and only agent_response and agent_input are provided. Implemented in `SubGoalCompletionValidator` class, `validate` function.
* **Metric:** Added subgoal validation support for multi-turn conversation with only current turn information (agent trace, agent input, agent output) provided. Implemented in `SubGoalCompletionValidator` class, `validate` function.
* **Metric:** Added subgoal validation support for multi-turn conversation with both past user-agent chat history and current turn information (agent trace, agent input, agent output) provided. Implemented in `SubGoalCompletionValidator` class, `validate` function.
* **Metric:** Added judge trial optimization for subgoal validation, default to False (no optimization). Implemented in `SubGoalCompletionValidator` class, `validate` and `validate_dynamic` functions via `OPTIMIZE_JUDGE_TRIALS` flag.
* **Metric:** Added retry mechanism in `SubGoalCompletionValidator` class, `validate` and `validate_dynamic` functions if the LLM judge trials cannot reach to conclusion. Retry attempt can be controlled with `MAX_RETRY_JUDGE_TRIALS_DEFAULT` flag.
* **LLM Client.** Added an out-of-the-box Lite LLM client at `agent_inspect.clients.litellm_client.py`.
* **LLM Client.** Added data class for LLM request payloads at `agent_inspect.models.llm_payload` to be consumed by User Proxy.
* **LLM Client.** Added `make_request_with_payload` interface to LLM clients for user-proxy to make request with LLM payload objects.
* **Documentation:** Added comprehensive documentation for all modules, classes, and functions via github pages.
* **Examples:** Added Customer Runner (`runner.py`) containing full example usages of our metric package under `paper_experiments` folder.
* **Examples:** Added Demo ui for viewing error analysis results under `demo` folder.

### Improvements
* Renamed root package from `agentic_ai_eval` to `agent_inspect`.
* Centralised all data models under `agent_inspect.models` with respective subfolders and files.
* Moving all metric data classes to be under `agent_inspect.models.metrics`.
* Moving all user proxy data classes to be under `agent_inspect.models.user_proxy`.
* Moving all tool related data classes to be under `agent_inspect.models.tools`.
* Moving client payload and response data classes into `llm_payload.py` and `llm_response.py` respectively.
* Moving out clients and exceptions files and classes to be under `agent_inspect.clients` and `agent_inspect.exceptions` respectively.
* Adding documentation and docstrings for all exposed modules, classes, and functions.
* Exposing front facing modules, classes, and functions to be in respective `__init__.py` files for better discoverability and usability.
* Added src folder layer (this doesn't affect pip installation).
* **User Proxy:** Added unit tests and acceptance tests for the user proxy.
* **Error Analysis Tools:** Added unit tests and acceptance tests for error analysis tools.

# 1.0.2a1 (2025-12-10)
* Changed the return type of `get_success_score_from_validation_results` to Numerical Score

# 1.0.1a1 (2025-11-12)

### Bug Fixes
* Removal of gen ai hub sdk from the dependencies

### Improvements
* Moving gen ai hub client to be in the acceptance test dependency
* Removing gen ai hub client from unit test


# 1.0.0a1 (2025-11-10)

### Features

* Initial alpha release of the project.
* **Metrics:** subgoal completion validator (MVS)
* **Metrics:** tool call completion validator (MVS)
* **Metrics:** total latency, average latency (MVS)
* **Metrics:** token consumption (input, output, reasoning) (MVS)
* **Metrics:** tool call count (MVS)
* **Exception:** custom exceptions for better error handling (MVS)
* **Metrics:** progress score, progress through turns (out of MVS)
* **Metrics:** success score, success score final turn (out of MVS)
* **Metrics:** auc score (out of MVS)
* **Client:** gen ai hub client for llm as a judge connection (out of MVS)
* **Adapters:** tau2bench adapter, toolsandbox adapter for converting traces to data class (out of MVS)

### Bug Fixes
* Added unit tests
* Added acceptance tests
* Fixed various bugs in metric calculations

### Improvements
* Fixed code structure and organization for better maintainability according to code reviews
