# Paper Experiments Runner

This directory contains scripts for running agent evaluation experiments with user proxy interactions. The runner evaluates agents across multiple samples and trials, calculating metrics like AUC and PPT, and performing error analysis.

## Prerequisites

1. **Agent Setup** (separate from this repository): [TODO: Provide links to agent repositories]
   - For Tau2Bench agent: Clone the tau2bench repository and set it up by following the instructions in its README
   - For ToolSandbox agent: Clone the toolsandbox repository and set it up by following the instructions in its README

2. **Azure OpenAI Configuration**:
   - Create a `.env` file in the **project root directory** with the following variables:
     ```bash
     AZURE_API_VERSION=your_api_version      # e.g., 2024-02-15-preview
     AZURE_API_BASE=your_azure_endpoint      # e.g., https://your-resource.openai.azure.com/
     AZURE_API_KEY=your_api_key              # Your Azure OpenAI API key
     ```
   - The runner uses Azure OpenAI for:
     - User proxy agent interactions (`--user-proxy-model`)
     - LLM-based metric evaluation (Progress, AUC, PPT)
     - Error analysis

3. **Environment Setup**:
   - Install our evaluation package from the project root:
     ```bash
     pip install -e .
     ```

## Running Evaluations

### Basic Usage

Run the evaluation from the **project root directory**:

```bash
python -m paper_experiments.runner
```

This will use default settings (tau2bench agent with sample dataset).

### Command-Line Arguments

Customize your evaluation with these arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--agent` | Agent type to use (`tau2bench` or `toolsandbox`) | `tau2bench` |
| `--agent-model` | Agent model to use (e.g., `azure/gpt-4.1`) | `azure/gpt-4.1` |
| `--samples-file` | Path to samples JSON file (from project root) | `paper_experiments/datasets/sample.json` |
| `--max-turns` | Maximum conversation turns | `15` (tau2bench), `8` (toolsandbox) |
| `--n-trials` | Number of trials per sample | `20` (tau2bench), `8` (toolsandbox) |
| `--max-workers` | Number of parallel samples to run | `3` (tau2bench), `1` (toolsandbox) |
| `--user-proxy-model` | User proxy model | `gpt-4.1` |
| `--user-proxy-persona` | User proxy persona (`expert` or `non-expert`) | `expert` |
| `--output-dir` | Output directory name (created in `paper_experiments/`) | `experiment_outputs` |

### Example Commands

**To reproduce the results from our paper, run the following commands:**

**Tau2Bench:**
```bash
python -m paper_experiments.runner \
  --agent tau2bench \
  --samples-file <path_to_dataset> \
  --max-turns 15 \
  --n-trials 20 \
  --user-proxy-persona <persona>
```

**ToolSandbox:**
```bash
python -m paper_experiments.runner \
  --agent toolsandbox \
  --samples-file <path_to_dataset> \
  --max-turns 8 \
  --n-trials 8 \
  --max-workers 1 \
  --user-proxy-persona <persona>
```

**Valid options for `<path_to_dataset>` and `<persona>`:**

- **Tau2Bench datasets:**
  - `paper_experiments/datasets/tau2bench_dataset_easy.json`
  - `paper_experiments/datasets/tau2bench_dataset_hard.json`
  - `paper_experiments/datasets/tau2bench_dataset.json` (full)

- **ToolSandbox dataset:**
  - `paper_experiments/datasets/toolsandbox_dataset.json`

- **User proxy personas:**
  - `expert`
  - `non-expert`

Run all combinations of datasets and personas to fully reproduce the paper results (6 runs for Tau2Bench, 2 runs for ToolSandbox).

> **Note:** `max-workers` for ToolSandbox should be kept to 1 because ToolSandbox's results might be inaccurate when run in parallel.

## Output Files

Results are saved in `paper_experiments/experiment_outputs_<timestamp>/` with the following structure:

### Generated Files

1. **`trial_<N>_results.json`** - Results for each trial, containing:
   - Metadata (agent model, max turns, user proxy model, etc.)
   - Per-sample results with trajectories, metrics (AUC, PPT), and turn counts
   - Success/failure status for each run

2. **`aggregate_metrics_results.json`** - Aggregate metrics (MaxAUC@k, MaxPPT@k) across all trials

3. **`evaluation_results.pkl`** - Serialized evaluation results (Python pickle format)

4. **`error_analysis.pkl`** - Error analysis results (Python pickle format)

5. **`evaluation.log`** - Detailed execution logs

## Error Analysis UI

View detailed error analysis results in an interactive Streamlit UI:

```bash
python -m streamlit run paper_experiments/view_results.py -- --output-dir paper_experiments/experiment_outputs_<timestamp>
```

Replace `<timestamp>` with the actual timestamp from your output directory (e.g., `experiment_outputs_20260116_143022`).

### Example

If your results are in `paper_experiments/experiment_outputs_20260116_143022/`:

```bash
python -m streamlit run paper_experiments/view_results.py -- --output-dir paper_experiments/experiment_outputs_20260116_143022
```

The Streamlit app will launch in your browser at `http://localhost:8501`.