# Streamlit UI for viewing Agent Diagnosis
This directory contains a user interface (UI) for viewing error analysis produced during diagnosis of LLM agents. The UI is built using Streamlit.

## How to run
1. In the project root directory, install our evaluation package:
   ```bash
   pip install -e .
   ```
2. Then, install the required dependencies for the UI:
   ```bash
   pip install -r demo/ui_for_agent_diagnosis/requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   python -m streamlit run demo/ui_for_agent_diagnosis/app.py
   ```
4. Open your web browser and navigate to `http://localhost:8501` to view.

Please note that when you are running our UI with above command, it is loading the data from the default path `demo/ui_for_agent_diagnosis/sample_data/`. If you want to load your own diagnosis data, you can modify the `PATH_TO_ERROR_SUMMARY_CSV` and `example_path` variable in `app.py` to point to your desired data files.