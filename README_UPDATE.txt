CREDIT CARD FRAUD DETECTION — SAFE UPDATE
==========================================

FILES IN THIS PATCH
-------------------
1. main.py          Complete replacement Streamlit application.
2. requirements.txt Dependency list. scikit-learn is pinned to 1.9.1 because
                     the supplied Isolation Forest pickle was saved with 1.9.1.

BEFORE REPLACING ANYTHING
-------------------------
1. Stop the currently running Streamlit app (Ctrl+C in its terminal).
2. Make a backup of your existing project-root main.py, for example:
      copy main.py main_before_update_backup.py
3. Keep your existing models, data, outputs, src and .streamlit folders.
4. Replace ONLY the project-root main.py with the main.py from this patch.
5. Replace requirements.txt with the requirements.txt from this patch.

REQUIRED EXISTING PROJECT FILES
-------------------------------
models/isolation_forest_improved.pkl
outputs/improved_predictions.csv

Do not delete or rename these files. The new app uses your existing trained
model and the existing held-out test predictions. It does not retrain the model.

INSTALL / RUN (Windows PowerShell)
----------------------------------
Activate your existing environment (your screenshot shows a folder named venv):

    .\venv\Scripts\Activate.ps1

Then install dependencies:

    python -m pip install -r requirements.txt

Run the application from the project root:

    python -m streamlit run main.py

If PowerShell blocks activation, you can run Streamlit using the environment's
Python directly:

    .\venv\Scripts\python.exe -m pip install -r requirements.txt
    .\venv\Scripts\python.exe -m streamlit run main.py

IMPORTANT MODEL CORRECTION
--------------------------
Your training script selects the threshold using:
    anomaly_score = -model.decision_function(X)
    fraud = anomaly_score >= threshold

The old main.py compared the threshold to decision_function(X) directly.
The replacement fixes that sign mismatch. Higher anomaly scores now mean
more suspicious, matching src/improve_model.py and outputs/improved_predictions.csv.

PERFORMANCE CHANGES
-------------------
- Model predictions are calculated when the user presses Run Fraud Detection,
  not again when navigating between pages.
- Model Performance defaults to outputs/improved_predictions.csv, so it does
  not score the full 284,807-row dataset just to open the page.
- Existing analysis scores are reused when evaluating an already-analysed
  labelled dataset.
- Large transaction tables are limited to a 100-row preview; CSV downloads
  still contain the complete results.
- CSV download payloads are serialized once per session.

MODES
-----
Demo Mode:
  Generated sample transactions, illustrative only.

Dataset Analysis Mode:
  Uses the uploaded CSV, calculates predictions once, then displays charts,
  suspicious transactions and a complete CSV download.

Manual Transaction Mode:
  Single or batch transaction entry, quick entry using uploaded-data medians,
  full entry, history, charts and downloads.

Model Performance:
  Confusion matrix, precision, recall, F1, ROC curve, precision-recall curve,
  score distribution and model information.

NOTE
----
The saved test prediction file currently contains 56,746 held-out records.
Its metrics are calculated from that file, not invented or tuned in the UI.
The expected values from the supplied file are approximately:
Precision 14.12%, Recall 38.95%, F1 20.73%, ROC-AUC 95.12%.
