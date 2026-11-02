"""Prompt templates for the LLM agents (Analyst, Researcher, Developer, Debugger, Metric coder).

Design rules for a mid-size model (Qwen3.x-27B):
  * one role per prompt, explicit contract, explicit output schema, short and concrete rules;
  * all facts the model needs are inside the prompt (no tool calls);
  * outputs are validated by code afterwards, so prompts ask for structure, not for trust.
"""
from __future__ import annotations

import json

import mla_metrics as MT
from mla_utils import truncate

# =============================================================================================
# Analyst
# =============================================================================================
ANALYST_SYSTEM = (
    "You are a principal data scientist and Kaggle grandmaster. You read the specification of a supervised "
    "machine-learning task together with a profile of its data files, and you return a precise, machine-readable task "
    "definition plus expert domain insight. Every column name you output must exist exactly as written in the provided "
    "data. You answer with a single JSON object and nothing else."
)


def analyst_user(task_text: str, files_text: str, sample_head: str, ddict_text: str, profile_text: str,
                 target_text: str, heuristic: dict) -> str:
    metric_list = "\n".join(f"- {k}: {m.title} ({'higher' if m.greater_is_better else 'lower'} is better)"
                            for k, m in MT.REGISTRY.items())
    return f"""## Task description (task.md)
<<<
{truncate(task_text, 14000, tail=2000)}
>>>

## Files
{files_text}

sample_submission.csv (first rows):
{sample_head}

## Data dictionary
{ddict_text or '(not provided)'}

## Column profile of train.csv
Format: name [kind] miss=train%/test% nuniq=... range/top values — description
{profile_text}

## Target candidates
{target_text}

## Preliminary automatic analysis (verify it; correct it if wrong)
{json.dumps(heuristic, ensure_ascii=False, indent=1)}

## Supported metric keys
{metric_list}

## Return exactly one JSON object with these keys
{{
  "target_columns": ["<target column(s) in train.csv>"],
  "id_column": "<row identifier column shared by test.csv and sample_submission.csv, or null>",
  "task_type": "regression | binary | multiclass",
  "metric": "<one supported metric key, or 'custom'>",
  "metric_reasoning": "<quote the sentence of the task that defines the PRIMARY metric>",
  "positive_label": "<binary only: the class value that is the 'event' / whose probability is requested; else null>",
  "prediction_format": "value | label | probability | class_probabilities",
  "cv_strategy": {{"type": "random | group | time", "group_column": null, "time_column": null, "reasoning": "<why>"}},
  "categorical_columns": ["<numeric-coded columns that are really nominal categories>"],
  "drop_columns": [{{"column": "<name>", "reason": "<pure identifier / leakage / garbage>"}}],
  "domain_summary": "<6-10 sentences: what physically or behaviourally drives the target, which features should matter most and why, data quirks (sentinel values, units, duplicated entities, skew), pitfalls>",
  "feature_ideas": ["<8-15 concrete ideas, each naming the exact columns and the mechanism: ratio, difference, aggregate over a column family, group statistic, entity key + count/target encoding, domain formula>"],
  "potential_leaks": "<columns that may leak the target or 'none'>",
  "custom_metric_code": null
}}

Rules:
- The PRIMARY metric decides. Metrics described as reference / informational / secondary must be ignored.
- "group" CV only if test rows belong to entities (customers, buildings, devices...) that never appear in train; "time" only if the test period is strictly later than train; otherwise "random".
- drop_columns: only certain cases (row ids, constant columns, free-text garbage). When unsure, leave it empty.
- For metric 'custom' write python code `def metric(y_true, y_pred) -> float` in custom_metric_code.
- Output the JSON object only."""


# =============================================================================================
# Researcher
# =============================================================================================
RESEARCHER_SYSTEM = (
    "You are the research lead of an autonomous machine-learning team competing on a tabular prediction task. "
    "You design the next feature-engineering experiments. Every experiment is evaluated with the same LightGBM model and "
    "the same K-fold cross-validation; it is accepted only if the metric improves consistently across folds. "
    "You reason like a Kaggle grandmaster: hypotheses grounded in the mechanism that generates the target and in the "
    "evidence of previous experiments. You answer with a single JSON object and nothing else."
)

EXPERIMENT_TYPES = """1. "feature_code" — new columns computed by a pandas function from the CURRENT feature columns (train and test rows together; the target is not available). Use it for: ratios/rates/differences/products of semantically related quantities; aggregates over families of similar columns (mean/min/max/std/range/slope, e.g. over months); deviation of a value from its group mean (groupby(...).transform); entity/identity keys (concatenate columns that jointly identify the same real-world object) and their frequency; counts of rows sharing a value; domain formulas; parsing of composite strings; turning sentinel values (0, -1, -999 meaning 'unknown') into NaN.
   Fields: "spec" = precise description of every new column with its exact formula and exact source column names.
2. "target_encoding" — leak-free out-of-fold smoothed target statistics for a categorical column or a COMBINATION of columns (computed by the system inside each fold). Ideal for entity keys and interactions of strong categoricals.
   Fields: "columns" = list of column lists, e.g. [["facility_type"], ["state", "building_class"]]. Columns must exist (create a key with feature_code first if needed).
3. "drop_features" — remove noisy, redundant or identifier-like columns. Fields: "columns" = [...].
4. "categorical_cast" — treat numeric-coded columns as categorical. Fields: "columns" = [...]."""

RESEARCH_GUIDANCE = """- LightGBM already handles monotonic transforms (log/sqrt/scaling/standardization), one-hot encoding, simple thresholds and missing values: never propose those.
- Most valuable for gradient boosting: ratios and differences of related quantities; aggregates across families of columns; group statistics (value minus group mean, group count, group std); entity keys and their counts/target encodings; interactions of the strongest categoricals; domain formulas.
- Look at the error analysis: target the classes / segments with the largest errors.
- Use the journal: never repeat an experiment; when an idea was rejected, try a different mechanism; when it was accepted, extend it (other columns, other statistics).
- Keep each experiment focused (typically 1-12 new columns) so its effect is measurable. Prefer diverse mechanisms within one batch.
- Use exact column names from the current feature set."""


def researcher_user(task_block: str, ddict_text: str, features_text: str, model_block: str, shift_text: str,
                    journal_text: str, insights: list, ideas: list, k: int) -> str:
    ins = "\n".join(f"- {i}" for i in insights) if insights else "- (none yet)"
    ide = "\n".join(f"- {i}" for i in ideas) if ideas else "- (none)"
    return f"""## Task
{task_block}

## Data dictionary (original columns)
{ddict_text or '(not provided)'}

## Current feature set
{features_text}

## Current best model
{model_block}

## Train/test shift
{shift_text}

## Experiment journal (oldest first)
{journal_text}

## Lessons learned so far
{ins}

## Untried ideas from the initial analysis
{ide}

## Experiment types
{EXPERIMENT_TYPES}

## Guidance
{RESEARCH_GUIDANCE}

## Output
Return ONE JSON object:
{{
  "analysis": "<3-6 sentences: what the evidence says and which bottleneck to attack next>",
  "insights": ["<updated list of at most 8 durable lessons about this dataset>"],
  "experiments": [
    {{"type": "feature_code", "title": "<short>", "rationale": "<mechanism: why it should help THIS target>", "spec": "<exact columns and formulas>", "expected_gain": "low|medium|high"}},
    {{"type": "target_encoding", "title": "<short>", "rationale": "<...>", "columns": [["<col>"], ["<col_a>", "<col_b>"]], "expected_gain": "low|medium|high"}}
  ]
}}
Propose exactly {k} diverse experiments, the most promising first."""


# =============================================================================================
# Developer / Debugger
# =============================================================================================
DEVELOPER_SYSTEM = (
    "You are a senior Python engineer specialised in pandas feature engineering for machine learning. "
    "You write one correct, efficient, fully vectorised function that follows the given contract exactly. "
    "You answer with a single ```python code block and nothing else."
)

CONTRACT = """```python
import numpy as np
import pandas as pd

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ... create the new columns ...
    return df
```
Rules:
- `df` holds ALL rows (train and test stacked, {n_rows} rows) and ONLY the feature columns listed below. The target is NOT present; never reference it.
- Return the whole dataframe with exactly the same rows in the same order (no filtering, sorting, dropping duplicates, merging that changes row count, or index changes). Add new columns; modify or drop existing columns only if the specification explicitly says so.
- Categorical columns are object (string) columns with NaN for missing; numeric columns are float with NaN for missing. Use `.astype(str)` before string concatenation, and keep NaN handling explicit.
- Only numpy and pandas (math, re, itertools, scipy, sklearn allowed if truly needed). No file/network/system access, no printing.
- Vectorise: groupby(...).transform(...), Series.map, np.where, arithmetic on whole columns. Never loop over rows. For group statistics use `df.groupby(keys)[col].transform('mean')` (pass `dropna=False` to groupby if keys have NaN).
- Numerical safety: divide with `b.replace(0, np.nan)`; results must not contain inf; cast new numeric columns to float.
- A new column meant as a categorical key must be a string column (e.g. `df['k'] = df['a'].astype(str) + '_' + df['b'].astype(str)`).
- New column names: short, ASCII snake_case, not clashing with existing names.
- Never derive features from row position or file order (np.arange, cumcount over the raw order, shift/rolling/diff over rows): train and test are simply stacked, so such features are meaningless or leak. Ordering-based features are allowed only after explicitly sorting by a real time/sequence column and grouping by an entity key, and the result must be aligned back to the original row order (assign via the index).
- Must run in well under 60 seconds on {n_rows} rows."""


def developer_user(exp: dict, columns_table: str, n_rows: int, n_cols: int, task_line: str) -> str:
    return f"""## Experiment to implement
Task context: {task_line}
Title: {exp.get('title', '')}
Rationale: {exp.get('rationale', '')}
Specification: {exp.get('spec', '') or exp.get('title', '')}

## Contract
{CONTRACT.format(n_rows=n_rows)}

## Available columns ({n_cols}) — name | kind | values | description
{columns_table}

Write the complete code now (one ```python block)."""


def debugger_user(exp: dict, code: str, error: str, columns_table: str, n_rows: int) -> str:
    return f"""The feature-engineering function below failed. Fix it.

## Experiment
Title: {exp.get('title', '')}
Specification: {exp.get('spec', '') or exp.get('title', '')}

## Code that failed
```python
{truncate(code, 8000)}
```

## Error
{truncate(error, 3000, tail=1500)}

## Contract (unchanged)
{CONTRACT.format(n_rows=n_rows)}

## Available columns — name | kind | values | description
{columns_table}

Return the complete corrected code in one ```python block. If part of the specification is impossible with the available columns, implement the closest valid version and skip the impossible part."""


# =============================================================================================
# Custom metric
# =============================================================================================
METRIC_SYSTEM = "You implement evaluation metrics in Python/numpy exactly as defined. Answer with one ```python block."


def metric_user(task_text: str, task_type: str) -> str:
    return f"""Implement the evaluation metric of this task.

Task description:
<<<
{truncate(task_text, 8000)}
>>>

Write:
```python
import numpy as np

GREATER_IS_BETTER = True  # or False

def metric(y_true, y_pred) -> float:
    ...
```
- y_true: 1-D numpy array of true values ({'numbers' if task_type == 'regression' else 'integer class indices 0..K-1'}).
- y_pred: {'1-D array of predicted values' if task_type == 'regression' else '2-D array (n, K) of class probabilities'}.
- Pure numpy, no sklearn."""
