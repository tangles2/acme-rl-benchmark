"""
Finetune / refinement pipeline.

Extracts (context_text → next_tool_name) training examples from the provided
training_traces.jsonl. Each step in each trace becomes one supervised example:
  input  = task description + names of tools called so far + key result values
  label  = name of the NEXT tool to call

Trains a TF-IDF + LogisticRegression next-action classifier (CPU-friendly,
no GPU required, trains in < 1 second).

The trained model is saved to artifacts/ and returned as a PolicyAgent.

Usage:
    from src.train import train
    policy_agent = train()
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.agents import PolicyAgent

FIXTURES   = Path(__file__).parent.parent / "fixtures" / "rl-finetuning"
ARTIFACTS  = Path(__file__).parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS / "next_action_policy.pkl"


# ---------------------------------------------------------------------------
# Example extraction
# ---------------------------------------------------------------------------

def _context_at_step(task_input: str, messages: list[dict], step_idx: int) -> str:
    """
    Build a text representation of agent state at step `step_idx`.
    Includes the task description and all tool calls made before this step.
    """
    parts = [task_input]
    for i, msg in enumerate(messages[:step_idx]):
        tool_call = msg.get("tool_call", {})
        parts.append(tool_call.get("name", ""))
        args = tool_call.get("arguments", {})
        parts.extend(str(v) for v in args.values() if v is not None)
    return " ".join(parts)


def extract_examples(traces_path: Path) -> tuple[list[str], list[str]]:
    """
    Parse training_traces.jsonl and return (X_texts, y_labels).

    Each consecutive pair of steps in a trace generates one example:
      X = context at step i  (task + prior tool names + arg values)
      y = tool name at step i+1
    """
    X, y = [], []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trace = json.loads(line)
            task_input = trace.get("input", "")
            messages   = trace.get("messages", [])

            for i in range(len(messages)):
                context = _context_at_step(task_input, messages, i)
                label   = messages[i]["tool_call"]["name"]
                X.append(context)
                y.append(label)

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(verbose: bool = True) -> PolicyAgent:
    """
    Train the next-action policy on training_traces.jsonl.

    Steps:
      1. Extract (context, next_tool) pairs from traces.
      2. Fit TF-IDF vectorizer on context texts.
      3. Train LogisticRegression classifier.
      4. Persist model artifact to artifacts/.
      5. Return a PolicyAgent wrapping the fitted model.
    """
    traces_path = FIXTURES / "training_traces.jsonl"
    X, y = extract_examples(traces_path)

    if verbose:
        print(f"\n[train] Extracted {len(X)} examples from {traces_path.name}")
        label_counts = {}
        for label in y:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"[train] Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"          {label:30s} {count:3d}")

    # TF-IDF: character n-grams work well for short tool-name tokens.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X_vec = vectorizer.fit_transform(X)

    # LogisticRegression with high max_iter (small data converges instantly).
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="auto",
    )
    clf.fit(X_vec, y)

    train_acc = clf.score(X_vec, y)
    if verbose:
        print(f"[train] Training accuracy: {train_acc*100:.1f}%")
        print(f"[train] Classes: {list(clf.classes_)}")

    # Persist artifact.
    ARTIFACTS.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)
    if verbose:
        print(f"[train] Model saved → {MODEL_PATH}")

    return PolicyAgent(model=clf, vectorizer=vectorizer)


# ---------------------------------------------------------------------------
# Load pre-trained model
# ---------------------------------------------------------------------------

def load_policy() -> PolicyAgent:
    """Load a previously trained PolicyAgent from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train() first."
        )
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return PolicyAgent(model=obj["clf"], vectorizer=obj["vectorizer"])
