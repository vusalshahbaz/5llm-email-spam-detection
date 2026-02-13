#!/usr/bin/env python3
"""
Evaluate Ollama LLMs on datasets/ling/test.csv: classify each row, compute
accuracy, per-class and macro precision/recall/F1, save report and predictions.
Supports --mode chat (system+user, /api/chat) or generate (single prompt, /api/generate).
Output: datasets/ling/<mode>/<model>/ (e.g. ling/chat/llama3.1:8b, ling/generate/llama3.1:8b).
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

# Models to evaluate (must be available in Ollama)
MODELS = [
    "mistral:7b",
    "deepseek-r1:8b",
    "falcon3:7b",
    "gemma3:4b",
    "llama3.1:8b",
]

# Endpoints: chat = system+user messages, generate = single prompt
CHAT_URL = os.environ.get("OLLAMA_EVAL_URL", "http://localhost:11434/api/chat")
GENERATE_URL = os.environ.get("OLLAMA_EVAL_GENERATE_URL", "http://localhost:11434/api/generate")
DEFAULT_TEST_CSV = Path(__file__).resolve().parent / "datasets/ling/test.csv"
DEFAULT_LING_DIR = Path(__file__).resolve().parent / "datasets/ling"
MAX_MESSAGE_LEN = 5000  # truncate message/email body beyond this (0 = no truncation)
PRINT_CONTENT_MAX = 300  # max chars of content to print per request
TIMEOUT_SEC = 120  # longer timeout when sending full emails (no truncation)
MAX_RETRIES = 3

# Compiled once for parse_prediction (avoids re-compiling per request)
_RE_01 = re.compile(r"\b[01]\b")


def load_test_rows(csv_path: Path):
    """Yield dicts with keys: subject, message, label (str '0' or '1')."""
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = row["label"].strip()
            if row["label"] not in ("0", "1"):
                continue
            yield row


SYSTEM_PROMPT = (
    "You are a spam email classifier. Output exactly one line in this format:\n"
    "  <label> | <reason>\n"
    "Label 0 = ham: genuine personal or professional correspondence, discussion, mailing list replies, no promotion.\n"
    "Label 1 = spam: phishing, scam, commercial or adult/dating promotion, bulk marketing or malicious content.\n"
    "Friendly tone alone is not ham. If it promotes a product, site, or service (including adult/dating), use 1. If it is real discussion or private correspondence, use 0.\n"
    "<reason> is one short sentence. Examples: 0 | Academic discussion.  1 | Promotional content with call-to-action link."
)


def build_messages_ling(subject: str, message: str) -> list[dict]:
    """Return [system, user] messages for /api/chat."""
    subject = (subject or "").strip()
    message = (message or "").strip()
    if MAX_MESSAGE_LEN and len(message) > MAX_MESSAGE_LEN:
        message = message[:MAX_MESSAGE_LEN] + "..."
    user_content = f'Subject: "{subject}"\n\nMessage: "{message}"'
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_prompt_ling(subject: str, message: str) -> str:
    """Single prompt for /api/generate (no system/user)."""
    subject = (subject or "").strip()
    message = (message or "").strip()
    if MAX_MESSAGE_LEN and len(message) > MAX_MESSAGE_LEN:
        message = message[:MAX_MESSAGE_LEN] + "..."
    return (
        SYSTEM_PROMPT + "\n\n"
        f'Subject: "{subject}"\n\nMessage: "{message}"'
    )


def parse_prediction(response_text: str) -> str | None:
    """Extract 0 or 1 from model output (format '<label> | <reason>'). Returns None if not found."""
    if not response_text:
        return None
    text = response_text.strip()
    # Expected format: "<label> | <reason>" — label is single digit 0 or 1
    parts = text.split("|", 1)
    if parts:
        label_part = parts[0].strip()
        if label_part in ("0", "1"):
            return label_part
    # Fallback: first standalone 0 or 1
    m = _RE_01.search(text)
    if m:
        return m.group(0)
    if "1" in text and "0" not in text[:20]:
        return "1"
    if "0" in text:
        return "0"
    return None


def predict_one_chat(
    session: requests.Session,
    url: str,
    model: str,
    messages: list[dict],
    timeout: int = TIMEOUT_SEC,
) -> tuple[str | None, str]:
    """POST /api/chat; return (parsed label or None, raw response text)."""
    payload = {"model": model, "messages": messages, "stream": False}
    if "deepseek" in model.lower():
        payload["think"] = False
    try:
        r = session.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        text = (msg.get("content") or data.get("response") or "").strip()
        pred = parse_prediction(text)
        return (pred, text or "(empty response)")
    except Exception as e:
        return (None, str(e))


def predict_one_generate(
    session: requests.Session,
    url: str,
    model: str,
    prompt: str,
    timeout: int = TIMEOUT_SEC,
) -> tuple[str | None, str]:
    """POST /api/generate with single prompt; return (parsed label or None, raw response text)."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    if "deepseek" in model.lower():
        payload["think"] = False
    try:
        r = session.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get("response") or "").strip()
        pred = parse_prediction(text)
        return (pred, text or "(empty response)")
    except Exception as e:
        return (None, str(e))


def compute_metrics(y_true: list[str], y_pred: list[str]):
    """Returns dict with accuracy, per-class and macro precision/recall/f1."""
    n = len(y_true)
    if n == 0:
        return {"error": "no samples"}

    # Confusion: true label -> predicted label
    tp0 = fp0 = fn0 = tn0 = 0
    tp1 = fp1 = fn1 = tn1 = 0
    for t, p in zip(y_true, y_pred):
        if p is None:
            continue
        if t == "0" and p == "0":
            tn1 += 1
            tp0 += 1
        elif t == "0" and p == "1":
            fp1 += 1
            fn0 += 1
        elif t == "1" and p == "0":
            fn1 += 1
            fp0 += 1
        elif t == "1" and p == "1":
            tp1 += 1
            tn0 += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    # Class 0 (negative/legit)
    prec0 = safe_div(tp0, tp0 + fp0)
    rec0 = safe_div(tp0, tp0 + fn0)
    f1_0 = safe_div(2 * prec0 * rec0, prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

    # Class 1 (positive/spam)
    prec1 = safe_div(tp1, tp1 + fp1)
    rec1 = safe_div(tp1, tp1 + fn1)
    f1_1 = safe_div(2 * prec1 * rec1, prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

    correct = sum(1 for t, p in zip(y_true, y_pred) if p is not None and t == p)
    valid = sum(1 for p in y_pred if p is not None)
    accuracy = safe_div(correct, valid) if valid else 0.0

    macro_precision = (prec0 + prec1) / 2
    macro_recall = (rec0 + rec1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    return {
        "accuracy": round(accuracy, 4),
        "by_class": {
            "0": {"precision": round(prec0, 4), "recall": round(rec0, 4), "f1": round(f1_0, 4)},
            "1": {"precision": round(prec1, 4), "recall": round(rec1, 4), "f1": round(f1_1, 4)},
        },
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "n_total": n,
        "n_valid": valid,
        "n_failed": n - valid,
    }


def run_model(
    url: str,
    model: str,
    csv_path: Path,
    ling_dir: Path,
    mode: str,
) -> None:
    """Run evaluation for one model; write under ling_dir/<mode>/<model>/ (e.g. ling/chat/llama3.1:8b)."""
    report_dir = ling_dir / mode / model
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = report_dir / "predictions.jsonl"
    errors_path = report_dir / "errors.jsonl"
    is_chat = mode == "chat"

    rows = list(load_test_rows(csv_path))
    n = len(rows)
    print(f"\n--- Model: {model} mode={mode} ({n} rows) ---", flush=True)

    y_true = []
    y_pred = []
    error_records = []

    with requests.Session() as session, open(predictions_path, "w", encoding="utf-8") as pred_f:
        for i, row in enumerate(rows):
            num = i + 1
            subject = (row.get("subject") or "").strip()
            content = (row.get("message") or "").strip()
            label = row["label"]

            pred = None
            last_error = None
            raw_response = ""
            prompt_used = None  # for generate: save full prompt in JSONL

            if is_chat:
                messages = build_messages_ling(subject, content)
                for attempt in range(MAX_RETRIES):
                    pred, raw_response = predict_one_chat(session, url, model, messages, timeout=TIMEOUT_SEC)
                    if pred is not None:
                        break
                    last_error = raw_response
                    if attempt < MAX_RETRIES - 1:
                        print(f"  [{num}/{n}] retry {attempt + 1}/{MAX_RETRIES}", file=sys.stderr)
            else:
                prompt_used = build_prompt_ling(subject, content)
                for attempt in range(MAX_RETRIES):
                    pred, raw_response = predict_one_generate(session, url, model, prompt_used, timeout=TIMEOUT_SEC)
                    if pred is not None:
                        break
                    last_error = raw_response
                    if attempt < MAX_RETRIES - 1:
                        print(f"  [{num}/{n}] retry {attempt + 1}/{MAX_RETRIES}", file=sys.stderr)

            if pred is None:
                error_records.append({
                    "subject": subject,
                    "message": content,
                    "label": label,
                    "predicted": None,
                    "error": last_error,
                    "row": num,
                })
                y_pred.append(None)
            else:
                y_pred.append(pred)

            y_true.append(label)

            rec = {
                "subject": subject,
                "message": content,
                "label": label,
                "predicted": pred,
                "response": raw_response,
            }
            if prompt_used is not None:
                rec["prompt"] = prompt_used
            if pred is None:
                rec["incorrect"] = True
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # One line per row to reduce terminal I/O and string formatting
            print(f"[{num}/{n}] label={label} predicted={pred if pred is not None else '?'}", flush=True)

            # Pause every 20 predictions to ease load on the LLM server
            if num % 20 == 0 and num < n:
                time.sleep(5)

        pred_f.flush()

    # Write errors.jsonl
    with open(errors_path, "w", encoding="utf-8") as err_f:
        for rec in error_records:
            err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = compute_metrics(y_true, y_pred)
    if "error" in metrics:
        print(metrics["error"], file=sys.stderr)
        return

    # Summary .txt: accuracy, macro (precision recall f1), per-class (0, 1) precision recall f1
    report_txt = report_dir / "eval_report.txt"
    lines = [
        "Ling test set – Ollama evaluation report",
        "=" * 50,
        f"Mode: {mode}",
        f"URL: {url}",
        f"Model: {model}",
        f"Samples: {metrics['n_total']} total, {metrics['n_valid']} valid, {metrics['n_failed']} failed",
        "",
        "Accuracy",
        "-" * 20,
        f"  {metrics['accuracy']:.4f}",
        "",
        "Macro (precision, recall, f1)",
        "-" * 30,
        f"  precision={metrics['macro']['precision']:.4f}  recall={metrics['macro']['recall']:.4f}  f1={metrics['macro']['f1']:.4f}",
        "",
        "By class (0 = legit, 1 = spam)",
        "-" * 30,
        f"  Class 0: precision={metrics['by_class']['0']['precision']:.4f}  recall={metrics['by_class']['0']['recall']:.4f}  f1={metrics['by_class']['0']['f1']:.4f}",
        f"  Class 1: precision={metrics['by_class']['1']['precision']:.4f}  recall={metrics['by_class']['1']['recall']:.4f}  f1={metrics['by_class']['1']['f1']:.4f}",
        "",
    ]
    report_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_txt}")
    print(f"Predictions: {predictions_path}")
    if error_records:
        print(f"Errors ({len(error_records)}): {errors_path}")


def load_predictions(predictions_path: Path) -> list[dict]:
    """Load predictions.jsonl into a list of dicts (subject, message, label, predicted)."""
    out = []
    if not predictions_path.is_file():
        return out
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_errors(errors_path: Path) -> list[dict]:
    """Load errors.jsonl into a list of dicts (row, subject, message, label, error, ...)."""
    out = []
    if not errors_path.is_file():
        return out
    with open(errors_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def retry_errors_for_model(
    url: str,
    model: str,
    ling_dir: Path,
    mode: str,
) -> None:
    """Re-run prediction only for rows in errors.jsonl; update predictions.jsonl, errors.jsonl, and report."""
    report_dir = ling_dir / mode / model
    predictions_path = report_dir / "predictions.jsonl"
    errors_path = report_dir / "errors.jsonl"
    is_chat = mode == "chat"

    predictions = load_predictions(predictions_path)
    error_records = load_errors(errors_path)

    if not predictions:
        print(f"[{model}] No predictions.jsonl found; run full eval first.", file=sys.stderr)
        return
    if not error_records:
        print(f"[{model}] No errors to retry.", flush=True)
        return

    n_total = len(predictions)
    print(f"\n--- Retry errors: {model} mode={mode} ({len(error_records)} rows to retry, {n_total} total) ---", flush=True)

    rows_to_retry = {rec["row"] - 1 for rec in error_records}
    still_failed = []

    with requests.Session() as session:
        for idx in sorted(rows_to_retry):
            if idx < 0 or idx >= n_total:
                continue
            rec = predictions[idx]
            subject = rec.get("subject", "")
            content = rec.get("message", "")
            label = rec.get("label", "")

            pred = None
            last_error = None
            raw_response = ""
            if is_chat:
                messages = build_messages_ling(subject, content)
                for attempt in range(MAX_RETRIES):
                    pred, raw_response = predict_one_chat(session, url, model, messages, timeout=TIMEOUT_SEC)
                    if pred is not None:
                        break
                    last_error = raw_response
                    if attempt < MAX_RETRIES - 1:
                        print(f"  row {idx + 1} retry {attempt + 1}/{MAX_RETRIES}", file=sys.stderr)
            else:
                prompt_used = build_prompt_ling(subject, content)
                for attempt in range(MAX_RETRIES):
                    pred, raw_response = predict_one_generate(session, url, model, prompt_used, timeout=TIMEOUT_SEC)
                    if pred is not None:
                        break
                    last_error = raw_response
                    if attempt < MAX_RETRIES - 1:
                        print(f"  row {idx + 1} retry {attempt + 1}/{MAX_RETRIES}", file=sys.stderr)
                predictions[idx]["prompt"] = prompt_used

            predictions[idx]["predicted"] = pred
            predictions[idx]["response"] = raw_response
            predictions[idx]["incorrect"] = pred is None
            print(f"[{idx + 1}/{n_total}] label={label} predicted={pred if pred is not None else '?'}", flush=True)

            if pred is None:
                still_failed.append({
                    "subject": subject,
                    "message": content,
                    "label": label,
                    "predicted": None,
                    "error": last_error,
                    "row": idx + 1,
                })

            if (idx + 1) % 20 == 0:
                time.sleep(5)

    # Write back full predictions
    with open(predictions_path, "w", encoding="utf-8") as pred_f:
        for rec in predictions:
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write back errors (only still-failed)
    with open(errors_path, "w", encoding="utf-8") as err_f:
        for rec in still_failed:
            err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Recompute metrics and report from full predictions
    y_true = [r["label"] for r in predictions]
    y_pred = [r.get("predicted") for r in predictions]
    metrics = compute_metrics(y_true, y_pred)
    if "error" in metrics:
        print(metrics["error"], file=sys.stderr)
        return

    report_txt = report_dir / "eval_report.txt"
    lines = [
        "Ling test set – Ollama evaluation report",
        "=" * 50,
        f"Mode: {mode}",
        f"URL: {url}",
        f"Model: {model}",
        f"Samples: {metrics['n_total']} total, {metrics['n_valid']} valid, {metrics['n_failed']} failed",
        "",
        "Accuracy",
        "-" * 20,
        f"  {metrics['accuracy']:.4f}",
        "",
        "Macro (precision, recall, f1)",
        "-" * 30,
        f"  precision={metrics['macro']['precision']:.4f}  recall={metrics['macro']['recall']:.4f}  f1={metrics['macro']['f1']:.4f}",
        "",
        "By class (0 = legit, 1 = spam)",
        "-" * 30,
        f"  Class 0: precision={metrics['by_class']['0']['precision']:.4f}  recall={metrics['by_class']['0']['recall']:.4f}  f1={metrics['by_class']['0']['f1']:.4f}",
        f"  Class 1: precision={metrics['by_class']['1']['precision']:.4f}  recall={metrics['by_class']['1']['recall']:.4f}  f1={metrics['by_class']['1']['f1']:.4f}",
        "",
    ]
    report_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_txt}")
    print(f"Predictions: {predictions_path}")
    print(f"Errors remaining: {len(still_failed)} -> {errors_path}")


def main():
    p = argparse.ArgumentParser(description="Evaluate Ollama on ling test.csv (multiple models)")
    p.add_argument("--mode", choices=("chat", "generate"), default="chat",
                   help="chat = system+user /api/chat; generate = single prompt /api/generate (default: chat)")
    p.add_argument("--url", default=None, help="Override API URL (default: CHAT_URL or GENERATE_URL by mode)")
    p.add_argument("--models", nargs="+", default=MODELS, help="Model names (default: all 5)")
    p.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV, help="Path to test.csv")
    p.add_argument("--ling-dir", type=Path, default=DEFAULT_LING_DIR, help="datasets/ling base (output: ling/<mode>/<model>/)")
    p.add_argument("--retry-errors", action="store_true", help="Re-run prediction only for rows in errors.jsonl; update predictions and report")
    args = p.parse_args()

    mode = args.mode
    url = args.url if args.url is not None else (CHAT_URL if mode == "chat" else GENERATE_URL)
    ling_dir = args.ling_dir

    if args.retry_errors:
        print(f"Retry-errors: mode={mode} URL={url}", flush=True)
        print(f"Models: {args.models}", flush=True)
        for model in args.models:
            retry_errors_for_model(url, model, ling_dir, mode)
        print("\nDone.")
        return

    csv_path = args.test_csv
    if not csv_path.is_file():
        print(f"Not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {csv_path} mode={mode} URL={url}", flush=True)
    print(f"Models: {args.models}", flush=True)
    print(f"Output: {ling_dir}/{mode}/<model>/eval_report.txt, predictions.jsonl, errors.jsonl", flush=True)

    for model in args.models:
        run_model(url, model, csv_path, ling_dir, mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
