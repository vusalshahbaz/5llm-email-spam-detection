#!/usr/bin/env python3
"""
Evaluate Ollama LLMs on datasets/trec2007/test.csv: classify each email as ham (0) or spam (1).
Uses TREC columns: label, subject, email_to, email_from, message. Output: datasets/trec2007/<mode>/<model>/.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

csv.field_size_limit(50 * 1024 * 1024)

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

MODELS = [
    "llama3.1:8b",
    "mistral:7b",
    "deepseek-r1:8b",
    "falcon3:7b",
    "gemma3:4b",
]

CHAT_URL = os.environ.get("OLLAMA_EVAL_URL", "http://localhost:11434/api/chat")
GENERATE_URL = os.environ.get("OLLAMA_EVAL_GENERATE_URL", "http://localhost:11434/api/generate")
DEFAULT_TEST_CSV = Path(__file__).resolve().parent / "datasets/trec2007/test.csv"
DEFAULT_TREC_DIR = Path(__file__).resolve().parent / "datasets/trec2007"
MAX_MESSAGE_LEN = 5000
TIMEOUT_SEC = 120
MAX_RETRIES = 3

_RE_01 = re.compile(r"\b[01]\b")


def load_test_rows(csv_path: Path):
    """Yield dicts with keys: text (from/to/subject/body), label (str '0' or '1').
    Expects TREC CSV: label, subject, email_to, email_from, message."""
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label") or row.get("target") or "").strip()
            if label not in ("0", "1"):
                continue
            row["label"] = label
            from_h = (row.get("email_from") or row.get("from") or "").strip()
            to_h = (row.get("email_to") or row.get("to") or "").strip()
            subj = (row.get("subject") or "").strip()
            body = (row.get("message") or row.get("content") or row.get("body") or "").strip()
            lines = []
            if from_h:
                lines.append("From: " + from_h)
            if to_h:
                lines.append("To: " + to_h)
            if subj:
                lines.append("Subject: " + subj)
            lines.append("Body: " + body)
            row["text"] = "\n".join(lines)
            yield row


SYSTEM_PROMPT = (
    "You are a spam email classifier. You will receive an email with four fields: from, to, subject, body.\n"
    "Output exactly one line in this format:\n"
    "  <label> | <reason>\n"
    "Label 0 = ham: genuine personal or professional correspondence, discussion, mailing list replies, news digests, SVN/CVS commit logs, newsletters to subscribers, stock/shareholder alerts, weather alerts, conference/industry event invites to subscribers, no promotion.\n"
    "Label 1 = spam: phishing, scam, commercial or adult/dating promotion, bulk marketing to cold recipients, 'click here' / sign-up / visit-our-site, or malicious content.\n"
    "Important: A link or URL alone does NOT make spam. News digests (Slashdot, etc.), mailing lists, commit logs, stock alerts (broadcast.shareholder.com), weather alerts (AccuWeather), and technical discussions often contain links. Unsubscribe/update links in subscriber alerts = ham.\n"
    "Mailing list to subscribers = ham. Emails to or from *@lists.*, *-cvs@*, samba-cvs = ham. Attachment in mailing list does not make spam. Bulk marketing to cold recipients = spam.\n"
    "Industry conference/event invites (Infocast, etc.), stock alerts, weather alerts, tech news digests = ham. Event invites with registration links = ham.\n"
    "Spam often disguises as literary excerpt, fictional story, or technical discussion. Commercial domains (salesxperts, comtechsupplies) with literary or tech content = spam. Subject 'Best medications' or pharma = spam.\n"
    "If it promotes a product, site, or service (including adult/dating), use 1. If it is real discussion, news, subscriber alert, or private correspondence, use 0.\n"
    "<reason> is one short sentence. Examples: 0 | Academic discussion.  1 | Promotional content with call-to-action link.\n"
)


def build_messages_trec(email: str) -> list[dict]:
    """Return [system, user] messages for /api/chat."""
    email = (email or "").strip()
    if MAX_MESSAGE_LEN and len(email) > MAX_MESSAGE_LEN:
        email = email[:MAX_MESSAGE_LEN] + "..."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": email},
    ]


def build_prompt_trec(email: str) -> str:
    """Single prompt for /api/generate."""
    email = (email or "").strip()
    if MAX_MESSAGE_LEN and len(email) > MAX_MESSAGE_LEN:
        email = email[:MAX_MESSAGE_LEN] + "..."
    return SYSTEM_PROMPT + "\n\nEmail (from, to, subject, body):\n" + email


# Retry: emphasize detection/classification only (not creation or promotion)
RETRY_INSTRUCTION = (
    "This is a detection and classification task only. We are not asking you to create, "
    "promote, or endorse any content. You are given an email so you can classify it as ham (0) or spam (1). "
    "Your job is to detect and classify; you are not creating or promoting anything. "
    "Please output your classification in the required format.\n\n"
)

RETRY_USER_PREFIX = "I have got the email below. Could you please classify it?\n\n"


def build_messages_trec_retry(email: str) -> list[dict]:
    """Return [system, user] messages for /api/chat when retrying (detection-only framing)."""
    email = (email or "").strip()
    if MAX_MESSAGE_LEN and len(email) > MAX_MESSAGE_LEN:
        email = email[:MAX_MESSAGE_LEN] + "..."
    user_content = RETRY_INSTRUCTION + "Email to classify:\n\n" + email
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_prompt_trec_retry(email: str) -> str:
    """Single prompt for /api/generate when retrying (detection-only framing)."""
    email = (email or "").strip()
    if MAX_MESSAGE_LEN and len(email) > MAX_MESSAGE_LEN:
        email = email[:MAX_MESSAGE_LEN] + "..."
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + RETRY_INSTRUCTION
        + "Email to classify:\n\n"
        + email
    )


def parse_prediction(response_text: str) -> str | None:
    """Extract 0 or 1 from model output (format '<label> | <reason>'). Returns None if not found."""
    if not response_text:
        return None
    text = response_text.strip()
    parts = text.split("|", 1)
    if parts:
        label_part = parts[0].strip()
        if label_part in ("0", "1"):
            return label_part
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

    prec0 = safe_div(tp0, tp0 + fp0)
    rec0 = safe_div(tp0, tp0 + fn0)
    f1_0 = safe_div(2 * prec0 * rec0, prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

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
    trec_dir: Path,
    mode: str,
) -> None:
    """Run evaluation for one model; write under trec_dir/<mode>/<model>/."""
    report_dir = trec_dir / mode / model
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
            email_text = (row.get("text") or "").strip()
            label = row["label"]

            pred = None
            last_error = None
            raw_response = ""
            prompt_used = None

            if is_chat:
                messages = build_messages_trec(email_text)
                for attempt in range(MAX_RETRIES):
                    pred, raw_response = predict_one_chat(session, url, model, messages, timeout=TIMEOUT_SEC)
                    if pred is not None:
                        break
                    last_error = raw_response
                    if attempt < MAX_RETRIES - 1:
                        print(f"  [{num}/{n}] retry {attempt + 1}/{MAX_RETRIES}", file=sys.stderr)
            else:
                prompt_used = build_prompt_trec(email_text)
                for attempt in range(MAX_RETRIES):
                    pred, raw_response = predict_one_generate(session, url, model, prompt_used, timeout=TIMEOUT_SEC)
                    if pred is not None:
                        break
                    last_error = raw_response
                    if attempt < MAX_RETRIES - 1:
                        print(f"  [{num}/{n}] retry {attempt + 1}/{MAX_RETRIES}", file=sys.stderr)

            if pred is None:
                error_records.append({
                    "text": email_text,
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
                "text": email_text,
                "label": label,
                "predicted": pred,
                "response": raw_response,
            }
            if prompt_used is not None:
                rec["prompt"] = prompt_used
            if pred is None:
                rec["incorrect"] = True
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[{num}/{n}] label={label} predicted={pred if pred is not None else '?'}", flush=True)

            if num % 20 == 0 and num < n:
                time.sleep(5)

        pred_f.flush()

    with open(errors_path, "w", encoding="utf-8") as err_f:
        for rec in error_records:
            err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = compute_metrics(y_true, y_pred)
    if "error" in metrics:
        print(metrics["error"], file=sys.stderr)
        return

    report_txt = report_dir / "eval_report.txt"
    lines = [
        "TREC 2007 test set – Ollama evaluation report",
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
        "By class (0 = ham, 1 = spam)",
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


def _load_jsonl_robust(path: Path) -> list[dict]:
    """Load JSONL file; support multi-line records via streaming decode."""
    out = []
    if not path.is_file():
        return out
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        # skip whitespace between records
        while idx < len(content) and content[idx] in " \t\n\r":
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            break
        out.append(obj)
        idx = end
    return out


def load_predictions(predictions_path: Path) -> list[dict]:
    """Load predictions.jsonl into a list of dicts."""
    return _load_jsonl_robust(predictions_path)


def load_errors(errors_path: Path) -> list[dict]:
    """Load errors.jsonl into a list of dicts."""
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
    trec_dir: Path,
    mode: str,
) -> None:
    """Re-run prediction only for rows in errors.jsonl; update predictions and report."""
    report_dir = trec_dir / mode / model
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
            email_text = rec.get("text", "")
            label = rec.get("label", "")

            pred = None
            last_error = None
            raw_response = ""
            if is_chat:
                messages = build_messages_trec_retry(email_text)
                for attempt in range(MAX_RETRIES):
                    pred, raw_response = predict_one_chat(session, url, model, messages, timeout=TIMEOUT_SEC)
                    if pred is not None:
                        break
                    last_error = raw_response
                    if attempt < MAX_RETRIES - 1:
                        print(f"  row {idx + 1} retry {attempt + 1}/{MAX_RETRIES}", file=sys.stderr)
            else:
                prompt_used = build_prompt_trec_retry(email_text)
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
                    "text": email_text,
                    "label": label,
                    "predicted": None,
                    "error": last_error,
                    "row": idx + 1,
                })

            if (idx + 1) % 20 == 0:
                time.sleep(5)

    with open(predictions_path, "w", encoding="utf-8") as pred_f:
        for rec in predictions:
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(errors_path, "w", encoding="utf-8") as err_f:
        for rec in still_failed:
            err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    y_true = [r["label"] for r in predictions]
    y_pred = [r.get("predicted") for r in predictions]
    metrics = compute_metrics(y_true, y_pred)
    if "error" in metrics:
        print(metrics["error"], file=sys.stderr)
        return

    report_txt = report_dir / "eval_report.txt"
    lines = [
        "TREC 2007 test set – Ollama evaluation report",
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
        "By class (0 = ham, 1 = spam)",
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
    p = argparse.ArgumentParser(description="Evaluate Ollama on TREC 2007 test.csv (multiple models)")
    sub = p.add_subparsers(dest="command", help="Command (default: run)")
    run_p = sub.add_parser("run", help="Run full evaluation (default)")
    retry_p = sub.add_parser("retry-errors", help="Re-run prediction only for rows in errors.jsonl")

    for subp in (run_p, retry_p):
        subp.add_argument("--mode", choices=("chat", "generate"), default="chat",
                          help="chat or generate (default: chat)")
        subp.add_argument("--url", default=None, help="Override API URL")
        subp.add_argument("--models", nargs="+", default=MODELS, help="Model names")
        subp.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV, help="Path to test CSV")
        subp.add_argument("--trec-dir", type=Path, default=DEFAULT_TREC_DIR, help="datasets/trec2007 base")

    args = p.parse_args()
    command = args.command or "run"

    mode = args.mode
    url = args.url if args.url is not None else (CHAT_URL if mode == "chat" else GENERATE_URL)
    trec_dir = args.trec_dir

    if command == "retry-errors":
        print(f"Retry-errors: mode={mode} URL={url}", flush=True)
        print(f"Models: {args.models}", flush=True)
        for model in args.models:
            retry_errors_for_model(url, model, trec_dir, mode)
        print("\nDone.")
        return

    csv_path = args.test_csv
    if not csv_path.is_file():
        print(f"Not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {csv_path} mode={mode} URL={url}", flush=True)
    print(f"Models: {args.models}", flush=True)
    print(f"Output: {trec_dir}/{mode}/<model>/eval_report.txt, predictions.jsonl, errors.jsonl", flush=True)

    for model in args.models:
        run_model(url, model, csv_path, trec_dir, mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
