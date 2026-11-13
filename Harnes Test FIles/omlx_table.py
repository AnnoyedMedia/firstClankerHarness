#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_queries(path: str, limit: int | None = None) -> list[str]:
    """Load queries from a text file, one per line."""
    p = Path(path)

    if not p.exists():
        sys.exit(f"Input file not found: {p}")

    queries = []

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove simple bullet prefixes
        if line.startswith("-"):
            line = line.lstrip("- ").strip()
        elif line.startswith("*"):
            line = line.lstrip("* ").strip()

        if line:
            queries.append(line)

    if limit is not None:
        queries = queries[:limit]

    return queries


def normalize_cell(value: str) -> str:
    """Make a value safe for TSV output."""
    return (
        value
        .replace("\t", " ")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .strip()
    )


def markdown_cell(value: str) -> str:
    """Make a value safe for Markdown table output."""
    return (
        value
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .strip()
    )


def extract_text(data) -> str:
    """
    Try to extract the LLM response text from common response formats.

    This handles:
      - OpenAI-compatible: {"choices": [{"message": {"content": "..."}}]}
      - simple: {"message": "..."}
      - simple: {"text": "..."}
      - simple: {"output": "..."}
      - plain string
    """
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        if "choices" in data:
            try:
                choice = data["choices"][0]

                if "message" in choice:
                    content = choice["message"].get("content", "")
                    return content if content is not None else ""

                if "text" in choice:
                    return choice["text"]
            except Exception:
                pass

        if "message" in data:
            return str(data["message"])

        if "text" in data:
            return str(data["text"])

        if "output" in data:
            return str(data["output"])

        if "result" in data:
            return str(data["result"])

    return json.dumps(data, ensure_ascii=False)


def call_omlx(query: str, args) -> dict:
    """Send one query to the oMLX endpoint."""
    url = f"{args.base_url}{args.endpoint}"

    messages = []

    if args.system:
        messages.append({
            "role": "system",
            "content": args.system
        })

    messages.append({
        "role": "user",
        "content": query
    })

    payload = {
        "messages": messages,
        "stream": False,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if args.model:
        payload["model"] = args.model

    headers = {
        "Content-Type": "application/json"
    }

    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_error = None

    for attempt in range(args.retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)

        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {detail}"

        except Exception as e:
            last_error = str(e)

        if attempt < args.retries:
            time.sleep(args.retry_wait)

    raise RuntimeError(last_error or "Unknown error")


def process_one(query: str, args) -> dict:
    """Process one query and return a row."""
    try:
        data = call_omlx(query, args)
        text = extract_text(data)

        return {
            "query": query,
            "response": text,
            "status": "ok",
            "error": "",
        }

    except Exception as e:
        return {
            "query": query,
            "response": "",
            "status": "error",
            "error": str(e),
        }


def write_output(rows: list[dict], output_path: str, fmt: str) -> None:
    """Write rows to TSV, CSV, or Markdown."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["query", "response", "status", "error"]

    if fmt == "tsv":
        lines = ["\t".join(fieldnames)]

        for row in rows:
            lines.append("\t".join(normalize_cell(row[k]) for k in fieldnames))

        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    elif fmt == "csv":
        import csv

        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                writer.writerow(row)

    elif fmt == "markdown":
        lines = [
            "| " + " | ".join(fieldnames) + " |",
            "| " + " | ".join(["---"] * len(fieldnames)) + " |",
        ]

        for row in rows:
            lines.append(
                "| "
                + " | ".join(markdown_cell(row[k]) for k in fieldnames)
                + " |"
            )

        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    else:
        sys.exit(f"Unsupported format: {fmt}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run queries from a text file through oMLX and write a table."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input text file containing one query per line."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output table file. Use .tsv, .csv, or .md."
    )

    parser.add_argument(
        "--format",
        choices=["tsv", "csv", "markdown"],
        default="tsv",
        help="Output table format."
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8080"),
        help="Base URL for oMLX. Example: http://127.0.0.1:8080"
    )

    parser.add_argument(
        "--endpoint",
        default=os.getenv("OMLX_ENDPOINT", "/v1/chat/completions"),
        help="Endpoint path. Example: /v1/chat/completions"
    )

    parser.add_argument(
        "--model",
        default=os.getenv("OMLX_MODEL", ""),
        help="Model name to send to oMLX. Optional if oMLX uses a default model."
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("OMLX_API_KEY", ""),
        help="Optional API key if your local oMLX server requires one."
    )

    parser.add_argument(
        "--system",
        default="",
        help="Optional system prompt."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature."
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per query."
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Request timeout in seconds."
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per failed request."
    )

    parser.add_argument(
        "--retry-wait",
        type=float,
        default=3,
        help="Seconds to wait between retries."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel requests. For local models, keep this low, e.g. 1 or 2."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N queries. Useful for testing."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    queries = load_queries(args.input, args.limit)

    if not queries:
        sys.exit("No queries found in input file.")

    print(f"Processing {len(queries)} query...")
    print(f"oMLX URL: {args.base_url}{args.endpoint}")
    print(f"Output: {args.output}")
    print()

    if args.workers <= 1:
        rows = []

        for i, query in enumerate(queries, start=1):
            preview = query[:60].replace("\n", " ")
            print(f"[{i}/{len(queries)}] {preview}")

            rows.append(process_one(query, args))

    else:
        def job(query: str) -> dict:
            return process_one(query, args)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(job, queries))

    write_output(rows, args.output, args.format)

    print()
    print(f"Done. Results written to: {args.output}")


if __name__ == "__main__":
    main()
