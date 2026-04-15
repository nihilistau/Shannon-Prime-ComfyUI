"""Queue a ComfyUI API-format workflow and wait for completion.

Usage:
  python run_workflow.py <workflow.json> [--server http://127.0.0.1:8188]

Exits 0 on success, prints output file paths relative to the ComfyUI output dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def queue_workflow(server: str, workflow_path: str) -> tuple[str, float]:
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    payload = {"prompt": workflow, "client_id": "shannon_prime_test"}
    resp = _post_json(f"{server.rstrip('/')}/prompt", payload)
    return resp["prompt_id"], time.time()


def wait_for_completion(server: str, prompt_id: str, timeout: float = 1800.0,
                         poll_interval: float = 3.0) -> dict:
    t0 = time.time()
    last_status = None
    while time.time() - t0 < timeout:
        hist = _get_json(f"{server.rstrip('/')}/history/{prompt_id}")
        entry = hist.get(prompt_id)
        if entry is not None and entry.get("status", {}).get("completed") is True:
            return entry
        # Emit queue-position progress
        try:
            queue = _get_json(f"{server.rstrip('/')}/queue")
            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            status = f"running={len(running)} pending={len(pending)}"
            if status != last_status:
                print(f"[t={int(time.time()-t0):4d}s] {status}", flush=True)
                last_status = status
        except Exception:
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"Workflow {prompt_id} did not complete within {timeout}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    print(f"Queueing {args.workflow}", flush=True)
    prompt_id, t_queue = queue_workflow(args.server, args.workflow)
    print(f"Queued prompt_id={prompt_id}", flush=True)

    entry = wait_for_completion(args.server, prompt_id, timeout=args.timeout)
    t_done = time.time()
    print(f"Completed in {t_done - t_queue:.1f}s", flush=True)

    outputs = entry.get("outputs", {})
    if not outputs:
        print("WARN: no outputs in history entry", flush=True)
        return 1

    for node_id, node_out in outputs.items():
        for kind, items in node_out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and "filename" in it:
                    subfolder = it.get("subfolder", "")
                    fn = it["filename"]
                    print(f"  node {node_id} [{kind}]: {subfolder}/{fn}" if subfolder else f"  node {node_id} [{kind}]: {fn}",
                          flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
