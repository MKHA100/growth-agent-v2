"""Quick e2e test to capture tracebacks from domain agent failures."""
import json
import os
import urllib.request

BASE = os.environ.get("LANGGRAPH_API_URL", "http://127.0.0.1:2024")
ASSISTANT_ID = os.environ.get("LANGGRAPH_ASSISTANT_ID", "growth_intelligence")


def main() -> None:
    # Create thread
    req = urllib.request.Request(
        f"{BASE}/threads",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    thread = json.loads(urllib.request.urlopen(req).read())
    thread_id = thread["thread_id"]
    print(f"Thread: {thread_id}")

    # Run stream
    payload = json.dumps({
        "assistant_id": ASSISTANT_ID,
        "input": {
            "messages": [{"role": "human", "content": "Is Lilian competitive? How is Vector agents doing in the AI SDR market?"}]
        },
        "stream_mode": ["updates"],
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/threads/{thread_id}/runs/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)

    printed = False
    for raw_line in resp:
        line = raw_line.decode().strip()
        if line.startswith("data: "):
            data = json.loads(line[6:])
            for node, val in data.items():
                if node.startswith("run_") and not printed:
                    findings = val.get("domain_findings", {})
                    for d, f in findings.items():
                        if f.get("status") == "failed":
                            print(f"\n=== {d} FAILED ===")
                            reason = f.get("error_reason", "")
                            print(reason[:3000])
                            printed = True
                            break

    print("\nDone.")


if __name__ == "__main__":
    main()
