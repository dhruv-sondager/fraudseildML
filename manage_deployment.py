"""Cross-platform Deployment CLI for FraudShield ML.
Works directly via Python, avoiding PowerShell ExecutionPolicy restrictions.
"""

import sys
import os
import json
import urllib.request
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent
STATUS_FILE = BASE_DIR / "outputs" / "deployment_info.json"
SUPERVISOR_SCRIPT = BASE_DIR / "deploy_supervisor.py"
PYTHON_BIN = BASE_DIR / "venv" / "Scripts" / "python.exe"


def status():
    print("\n" + "=" * 64)
    print("         FraudShield ML - Production Deployment Status")
    print("=" * 64)
    if not STATUS_FILE.exists():
        print("Status:       NOT RUNNING (No deployment metadata found)")
        print("Action:       Run 'python manage_deployment.py start' to launch.")
        print("=" * 64 + "\n")
        return

    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading deployment metadata: {e}")
        return

    print(f"Status:       {data.get('status', 'UNKNOWN')}")
    print(f"Provider:     {data.get('provider', 'Cloudflare')}")
    print(f"Live URL:     {data.get('live_url', 'N/A')}")
    print(f"Local URL:    {data.get('local_url', 'N/A')}")
    print(f"SSL/TLS:      {'Enabled (HTTPS)' if data.get('ssl_enabled') else 'Disabled'}")
    print(f"Architecture: {data.get('architecture', 'N/A')}")
    print(f"Deployed At:  {data.get('deployed_at', 'N/A')}")

    # Probe live URL health if available
    live_url = data.get("live_url")
    if live_url and live_url != "N/A":
        health_url = f"{live_url}/_stcore/health"
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "StatusProbe/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print("Edge Health:  ONLINE (HTTP 200 OK - Publicly Reachable)")
                else:
                    print(f"Edge Health:  HTTP {resp.status}")
        except Exception as e:
            print(f"Edge Health:  CONNECTING / OFFLINE ({e})")

    print("=" * 64 + "\n")


def stop():
    print("Stopping deployment processes...")
    subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe"], capture_output=True)
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            spid = data.get("server_pid")
            tpid = data.get("tunnel_pid")
            if spid:
                subprocess.run(["taskkill", "/F", "/PID", str(spid)], capture_output=True)
            if tpid:
                subprocess.run(["taskkill", "/F", "/PID", str(tpid)], capture_output=True)
            data["status"] = "STOPPED"
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    try:
        out = subprocess.check_output("netstat -ano | findstr :8501", shell=True, text=True)
        for line in out.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = parts[-1]
                if pid and pid != "0":
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    except Exception:
        pass

    print("Deployment stopped cleanly.")


def start():
    print("Starting deployment supervisor...")
    proc = subprocess.Popen(
        [str(PYTHON_BIN), str(SUPERVISOR_SCRIPT)],
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
    )
    print(f"Supervisor initiated (PID: {proc.pid}).")
    print("Monitoring startup sequence...")


def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "status"
    if arg == "status":
        status()
    elif arg == "stop":
        stop()
    elif arg == "start":
        start()
    elif arg == "restart":
        stop()
        start()
    else:
        print("Usage: python manage_deployment.py [status|start|stop|restart]")


if __name__ == "__main__":
    main()
