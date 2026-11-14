"""Production Deployment Supervisor for Credit Card Fraud Detection Application.

Orchestrates Streamlit and Cloudflare Edge Tunnel with:
- Non-blocking real-time threaded stream reading and log flushing
- Dynamic port conflict resolution & stale process termination
- Fast URL discovery and edge-network propagation verification
- Real-time periodic healthchecks
- Auto-reconnect / auto-recovery if edge connection drops
"""

import sys
import os
import re
import time
import json
import socket
import threading
import subprocess
import urllib.request
import signal
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"
CLOUDFLARED_BIN = BASE_DIR / "tools" / "cloudflared.exe"
OUTPUTS_DIR = BASE_DIR / "outputs"
STATUS_FILE = OUTPUTS_DIR / "deployment_info.json"
ST_LOG_FILE = OUTPUTS_DIR / "streamlit.log"
CF_LOG_FILE = OUTPUTS_DIR / "cloudflared.log"

STREAMLIT_PORT = 8501
HEALTH_URL = f"http://127.0.0.1:{STREAMLIT_PORT}/_stcore/health"


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [DevOps Supervisor] {msg}", flush=True)


def cleanup_stale_processes():
    """Ensure no stale cloudflared or port 8501 listeners exist."""
    log("Cleaning up any existing cloudflared instances...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe"], capture_output=True, timeout=5)
    except Exception:
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex(("127.0.0.1", STREAMLIT_PORT))
        sock.close()
        if result == 0:
            log(f"Port {STREAMLIT_PORT} is in use. Freeing port...")
            out = subprocess.check_output(f"netstat -ano | findstr :{STREAMLIT_PORT}", shell=True, text=True, timeout=5)
            for line in out.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and "LISTENING" in parts:
                    pid = parts[-1]
                    if pid and pid != "0" and pid != str(os.getpid()):
                        log(f"Terminating stale process on port {STREAMLIT_PORT} (PID {pid})...")
                        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=5)
            time.sleep(1)
    except Exception as e:
        log(f"Port cleanup note: {e}")


def stream_reader(proc, log_file_path, url_event=None, url_holder=None):
    """Threaded stream consumer that flushes logs and extracts URLs in real time."""
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                f.write(line)
                f.flush()

                if url_event and url_holder is not None and not url_holder:
                    match = url_pattern.search(line)
                    if match:
                        url = match.group(0)
                        url_holder.append(url)
                        url_event.set()
    except Exception as e:
        log(f"Stream reader exception for {log_file_path.name}: {e}")


def wait_for_streamlit_healthy(timeout: int = 60) -> bool:
    """Poll Streamlit's local healthcheck endpoint until 200 OK."""
    start_time = time.time()
    log("Waiting for Streamlit server to report healthy (warmup & ML model loading)...")
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "HealthCheck/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    log("Streamlit local instance is healthy (HTTP 200 OK).")
                    return True
        except Exception:
            time.sleep(1.5)
    log("Timeout waiting for Streamlit health check.")
    return False


def verify_live_url(public_url: str, max_retries: int = 15) -> bool:
    """Verify that Cloudflare edge has propagated and proxies correctly to Streamlit."""
    log(f"Probing Cloudflare edge health at {public_url}/_stcore/health ...")
    health_url = f"{public_url}/_stcore/health"
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EdgeProbe/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    log(f"Public URL verified successfully on attempt {attempt}: HTTP 200 OK")
                    return True
        except Exception as e:
            log(f"Edge probe attempt {attempt}/{max_retries} pending DNS/Edge sync: {e}")
            time.sleep(2)
    return False


def start_streamlit():
    """Launch Streamlit in headless production mode."""
    log("Starting Streamlit production server...")
    streamlit_cmd = [
        str(VENV_PYTHON),
        "-m", "streamlit", "run", "main.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]

    proc = subprocess.Popen(
        streamlit_cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    t = threading.Thread(target=stream_reader, args=(proc, ST_LOG_FILE), daemon=True)
    t.start()
    return proc


def start_cloudflared(url_holder):
    """Launch Cloudflare tunnel and capture edge URL."""
    log("Spawning Cloudflare Edge Tunnel connector...")
    cf_cmd = [
        str(CLOUDFLARED_BIN),
        "tunnel",
        "--url", f"http://127.0.0.1:{STREAMLIT_PORT}",
        "--edge-ip-version", "4",
        "--no-autoupdate"
    ]

    proc = subprocess.Popen(
        cf_cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    url_event = threading.Event()
    t = threading.Thread(target=stream_reader, args=(proc, CF_LOG_FILE, url_event, url_holder), daemon=True)
    t.start()

    log("Awaiting edge tunnel allocation from Cloudflare...")
    signaled = url_event.wait(timeout=45)
    if signaled and url_holder:
        return proc, url_holder[0]
    return proc, None


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_processes()

    streamlit_proc = start_streamlit()

    if not wait_for_streamlit_healthy():
        log("ERROR: Streamlit failed to reach healthy state.")
        streamlit_proc.terminate()
        sys.exit(1)

    url_holder = []
    cf_proc, public_url = start_cloudflared(url_holder)

    if not public_url:
        log("ERROR: Could not discover public URL in Cloudflare tunnel logs.")
        streamlit_proc.terminate()
        if cf_proc:
            cf_proc.terminate()
        sys.exit(1)

    log("=" * 68)
    log(f"PUBLIC CLOUDFLARE EDGE URL ALLOCATED: {public_url}")
    log("=" * 68)

    # Edge propagation check
    is_live = verify_live_url(public_url)
    if not is_live:
        log("WARNING: Initial edge probe timed out, but tunnel process is active. Continuing...")

    deployment_info = {
        "status": "ONLINE",
        "provider": "Cloudflare Edge Global Anycast",
        "live_url": public_url,
        "local_url": f"http://127.0.0.1:{STREAMLIT_PORT}",
        "ssl_enabled": True,
        "websocket_enabled": True,
        "architecture": "Cloudflare Global Edge Anycast -> Cloudflare Tunnel (QUIC) -> Streamlit Microservice (Python 3.14)",
        "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "server_pid": streamlit_proc.pid,
        "tunnel_pid": cf_proc.pid
    }

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(deployment_info, f, indent=2)

    log(f"Deployment info recorded to {STATUS_FILE.name}")
    log("Supervisor entering active telemetry and edge watchdog loop...")

    def handle_exit(signum, frame):
        log("Shutdown signal received. Terminating processes...")
        streamlit_proc.terminate()
        cf_proc.terminate()
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["status"] = "STOPPED"
                with open(STATUS_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    cycle_count = 0
    while True:
        time.sleep(5)
        cycle_count += 1

        # Check Streamlit process
        if streamlit_proc.poll() is not None:
            log(f"CRITICAL: Streamlit terminated unexpectedly with code {streamlit_proc.returncode}")
            cf_proc.terminate()
            break

        # Check Cloudflared process
        if cf_proc.poll() is not None:
            log(f"WARNING: Cloudflared exited with code {cf_proc.returncode}. Reconnecting tunnel...")
            url_holder = []
            cf_proc, new_url = start_cloudflared(url_holder)
            if new_url:
                public_url = new_url
                log(f"NEW PUBLIC URL RE-ESTABLISHED: {public_url}")
                deployment_info["live_url"] = public_url
                deployment_info["deployed_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                with open(STATUS_FILE, "w", encoding="utf-8") as f:
                    json.dump(deployment_info, f, indent=2)

        # Periodic healthcheck every 30 seconds
        if cycle_count % 6 == 0:
            local_ok = False
            try:
                req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "Watchdog/1.0"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    local_ok = (resp.status == 200)
            except Exception:
                pass

            if not local_ok:
                log("Local healthcheck failed! Streamlit may be hung.")


if __name__ == "__main__":
    main()
