"""Start the server or platform launcher, make an HTTP request, then stop it."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher', action='store_true')
    args = parser.parse_args()
    if args.launcher:
        command = ['cmd', '/c', str(ROOT / 'run_windows.bat')] if os.name == 'nt' else ['bash', str(ROOT / 'run_mac_linux.sh')]
    else:
        command = [sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000']
    with tempfile.TemporaryFile(mode='w+b') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=os.name != 'nt')
        try:
            deadline = time.monotonic() + (240 if args.launcher else 30)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Server exited before it became ready.')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    with urllib.request.urlopen('http://127.0.0.1:8000/', timeout=2) as response:
                        assert b'<title>VisionLab Studio</title>' in response.read()
                    print('Launch and live HTTP checks passed.')
                    return
                except OSError:
                    time.sleep(.2)
            raise TimeoutError('Server did not start in time.')
        except Exception:
            log.seek(0)
            print(log.read().decode(errors='replace')[-8000:], file=sys.stderr)
            raise
        finally:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


if __name__ == '__main__':
    main()
