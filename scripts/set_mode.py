"""Switch audited modes at an idle boundary without discarding queued jobs."""
import argparse
import json
import subprocess
from pathlib import Path

from h3service.store import DATA, ROOT, database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tp6", "dual-tp3"])
    args = parser.parse_args()
    path = DATA / "verified-capabilities.json"
    cap = json.loads(path.read_text())
    if args.mode == "dual-tp3" and not cap.get("dual_tp3_verified"):
        parser.error("Cold and sustained dual-TP3 admission has not been recorded")
    drain = DATA / "drain"
    if drain.exists():
        parser.error("Another drain is already in progress")
    drain.touch(exist_ok=False)
    try:
        with database() as con:
            con.execute("BEGIN IMMEDIATE")
            active = con.execute("SELECT count(*) FROM jobs WHERE status IN ('starting','running')").fetchone()[0]
            if active:
                parser.error("Active jobs must finish before changing mode; queued jobs are preserved")
        def ctl(*command):
            subprocess.run(["systemctl", "--user", *command], check=True)
        ctl("stop", "aiserver-diffusers-worker-a.service", "aiserver-diffusers-worker-b.service")
        units = Path.home() / ".config/systemd/user"
        for name in ("aiserver-diffusers-inference.slice", "aiserver-diffusers-worker-a.service", "aiserver-diffusers-worker-b.service"):
            subprocess.run(["install", "-m", "644", str(ROOT / "deploy" / name), str(units / name)], check=True)
        override = units / "aiserver-diffusers-worker-a.service.d/dual-tp3.conf"
        if args.mode == "dual-tp3":
            override.parent.mkdir(parents=True, exist_ok=True)
            override.write_text("[Service]\nEnvironment=CUDA_VISIBLE_DEVICES=0,1,2\n")
            (DATA / "dual-tp3-verified").write_text("verified\n")
            cap["default_tp_size"] = 3
        else:
            override.unlink(missing_ok=True)
            cap["default_tp_size"] = 6
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(cap, indent=2))
        temp.replace(path)
        ctl("daemon-reload")
        ctl("enable", "--now", "aiserver-diffusers-worker-a.service")
        ctl("enable" if args.mode == "dual-tp3" else "disable", "aiserver-diffusers-worker-b.service")
        if args.mode == "dual-tp3":
            ctl("start", "aiserver-diffusers-worker-b.service")
        print(json.dumps({"mode": args.mode, "status": "workers_started", "readiness": "/ready"}))
    finally:
        drain.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
