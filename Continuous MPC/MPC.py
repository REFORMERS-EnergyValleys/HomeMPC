# scheduler_main.py
import subprocess
import sys
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler

RUN_SCRIPT = Path(__file__).with_name("MPC_single_run_24.py")

def launch_mpc():
    # inherits stdout/stderr, so you still see everything in tmux
    subprocess.run([sys.executable, str(RUN_SCRIPT)], check=True)

if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Europe/Brussels")
    scheduler.add_job(
        launch_mpc,
        "cron",
        minute="0,15,30,45",
        second=5,
        id="MPC_job",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )
    scheduler.start()