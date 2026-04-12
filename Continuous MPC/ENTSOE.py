# scheduler_main.py
import subprocess
import sys
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler

RUN_SCRIPT_ENTSOE = Path(__file__).with_name("ENTSOE_single_run.py")
RUN_SCRIPT_EXTENSION = Path(__file__).with_name("ENTSOE_extend.py")

def launch_entsoe():
    # inherits stdout/stderr, so you still see everything in tmux
    subprocess.run([sys.executable, str(RUN_SCRIPT_ENTSOE)], check=True)

def launch_extension():
    # inherits stdout/stderr, so you still see everything in tmux
    subprocess.run([sys.executable, str(RUN_SCRIPT_EXTENSION)], check=True)

if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Europe/Brussels")
    scheduler.add_job(launch_entsoe, "cron", hour=13, minute=30, id="ENTSOE_job", max_instances=1, coalesce=True, misfire_grace_time=60)
    scheduler.add_job(launch_extension, "cron", minute="0, 15, 30, 45", second=45, id="Extension_job", max_instances=1, coalesce=True, misfire_grace_time=60)
    scheduler.start()