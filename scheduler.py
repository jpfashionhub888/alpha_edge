# scheduler.py

"""
Auto-runs the scanner at market open every weekday.
Set it and forget it.

Usage: python scheduler.py
Leave it running. It handles everything.
"""

import schedule
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)


def run_scan():
    """Run the daily scan."""

    day = datetime.now().strftime('%A')

    # Skip weekends
    if day in ['Saturday', 'Sunday']:
        print(f"\n   📅 {day} - Market closed. Skipping.")
        return

    print(f"\n   📅 {day} - Running daily scan...")

    try:
        from main import run_daily_scan
        run_daily_scan()
    except Exception as e:
        logger.error(f"Scan failed: {e}")


def main():
    print("\n" + "⏰" * 25)
    print("ALPHAEDGE AUTO SCHEDULER")
    print("⏰" * 25)

    print("\nSchedule:")
    print("   Market scan: 9:45 AM ET (Mon-Fri)")
    print("   Mid-day update: 12:30 PM ET (Mon-Fri)")
    print("   End of day: 4:15 PM ET (Mon-Fri)")
    print("\nPress Ctrl+C to stop\n")

    # Schedule scans
    schedule.every().monday.at("09:45").do(run_scan)
    schedule.every().tuesday.at("09:45").do(run_scan)
    schedule.every().wednesday.at("09:45").do(run_scan)
    schedule.every().thursday.at("09:45").do(run_scan)
    schedule.every().friday.at("09:45").do(run_scan)

    # Mid-day update
    schedule.every().monday.at("12:30").do(run_scan)
    schedule.every().tuesday.at("12:30").do(run_scan)
    schedule.every().wednesday.at("12:30").do(run_scan)
    schedule.every().thursday.at("12:30").do(run_scan)
    schedule.every().friday.at("12:30").do(run_scan)

    # End of day
    schedule.every().monday.at("16:15").do(run_scan)
    schedule.every().tuesday.at("16:15").do(run_scan)
    schedule.every().wednesday.at("16:15").do(run_scan)
    schedule.every().thursday.at("16:15").do(run_scan)
    schedule.every().friday.at("16:15").do(run_scan)

    # Run first scan immediately
    run_scan()

    # Keep running forever
    while True:
        schedule.run_pending()
        next_run = schedule.next_run()
        if next_run:
            now_str = datetime.now().strftime('%H:%M:%S')
            next_str = next_run.strftime('%H:%M:%S')
            print(
                f"   ⏰ {now_str} | Next scan at {next_str}",
                end='\r'
            )
        time.sleep(60)


if __name__ == "__main__":
    main()