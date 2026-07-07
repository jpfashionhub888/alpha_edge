# scripts/check_key_age.py
"""
AlphaEdge — API Key Rotation Reminder

Tracks when each API key was last rotated and sends a weekly
Telegram reminder if any key exceeds ROTATION_DAYS.

Institutional standard: rotate all API keys every 90 days.

Run weekly via cron or alphaedge-audit.timer.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

KEY_AGE_FILE  = Path('logs/key_ages.json')
ROTATION_DAYS = 90    # alert if key is older than this

TRACKED_KEYS = [
    ('ALPACA',      'Alpaca paper/live trading API'),
    ('ALPACA_LIVE', 'Alpaca live account (if separate)'),
    ('GATE_IO',     'Gate.io exchange API'),
    ('BYBIT',       'Bybit exchange API'),
    ('GROQ',        'Groq AI (Llama3 veto agent)'),
    ('TELEGRAM',    'Telegram Bot Token'),
]


def load_key_ages() -> dict:
    try:
        with open(KEY_AGE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning('Could not load key ages: %s', e)
        return {}


def save_key_ages(ages: dict) -> None:
    try:
        KEY_AGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(KEY_AGE_FILE, 'w') as f:
            json.dump(ages, f, indent=2)
    except Exception as e:
        logger.warning('Could not save key ages: %s', e)


def record_rotation(key_name: str) -> None:
    """Call this after manually rotating a key to reset its age."""
    ages = load_key_ages()
    ages[key_name] = {
        'rotated_at' : datetime.now(timezone.utc).isoformat(),
        'note'       : 'Manually recorded rotation',
    }
    save_key_ages(ages)
    print(f'[OK] Recorded rotation for {key_name}')


def run_check(telegram=None) -> str:
    ages     = load_key_ages()
    now      = datetime.now(timezone.utc)
    lines    = []
    warnings = []

    for key_id, description in TRACKED_KEYS:
        entry      = ages.get(key_id, {})
        rotated_at = entry.get('rotated_at')

        if rotated_at:
            try:
                rotated_dt = datetime.fromisoformat(rotated_at.replace('Z', '+00:00'))
                age_days   = (now - rotated_dt).days
                status     = '[OK]' if age_days < ROTATION_DAYS else '[OVERDUE]'
                line       = f'{status} {key_id}: {age_days}d old (rotate every {ROTATION_DAYS}d)'
                lines.append(line)
                if age_days >= ROTATION_DAYS:
                    warnings.append(f'{key_id} ({description}) — {age_days}d since rotation')
            except Exception:
                lines.append(f'[WARN] {key_id}: invalid rotation date in key_ages.json')
        else:
            lines.append(f'[?] {key_id}: no rotation date recorded')
            warnings.append(f'{key_id} ({description}) — never recorded a rotation date')

    # Check if key_ages.json itself needs to be created
    if not ages:
        lines.insert(0,
            '[INFO] No key age records found. Run:\n'
            '  python scripts/check_key_age.py --record ALPACA\n'
            'after each key rotation to track age.\n'
        )

    report = (
        f'AlphaEdge Key Rotation Check\n'
        f'{"=" * 32}\n'
        f'{now.strftime("%Y-%m-%d %H:%M UTC")}\n\n'
    ) + '\n'.join(lines)

    if warnings:
        report += (
            f'\n\n⚠️ ROTATION NEEDED ({len(warnings)}):\n'
            + '\n'.join(f'  - {w}' for w in warnings)
            + f'\n\nRotate keys at their respective provider dashboards.\n'
            f'Then run: python scripts/check_key_age.py --record <KEY_NAME>'
        )
    else:
        report += '\n\n[OK] All keys within rotation window'

    if telegram and warnings:
        try:
            telegram.send_message(f'🔑 {report}')
        except Exception as e:
            logger.warning('Key age alert send failed: %s', e)

    return report


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if '--record' in sys.argv:
        idx = sys.argv.index('--record')
        if idx + 1 < len(sys.argv):
            record_rotation(sys.argv[idx + 1])
            sys.exit(0)
        else:
            print('Usage: python scripts/check_key_age.py --record <KEY_NAME>')
            sys.exit(1)

    telegram = None
    try:
        from monitoring.telegram_bot import TelegramBot
        telegram = TelegramBot()
    except Exception:
        pass

    print(run_check(telegram=telegram))
