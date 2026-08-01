"""
hardening/patch_live_sue.py
Wire SUE (earnings surprise) features into the live trading pipeline.

Injects 3 SUE features (sue_score, sue_days_since, sue_decay_weight)
into alpaca_live.py at two points:
  1. Before symbol loop: load earnings cache once for all watchlist symbols
  2. After add_all_features(): inject SUE columns and extend feature_names

These 3 features then compete in the existing SelectKBest(k=20) selection
alongside ~167 technical features. No architecture change required.

Run on VPS:
    cd /root/alpha_edge
    /root/alpha_edge/venv/bin/python3 hardening/patch_live_sue.py
"""
import sys
import os
import shutil
import re

TARGET = os.path.join(os.path.dirname(__file__), '..', 'alpaca_live.py')
TARGET = os.path.abspath(TARGET)

BACKUP = TARGET + '.bak_pre_live_sue'


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def save(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def patch_earnings_cache_load(content):
    """
    Insert earnings cache loader before the per-symbol loop.
    Anchors on the print statement that precedes the for loop.
    """
    OLD = "        print(f'\\n  Scoring {len(stock_data)} stocks...')"
    NEW = """\
        # -- SUE: load earnings cache once for all symbols --------------------
        _earnings_cache = {}
        try:
            from data.earnings_features import add_sue_features as _add_sue_fn
            from backtesting.data.loader import DataLoader as _EarningsLoader
            _el = _EarningsLoader(cache=True)
            _earnings_cache = _el.get_earnings_history(list(stock_data.keys()))
            logger.info(
                'SUE earnings cache loaded (%d symbols)', len(_earnings_cache)
            )
        except Exception as _sue_load_err:
            logger.warning(
                'SUE earnings cache failed -- SUE features disabled: %s',
                _sue_load_err
            )
        # -- end SUE cache load -----------------------------------------------
        print(f'\\n  Scoring {len(stock_data)} stocks...')"""

    if OLD not in content:
        print('[PATCH 1] ERROR: anchor line not found in alpaca_live.py')
        print('  Expected: ' + repr(OLD))
        return None

    if '_earnings_cache = {}' in content:
        print('[PATCH 1] Already applied -- skipping')
        return content

    patched = content.replace(OLD, NEW, 1)
    print('[PATCH 1] Earnings cache loader inserted before symbol loop')
    return patched


def patch_sue_inject(content):
    """
    Insert SUE feature injection after add_all_features() + get_feature_names().
    Anchors on the two-line block that always appears together.
    """
    OLD = (
        "                df           = engine.add_all_features(raw_df)\n"
        "                feature_names = engine.get_feature_names()"
    )
    NEW = """\
                df           = engine.add_all_features(raw_df)
                feature_names = engine.get_feature_names()
                # -- SUE: inject earnings surprise features -------------------
                if _earnings_cache:
                    try:
                        from data.earnings_features import add_sue_features as _add_sue_fn
                        df = _add_sue_fn(df, _earnings_cache.get(symbol))
                        _sue_cols = [
                            c for c in
                            ['sue_score', 'sue_days_since', 'sue_decay_weight']
                            if c in df.columns
                        ]
                        if _sue_cols:
                            feature_names = list(feature_names) + _sue_cols
                            logger.debug(
                                '%s: %d SUE features added to pool',
                                symbol, len(_sue_cols)
                            )
                    except Exception as _sue_err:
                        logger.debug(
                            'SUE inject failed for %s: %s', symbol, _sue_err
                        )
                # -- end SUE inject -------------------------------------------"""

    if OLD not in content:
        print('[PATCH 2] ERROR: anchor block not found in alpaca_live.py')
        print('  Expected the two-line add_all_features + get_feature_names block')
        return None

    if '_sue_cols' in content:
        print('[PATCH 2] Already applied -- skipping')
        return content

    patched = content.replace(OLD, NEW, 1)
    print('[PATCH 2] SUE inject block inserted after add_all_features()')
    return patched


def verify(content):
    checks = [
        ('_earnings_cache = {}', 'earnings cache init'),
        ('_sue_cols', 'SUE column selection'),
        ('feature_names = list(feature_names) + _sue_cols', 'feature_names extension'),
        ('add_sue_features', 'add_sue_features import'),
    ]
    all_ok = True
    for token, label in checks:
        if token in content:
            print(f'  [OK] {label}')
        else:
            print(f'  [FAIL] {label} — token not found: {repr(token)}')
            all_ok = False
    return all_ok


def main():
    print(f'Target: {TARGET}')

    if not os.path.exists(TARGET):
        print(f'ERROR: {TARGET} not found')
        sys.exit(1)

    # Backup
    shutil.copy2(TARGET, BACKUP)
    print(f'Backup: {BACKUP}')

    content = load(TARGET)

    # Apply patches
    content = patch_earnings_cache_load(content)
    if content is None:
        print('Patch 1 failed -- aborting, original file unchanged')
        sys.exit(1)

    content = patch_sue_inject(content)
    if content is None:
        print('Patch 2 failed -- aborting, restoring backup')
        shutil.copy2(BACKUP, TARGET)
        sys.exit(1)

    save(TARGET, content)
    print('\nVerification:')
    ok = verify(load(TARGET))

    if ok:
        print('\nSUCCESS: alpaca_live.py patched with SUE features')
        print('Next: restart alphaedge.service and check logs for')
        print('  "SUE earnings cache loaded" and "SUE features added to pool"')
    else:
        print('\nVERIFICATION FAILED -- restoring backup')
        shutil.copy2(BACKUP, TARGET)
        sys.exit(1)


if __name__ == '__main__':
    main()
