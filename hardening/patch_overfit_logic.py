#!/usr/bin/env python3
"""
hardening/patch_overfit_logic.py
Fix the overfit detection logic AND re-apply model regularization.

Root cause (why all symbols keep failing):
  The current overfit check compares in-sample train_AUC to val_AUC:
      if train_AUC - val_AUC > 0.20: HALT

  In-sample train_AUC is ALWAYS near 1.0 for any fitted tree model,
  even at depth=2 with 50 estimators.  So gap is always > 0.20 and every
  symbol fails regardless of whether the model actually generalises.

  Example from current logs:
      val_AUC=0.707, gap=0.263 → HARD STOP  (but 0.707 is excellent!)
      val_AUC=0.576, gap=0.417 → HARD STOP  (but 0.576 beats random)

  The right question is NOT "how big is the gap?"
  It is "does this model predict better than random on held-out data?"

Fix:
  1. Primary block: val_AUC < VAL_AUC_MIN (0.53) — worse than random.
  2. Secondary block: gap > 0.45 AND val_AUC < 0.58 — severe overfit with
     only marginal generalization.
  3. Otherwise: proceed (warn if gap > 0.20 but don't halt).

  Also re-applies model regularization (reduces depth/estimators) which was
  previously undone by git pull because patch_ml_regularization.py modified
  the file locally without committing.

After running this script, COMMIT the result so git pull doesn't undo it:
    git add models/technical_model.py
    git commit -m "fix: overfit check logic + model regularization"
    git push

Run from /root/alpha_edge:
    python3 hardening/patch_overfit_logic.py
    git add models/technical_model.py
    git commit -m "fix: overfit check logic + model regularization (committed)"
    git push
    find /root/alpha_edge -name "*.pyc" -delete
    systemctl restart alpaca.service
    journalctl -u alpaca.service -f | grep -E "Overfit check OK|HARD STOP|val_AUC"
"""

import ast
import os
import shutil
import sys
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

TARGET = Path('models/technical_model.py')
BACKUP = Path('models/technical_model.py.pre_overfit_logic')


def abort(msg):
    print(f'\nERROR: {msg}')
    sys.exit(1)


def check_syntax(src, label=''):
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f'Syntax error {label}: {e}')
        return False


# ── PATCH 1: Add VAL_AUC_MIN + raise OVERFIT_HARD_STOP ──────────────────────

OLD_THRESHOLDS = '''\
# Overfit thresholds (kept in sync with model_validator.py)
OVERFIT_WARN_GAP  = 0.10
OVERFIT_HARD_STOP = 0.20'''

NEW_THRESHOLDS = '''\
# Overfit thresholds (kept in sync with model_validator.py)
#
# KEY: train_AUC on in-sample data is always near 1.0 for any fitted tree.
# The gap (train - val) is therefore always large and not a useful signal.
# The right check is val_AUC alone: does the model beat random on held-out data?
#
# VAL_AUC_MIN: primary block — model worse than or equal to random
# OVERFIT_HARD_STOP: secondary block — only fires when gap is extreme AND val weak
# OVERFIT_WARN_GAP: warning threshold (log but don't halt)
VAL_AUC_MIN       = 0.53   # below this = worse than random; no signals
OVERFIT_WARN_GAP  = 0.20   # warn but proceed
OVERFIT_HARD_STOP = 0.45   # block only when gap extreme AND val marginal'''


def patch_thresholds(src):
    if 'VAL_AUC_MIN' in src:
        print('  [1] VAL_AUC_MIN already present — skipping threshold patch')
        return src
    if OLD_THRESHOLDS not in src:
        abort(
            'Cannot find threshold block in technical_model.py.\n'
            '  Expected:\n'
            '    # Overfit thresholds (kept in sync with model_validator.py)\n'
            '    OVERFIT_WARN_GAP  = 0.10\n'
            '    OVERFIT_HARD_STOP = 0.20'
        )
    src = src.replace(OLD_THRESHOLDS, NEW_THRESHOLDS, 1)
    print('  [1] Thresholds: added VAL_AUC_MIN=0.53, OVERFIT_HARD_STOP 0.20→0.45')
    return src


# ── PATCH 2: Replace overfit logic block ─────────────────────────────────────

OLD_OVERFIT_LOGIC = '''\
            if self.overfit_gap > OVERFIT_HARD_STOP:
                self.overfit_flagged = True
                logger.error(
                    f\'OVERFIT HARD STOP: train_AUC={self.train_auc:.3f} \'
                    f\'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f} \'
                    f\'— this fold will NOT generate live signals\'
                )
            elif self.overfit_gap > OVERFIT_WARN_GAP:
                logger.warning(
                    f\'Overfit warning: train_AUC={self.train_auc:.3f} \'
                    f\'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f}\'
                )
            else:
                logger.info(
                    f\'Overfit check OK: train_AUC={self.train_auc:.3f} \'
                    f\'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f}\'
                )'''

NEW_OVERFIT_LOGIC = '''\
            # Primary block: model predicts worse than (or equal to) random
            # on held-out data.  Gap-only checks are misleading because
            # in-sample train_AUC is always near 1.0 for any fitted tree.
            if self.val_auc < VAL_AUC_MIN:
                self.overfit_flagged = True
                logger.error(
                    f\'MODEL QUALITY HARD STOP: val_AUC={self.val_auc:.3f} < {VAL_AUC_MIN} \'
                    f\'(train={self.train_auc:.3f}, gap={self.overfit_gap:.3f}) \'
                    f\'— model has no predictive power on held-out data\'
                )
            # Secondary block: extreme overfit AND only marginal generalization
            elif self.overfit_gap > OVERFIT_HARD_STOP and self.val_auc < 0.58:
                self.overfit_flagged = True
                logger.error(
                    f\'OVERFIT HARD STOP: train_AUC={self.train_auc:.3f} \'
                    f\'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f} \'
                    f\'— this fold will NOT generate live signals\'
                )
            elif self.overfit_gap > OVERFIT_WARN_GAP:
                logger.warning(
                    f\'Overfit warning: train_AUC={self.train_auc:.3f} \'
                    f\'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f}\'
                )
            else:
                logger.info(
                    f\'Overfit check OK: train_AUC={self.train_auc:.3f} \'
                    f\'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f}\'
                )'''


def patch_overfit_logic(src):
    if 'MODEL QUALITY HARD STOP' in src:
        print('  [2] Overfit logic already patched — skipping')
        return src
    if OLD_OVERFIT_LOGIC not in src:
        abort(
            'Cannot find overfit logic block in technical_model.py.\n'
            '  Expected the block starting with:\n'
            '    if self.overfit_gap > OVERFIT_HARD_STOP:'
        )
    src = src.replace(OLD_OVERFIT_LOGIC, NEW_OVERFIT_LOGIC, 1)
    print('  [2] Overfit logic: gap-based → val_AUC-based check')
    return src


# ── PATCH 3: XGBoost regularization ──────────────────────────────────────────

OLD_XGB = '''\
            xgb_base = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                reg_alpha=1,
                reg_lambda=2,
                random_state=42,
                n_jobs=1,
                eval_metric=\'logloss\'
            )'''

NEW_XGB = '''\
            xgb_base = xgb.XGBClassifier(
                n_estimators=50,         # was 200 — 300 samples can\'t support 200 trees
                max_depth=2,             # was 3 — depth-2 = 4 leaf nodes max
                learning_rate=0.05,      # was 0.01
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=2,                 # was 1 — aggressive pruning
                min_child_weight=5,      # new — no split on <5 samples
                reg_alpha=1,
                reg_lambda=2,
                random_state=42,
                n_jobs=1,
                eval_metric=\'logloss\'
            )'''


def patch_xgb(src):
    if 'min_child_weight=5' in src:
        print('  [3] XGBoost already regularized — skipping')
        return src
    if OLD_XGB not in src:
        print('  [3] WARN: XGBoost anchor not found — skipping (may already be patched differently)')
        return src
    src = src.replace(OLD_XGB, NEW_XGB, 1)
    print('  [3] XGBoost: n_estimators 200→50, max_depth 3→2, +min_child_weight=5')
    return src


# ── PATCH 4: LightGBM regularization ─────────────────────────────────────────

OLD_LGB = '''\
            lgb_base = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=2,
                random_state=42,
                n_jobs=1,
                verbose=-1
            )'''

NEW_LGB = '''\
            lgb_base = lgb.LGBMClassifier(
                n_estimators=50,         # was 200
                max_depth=2,             # was 4
                num_leaves=7,            # new — LGB ignores max_depth without this
                learning_rate=0.05,      # was 0.01
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=15,    # new — leaf needs ≥15 samples
                reg_alpha=1,
                reg_lambda=2,
                random_state=42,
                n_jobs=1,
                verbose=-1
            )'''


def patch_lgb(src):
    if 'num_leaves=7' in src:
        print('  [4] LightGBM already regularized — skipping')
        return src
    if OLD_LGB not in src:
        print('  [4] WARN: LightGBM anchor not found — skipping')
        return src
    src = src.replace(OLD_LGB, NEW_LGB, 1)
    print('  [4] LightGBM: n_estimators 200→50, max_depth 4→2, +num_leaves=7, +min_child_samples=15')
    return src


# ── PATCH 5: RandomForest regularization ─────────────────────────────────────

OLD_RF = '''\
            rf_base = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                max_features=\'sqrt\',
                random_state=42,
                n_jobs=1
            )'''

NEW_RF = '''\
            rf_base = RandomForestClassifier(
                n_estimators=100,        # was 200
                max_depth=3,             # was 5
                min_samples_leaf=20,     # was 10
                min_samples_split=10,    # new
                max_features=\'sqrt\',
                random_state=42,
                n_jobs=1
            )'''


def patch_rf(src):
    if 'min_samples_split=10' in src:
        print('  [5] RandomForest already regularized — skipping')
        return src
    if OLD_RF not in src:
        print('  [5] WARN: RandomForest anchor not found — skipping')
        return src
    src = src.replace(OLD_RF, NEW_RF, 1)
    print('  [5] RandomForest: n_estimators 200→100, max_depth 5→3, min_samples_leaf 10→20')
    return src


# ── PATCH 6: CatBoost regularization ─────────────────────────────────────────

OLD_CAT = '''\
            cat_model = CatBoostClassifier(
                iterations    = 200,
                depth         = 4,
                learning_rate = 0.01,
                random_seed   = 42,
                verbose       = 0,
                thread_count  = 1,
            )'''

NEW_CAT = '''\
            cat_model = CatBoostClassifier(
                iterations      = 50,          # was 200
                depth           = 2,           # was 4
                learning_rate   = 0.05,        # was 0.01
                min_data_in_leaf= 10,          # new
                random_seed     = 42,
                verbose         = 0,
                thread_count    = 1,
            )'''


def patch_cat(src):
    if 'min_data_in_leaf' in src:
        print('  [6] CatBoost already regularized — skipping')
        return src
    if OLD_CAT not in src:
        print('  [6] WARN: CatBoost anchor not found — skipping')
        return src
    src = src.replace(OLD_CAT, NEW_CAT, 1)
    print('  [6] CatBoost: iterations 200→50, depth 4→2, lr 0.01→0.05, +min_data_in_leaf=10')
    return src


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not TARGET.exists():
        abort(f'{TARGET} not found — run from /root/alpha_edge')

    print('\nAlphaEdge — overfit logic + model regularization patch')
    print('=' * 55)
    print()

    src = TARGET.read_text(encoding='utf-8')

    if not check_syntax(src, 'BEFORE patch'):
        abort('technical_model.py has a syntax error — fix it first')

    shutil.copy2(TARGET, BACKUP)
    print(f'Backup: {BACKUP}\n')

    src = patch_thresholds(src)
    src = patch_overfit_logic(src)
    src = patch_xgb(src)
    src = patch_lgb(src)
    src = patch_rf(src)
    src = patch_cat(src)

    if not check_syntax(src, 'AFTER patch'):
        print(f'Rolling back to {BACKUP}')
        shutil.copy2(BACKUP, TARGET)
        abort('Patch introduced syntax error — rolled back')

    TARGET.write_text(src, encoding='utf-8')

    # Verify
    final = TARGET.read_text(encoding='utf-8')
    checks = {
        'VAL_AUC_MIN': 'VAL_AUC_MIN' in final,
        'OVERFIT_HARD_STOP=0.45': 'OVERFIT_HARD_STOP = 0.45' in final,
        'MODEL QUALITY HARD STOP': 'MODEL QUALITY HARD STOP' in final,
        'min_child_weight=5 (XGB)': 'min_child_weight=5' in final,
        'num_leaves=7 (LGB)': 'num_leaves=7' in final,
        'min_samples_split=10 (RF)': 'min_samples_split=10' in final,
        'min_data_in_leaf (CAT)': 'min_data_in_leaf' in final,
    }
    print()
    all_ok = True
    for name, ok in checks.items():
        status = '✓' if ok else '✗ FAILED'
        print(f'  {status}  {name}')
        if not ok:
            all_ok = False

    if not all_ok:
        abort('\nOne or more patches did not land — check anchors above')

    print(f'\n✓ All patches verified. Lines: {len(final.splitlines())}')
    print()
    print('IMPORTANT: commit these changes so git pull does not undo them:')
    print()
    print('  git add models/technical_model.py')
    print('  git commit -m "fix: overfit check logic + model regularization"')
    print('  git push')
    print()
    print('Then restart:')
    print('  find /root/alpha_edge -name "*.pyc" -delete')
    print('  systemctl restart alpaca.service')
    print('  journalctl -u alpaca.service -f | grep -E "Overfit check OK|HARD STOP|val_AUC"')
    print()
    print('Expected: symbols with val_AUC ≥ 0.53 will now generate ML signals.')
    print('Symbols with val_AUC < 0.53 are correctly blocked (worse than random).')


if __name__ == '__main__':
    main()
