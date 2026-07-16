#!/usr/bin/env python3
"""
hardening/patch_technical_model.py
Patches models/technical_model.py to fix the p >> n overfit problem.

Root cause: 167 features trained on ~144 samples gives p/n ≈ 1.16 —
guaranteed memorisation in all tree models regardless of regularisation.

Fix: SelectKBest(mutual_info_classif, k=20) reduces to p/n ≈ 0.14.
Selector is fitted on train split only — no data leakage.

Run from /root/alpha_edge:
    python3 hardening/patch_technical_model.py
"""

import ast
import os
import shutil
import sys
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

TARGET = Path('models/technical_model.py')
BACKUP = Path('models/technical_model.py.pre_feature_selection')


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


# ── PATCH 1: add pandas + SelectKBest imports ────────────────────────────────

IMPORT_ANCHOR = 'import numpy as np'

NEW_IMPORTS = '''\
import numpy as np
import pandas as pd
'''

SELECTKBEST_IMPORT_ANCHOR = 'from sklearn.calibration import CalibratedClassifierCV'

SELECTKBEST_IMPORT = '''\
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
'''


def patch_imports(src):
    if 'SelectKBest' in src:
        print('  [1] SelectKBest import already present — skipping')
        return src

    # Add pandas import
    if 'import pandas as pd' not in src:
        src = src.replace(IMPORT_ANCHOR, NEW_IMPORTS, 1)
        print('  [1a] Added: import pandas as pd')

    # Add SelectKBest import
    src = src.replace(SELECTKBEST_IMPORT_ANCHOR, SELECTKBEST_IMPORT, 1)
    print('  [1b] Added: from sklearn.feature_selection import SelectKBest, mutual_info_classif')
    return src


# ── PATCH 2: add MAX_FEATURES class variable ─────────────────────────────────

CLASS_ANCHOR = '    def __init__(self, use_lstm=True):'

MAX_FEATURES_BLOCK = '''\
    # Maximum features fed to tree models.
    # Rule: n_samples / 8-10.  With ~144 training rows, 20 keeps p/n ≈ 0.14.
    MAX_FEATURES = 20

    def __init__(self, use_lstm=True):
'''


def patch_class_var(src):
    if 'MAX_FEATURES' in src:
        print('  [2] MAX_FEATURES already present — skipping')
        return src
    if CLASS_ANCHOR not in src:
        abort('Cannot find __init__ anchor in technical_model.py')
    src = src.replace(CLASS_ANCHOR, MAX_FEATURES_BLOCK, 1)
    print('  [2] Added: MAX_FEATURES = 20 class variable')
    return src


# ── PATCH 3: add selected_features / feature_selector to __init__ ────────────

INIT_ANCHOR = '        self.feature_names= []'

INIT_PATCH = '''\
        self.feature_names    = []
        self.selected_features= []    # post-selection feature names
        self.feature_selector = None  # fitted SelectKBest
'''


def patch_init(src):
    if 'selected_features' in src:
        print('  [3] selected_features already in __init__ — skipping')
        return src
    if INIT_ANCHOR not in src:
        abort('Cannot find feature_names anchor in __init__')
    src = src.replace(INIT_ANCHOR, INIT_PATCH, 1)
    print('  [3] Added: selected_features + feature_selector to __init__')
    return src


# ── PATCH 4: add feature selection block in train() ──────────────────────────

SELECTION_ANCHOR = '        min_class = y_tr.value_counts().min()'

SELECTION_BLOCK = '''\
        # ── Feature selection (p >> n guard) ─────────────────────────────────
        # With 167 features on ~144 training rows, p/n ≈ 1.16 → guaranteed
        # memorisation.  Reduce to MAX_FEATURES using mutual information so
        # p/n < 0.15.  Selector is fitted on train only (no data leakage).
        _n_raw = X_tr.shape[1]
        _k = min(self.MAX_FEATURES, _n_raw)
        if _n_raw > _k:
            try:
                _sel = SelectKBest(mutual_info_classif, k=_k)
                _sel.fit(X_tr, y_tr)
                _mask  = _sel.get_support()
                _cols  = [c for c, s in zip(X_tr.columns, _mask) if s]
                X_tr        = X_tr[_cols]
                X_val_raw   = X_val_raw[_cols]
                self.feature_selector  = _sel
                self.selected_features = _cols
                logger.info(
                    f'Feature selection: {_n_raw} → {_k} features '
                    f'(p/n: {_n_raw / max(len(X_tr), 1):.2f} → '
                    f'{_k / max(len(X_tr), 1):.2f})'
                )
            except Exception as _e:
                logger.warning(
                    f'Feature selection failed ({_e}) — using all {_n_raw} features'
                )
                self.feature_selector  = None
                self.selected_features = list(X_tr.columns)
        else:
            self.selected_features = list(X_tr.columns)

        min_class = y_tr.value_counts().min()
'''


def patch_feature_selection(src):
    if 'Feature selection' in src:
        print('  [4] Feature selection block already present — skipping')
        return src
    if SELECTION_ANCHOR not in src:
        abort('Cannot find min_class anchor in train()')
    src = src.replace(SELECTION_ANCHOR, SELECTION_BLOCK, 1)
    print('  [4] Added: SelectKBest feature selection block in train()')
    return src


# ── PATCH 5: add _apply_feature_selection() helper method ────────────────────

PREDICT_ANCHOR = '    def predict(self, X):'

APPLY_FS_METHOD = '''\
    def _apply_feature_selection(self, X):
        """Apply the same feature selection that was used during training."""
        if self.selected_features and isinstance(X, pd.DataFrame):
            available = [c for c in self.selected_features if c in X.columns]
            if len(available) == len(self.selected_features):
                return X[self.selected_features]
            if len(available) > 0:
                logger.warning(
                    f'Feature mismatch: expected {len(self.selected_features)}, '
                    f'found {len(available)} — using available subset'
                )
                return X[available]
        return X

    def predict(self, X):
'''


def patch_apply_fs(src):
    if '_apply_feature_selection' in src:
        print('  [5] _apply_feature_selection already present — skipping')
        return src
    if PREDICT_ANCHOR not in src:
        abort('Cannot find predict() anchor in technical_model.py')
    src = src.replace(PREDICT_ANCHOR, APPLY_FS_METHOD, 1)
    print('  [5] Added: _apply_feature_selection() helper method')
    return src


# ── PATCH 6: call _apply_feature_selection in predict() ──────────────────────

PREDICT_TRAINED_CHECK = '''\        if not self.trained:
            raise Exception("Model not trained yet")

        all_predictions = []
        weights = []'''

PREDICT_WITH_FS = '''\        if not self.trained:
            raise Exception("Model not trained yet")

        X = self._apply_feature_selection(X)

        all_predictions = []
        weights = []'''


def patch_predict_call(src):
    # We need to patch the first occurrence (predict) and second
    # (predict_with_agreement) separately
    count = src.count('        all_predictions = []\n        weights = []')
    if count == 0:
        print('  [6] WARN: predict() body anchor not found — skipping')
        return src

    # First occurrence: predict()
    src = src.replace(
        '        if not self.trained:\n'
        '            raise Exception("Model not trained yet")\n'
        '\n'
        '        all_predictions = []\n'
        '        weights = []',
        '        if not self.trained:\n'
        '            raise Exception("Model not trained yet")\n'
        '\n'
        '        X = self._apply_feature_selection(X)\n'
        '\n'
        '        all_predictions = []\n'
        '        weights = []',
        1  # only first occurrence
    )
    print('  [6] Applied: _apply_feature_selection() call in predict()')
    return src


# ── PATCH 7: call _apply_feature_selection in predict_with_agreement() ───────

PWA_ANCHOR = (
    '        if not self.trained:\n'
    '            raise Exception("Model not trained yet")\n'
    '\n'
    '        all_predictions = []\n'
    '\n'
    '        for name, model in self.models.items():'
)

PWA_PATCHED = (
    '        if not self.trained:\n'
    '            raise Exception("Model not trained yet")\n'
    '\n'
    '        X = self._apply_feature_selection(X)\n'
    '\n'
    '        all_predictions = []\n'
    '\n'
    '        for name, model in self.models.items():'
)


def patch_pwa_call(src):
    if PWA_ANCHOR not in src:
        print('  [7] WARN: predict_with_agreement() anchor not found — skipping')
        return src
    src = src.replace(PWA_ANCHOR, PWA_PATCHED, 1)
    print('  [7] Applied: _apply_feature_selection() call in predict_with_agreement()')
    return src


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not TARGET.exists():
        abort(f'{TARGET} not found — run from /root/alpha_edge')

    print('\nAlphaEdge — technical_model.py feature selection patch')
    print('=' * 55)

    src = TARGET.read_text(encoding='utf-8')

    if not check_syntax(src, 'BEFORE patch'):
        abort('technical_model.py has a syntax error — fix it first')

    shutil.copy2(TARGET, BACKUP)
    print(f'Backup saved → {BACKUP}\n')

    src = patch_imports(src)
    src = patch_class_var(src)
    src = patch_init(src)
    src = patch_feature_selection(src)
    src = patch_apply_fs(src)
    src = patch_predict_call(src)
    src = patch_pwa_call(src)

    if not check_syntax(src, 'AFTER patch'):
        print(f'Rolling back to {BACKUP}')
        shutil.copy2(BACKUP, TARGET)
        abort('Patch introduced syntax error — rolled back')

    TARGET.write_text(src, encoding='utf-8')
    print(f'\n✓ All patches applied. Lines: {len(src.splitlines())}')
    print('\nNext:')
    print('  systemctl restart alpaca.service')
    print('  journalctl -u alpaca.service -f | grep -E '
          '"Feature selection|OVERFIT|Overfit check OK"')


if __name__ == '__main__':
    main()
