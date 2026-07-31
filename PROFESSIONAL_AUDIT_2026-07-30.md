# AlphaEdge — Professional Quant/Security Audit
**Date:** 2026-07-30  
**Auditor:** Independent review (Claude Sonnet)  
**Scope:** Trading logic, ML pipeline, risk management, security, backtest validity  
**Verdict: CONDITIONALLY DEPLOYABLE for paper trading. NOT ready for live capital.**

---

## Executive Summary

AlphaEdge has solid structural bones and several institutional-grade components. The risk management layer, circuit breaker, and fill model are genuinely well-designed. However, three unresolved issues prevent live-capital recommendation: (1) ML models are currently overfit and have never produced a live trade, (2) a GitHub PAT is committed to the repository in plaintext, and (3) the veto agent silently auto-approves all trades when GROQ_API_KEY is absent, defeating its purpose.

---

## CRITICAL (P0) — Must fix before live capital

### C1 — GitHub PAT exposed in committed file
**File:** `CODEBASE_AUDIT_2026-07-24.md:36`  
**Evidence:**
```
https://ghp_[REDACTED—token revoked]@github.com/jpfashionhub888/alpha_edge.git
```
**Risk:** This token is in git history and visible to anyone with repo access. If the repo is public or becomes public, it grants full write access to the account.  
**Fix:** Revoke this token immediately at github.com/settings/tokens. Scrub the file. Consider `git filter-repo` to remove it from history.

---

### C2 — Veto agent silently approves when disabled
**File:** `veto_agent.py:63-69`  
**Evidence:**
```python
if not self.enabled:
    return {'decision': 'APPROVE', 'reason': 'Veto agent disabled (no GROQ_API_KEY)', ...}
```
**Risk:** If GROQ_API_KEY is not set (e.g. after a VPS rebuild, env file missing, secret rotation), ALL trades bypass the AI review layer with no operator alert. The system treats "key missing" as "approved to trade" — exactly backwards.  
**Fix:** Fail-closed. Return VETO (not APPROVE) when disabled, OR send a Telegram alert when disabled and document it as a known operational degradation, not a silent pass.

---

### C3 — ML models have never produced a live signal above threshold
**Evidence:** `paper_trades_stocks_only.json` has empty `trade_history` since system inception. `model_auc.json` last updated 2026-07-28 with val_AUC as low as 0.12–0.52 (severe overfit).  
**Risk:** Deploying live capital on models that have never cleared the veto gate is operationally unvalidated. The overfit guard (`OVERFIT_HARD_STOP = 0.20`) is correct, but it means most models are being skipped — so the system effectively has no working signal engine until Sunday's retrain.  
**Fix:** Run at least 2 weeks of paper trading after a clean retrain (val_AUC > 0.65 across >50% of symbols) before live capital consideration.

---

## HIGH (P1) — Fix before extending position limits

### H1 — Train/val split uses last 20% of data, not walk-forward
**File:** `models/technical_model.py:65-68`  
**Evidence:**
```python
val_size  = max(30, int(len(X) * 0.20))
X_val_raw = X.iloc[-val_size:]
X_tr      = X.iloc[:-val_size]
```
**Risk:** This is a single train/val split, not walk-forward cross-validation. The model trains on all historical data except the last 20%, then validates on that fixed window. This overstates generalisability — a model can learn the specific regime of the last 20% period without generalising to future regimes. Industry standard is purged k-fold or expanding-window walk-forward.  
**Severity:** This is the root cause of the overfit problem observed in production.  
**Fix:** Implement TimeSeriesSplit (sklearn) with purge gap of 5 bars between folds. `backtesting/backtest/walk_forward.py` already has the infrastructure — integrate it into `TechnicalPredictor.train()`.

### H2 — Target variable uses 5-day forward return (survivorship not controlled)
**File:** `data/feature_engine.py:421-434`  
**Evidence:**
```python
forward_period = 5
df['future_return'] = df['close'].pct_change(forward_period).shift(-forward_period)
df['target'] = (df['future_return'] > 0).astype(int)
```
**Risk:** Binary classification "did price go up in 5 days?" on raw close prices is a weak, noisy label. At 5 days, ~50% of labels will be positive by chance in any trending market. The system is essentially predicting a coin flip. Professional shops use risk-adjusted returns, excess returns over SPY, or minimum magnitude thresholds (e.g., >0.5% net of cost).  
**Fix:** Add minimum return threshold: `(df['future_return'] > 0.005)` to filter out noise. Also add a short-side label: `abs(future_return) > threshold` as a tradeable signal filter.

### H3 — Position monitor uses last known price from Alpaca, not real-time quote
**File:** `alpaca_live.py:760-775`  
**Evidence:** `_check_stops_targets()` calls `self.broker.get_positions()` which returns Alpaca's cached position price, not a live market feed. During rapid moves, this price can be 60 seconds stale (monitor interval).  
**Risk:** Stop losses and take profits can miss by a full 60-second price move during volatile sessions. A 1% gap move on a $1,500 position = $15 missed per trade.  
**Fix:** For stop/target decisions, use a fresh quote call (`get_latest_quote(symbol)`) rather than position data. Or reduce monitor interval to 30 seconds during market hours.

---

## MEDIUM (P2) — Fix before scaling

### M1 — Kelly Criterion uses trade PnL in dollars, not % returns
**File:** `risk/position_sizer.py:318-323`  
**Evidence:**
```python
wins   = [t['pnl'] for t in sells if t['pnl'] > 0]
losses = [abs(t['pnl']) for t in sells if t['pnl'] <= 0]
return {
    'avg_win' : sum(wins)   / len(wins),
    'avg_loss': sum(losses) / len(losses),
```
**Risk:** Kelly formula requires `avg_win` and `avg_loss` as **return percentages** (e.g., 0.04 = 4%), not dollar amounts. Passing dollar PnL produces wildly incorrect Kelly fractions that scale with position size, not edge. A $120 win on a $1,500 position = 8% return, but a $120 win on a $500 position = 24% return — same dollar figure, very different Kelly sizing implication.  
**Fix:** Divide `pnl` by position cost basis: `t['pnl'] / t.get('cost_basis', 1)` or store `pnl_pct` in closed trades and use that.

### M2 — Correlation filter is a hardcoded lookup table, not computed
**File:** `risk/position_sizer.py:244-283`  
**Evidence:** `PEER_GROUPS` dict with manually assigned peer clusters.  
**Risk:** Lookup tables go stale (e.g., META/GOOGL correlation was near-zero pre-2022, near-0.9 post-2022). SOFI and HOOD being in the same cluster is a 2021 meme-stock assumption, not a structural relationship. The table has no mechanism to update.  
**Fix:** This is acceptable short-term. After 90 days of live data, replace with a rolling 60-day correlation matrix computed from actual daily returns. Add a staleness check to warn if PEER_GROUPS hasn't been reviewed in >180 days.

### M3 — MetaLabeler loads from wrong path
**File:** `alpaca_live.py:424`  
**Evidence:**
```python
_meta_path = os.path.join('model_cache', f'meta_{symbol}.pkl')
```
**Risk:** All other models are cached in `cache/models/*.pkl` (confirmed by health check). `model_cache/` is a separate directory and is not excluded from the correlation filter's `.gitignore` `!model_cache/*` rule — meaning MetaLabeler models are always missing, and the MetaLabeler gate is always skipped silently. This means a potentially useful false-positive filter is never active.  
**Fix:** Change path to `os.path.join('cache', 'models', f'meta_{symbol}.pkl')` or wherever `save_models()` actually writes.

### M4 — Circuit breaker daily reset on first scan of new day, not midnight
**File:** `risk_circuit_breaker.py:108-111`  
**Evidence:**
```python
today = now.strftime('%Y-%m-%d')
if self.state.get('daily_start') != today:
    self.state['daily_start']     = today
    self.state['daily_start_val'] = current_value
```
**Risk:** The daily loss limit resets when the first scan runs (16:15 ET), not at market open. A position entered at 9:30 AM that drops 4% by 16:15 doesn't trip the 5% daily loss limit because `daily_start_val` was set at the PREVIOUS day's close. Only losses occurring between two consecutive 16:15 scans count.  
**Practical impact:** Low, since the system only scans once daily. But if scan frequency increases, this becomes a gap.  
**Fix:** Reset daily tracking at midnight or market open (9:30 ET), not at scan time.

---

## LOW (P3) — Good to fix, not urgent

### L1 — No broker API rate limit handling
**File:** `execution/alpaca_broker.py` (not read in detail, inferred)  
**Risk:** Alpaca paper API has rate limits. Scanning 41 symbols, fetching positions, placing orders in rapid succession can trigger 429 errors. Current code catches exceptions broadly but doesn't implement exponential backoff with jitter.

### L2 — LSTM trained on full data, not split
**File:** `models/technical_model.py:228-242`  
**Evidence:** `lstm.train(X, y)` — passes full X/y, not the split `X_tr/y_tr` used for tree models.  
**Risk:** LSTM sees the validation period during training — its contribution to the ensemble has guaranteed lookahead on the val set. Minor in practice since LSTM weight is 0.8 vs 1.0, but it inflates the val AUC measurement.

### L3 — `_run_scan()` re-imports all modules on every scan
**File:** `alpaca_live.py:250-274`  
**Evidence:** All `from X import Y` statements inside `_run_scan()`, called once daily.  
**Risk:** No practical performance issue (called once per day). But it prevents early detection of import errors — failures only surface at scan time, not startup.

### L4 — Backtest fill model assumes unlimited liquidity for small account
**File:** `backtesting/engine/fill_model.py:96-101`  
**Risk:** ADV proxy uses `volume * arrival_price`. For a $10,000 account and large-cap stocks with ADV > $1B, `participation_rate` is always near 0, making market impact ~0. This is correct but means the cost model is optimistic — real small-account fills on illiquid names (MARA, SOFI, HOOD) will be worse.

---

## What the self-audit (deep_audit.py) missed

- The PAT in `CODEBASE_AUDIT_2026-07-24.md` (plain text file, not Python)
- Veto agent's fail-open behaviour when key is absent
- Kelly Criterion using dollar PnL instead of percentage returns
- LSTM trained on full dataset instead of train split
- MetaLabeler path mismatch (model cache directory mismatch)
- Target variable quality (binary 5-day, no minimum magnitude threshold)
- Single train/val split vs walk-forward cross-validation

The self-audit caught: syntax errors, import failures, service names, bare excepts, and structural wiring. It does not and cannot catch: logic correctness, statistical methodology, or operational edge cases.

---

## Readiness Assessment

| Dimension | Status | Confidence |
|-----------|--------|------------|
| System stability | ✅ Good | High |
| Risk controls | ✅ Good | High |
| Security | ⚠️ Token exposed | P0 action needed |
| ML methodology | ⚠️ Overfit, weak labels | Fix before live |
| Backtest validity | ✅ Fill model solid | Medium |
| Position sizing | ⚠️ Kelly input error | Fix before scaling |
| Live signal quality | ❌ 0 trades produced | Unvalidated |

**Recommendation:** Continue paper trading. Fix C1 (token), C2 (veto fail-open), M1 (Kelly inputs), M3 (MetaLabeler path) immediately. H1 and H2 (ML methodology) address after Sunday's retrain confirms whether val_AUC improves. Do not deploy live capital until the system has completed at least 10 live paper trades with documented win/loss attribution.
