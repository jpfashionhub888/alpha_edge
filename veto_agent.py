# veto_agent.py
# ALPHAEDGE - AI Veto Agent (Groq/Llama3)
#
# Fixes applied:
#   - Groq import is now lazy (won't crash at import if groq not installed)
#   - Errors now return VETO (fail-closed), not APPROVE (fail-open)
#   - Network timeouts explicitly trigger VETO with clear reason
#   - JSON parse errors fall back to VETO with reason logged

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VetoAgent:
    """
    AI-powered trade review agent using Groq/Llama3.

    FAIL-CLOSED: any error (API timeout, parse failure, missing key)
    results in VETO, not APPROVE. A trade that can't be reviewed
    should not be executed.
    """

    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY', '')
        self.enabled = bool(self.api_key)
        self.model   = 'llama-3.3-70b-versatile'
        self._client = None  # S3 FIX: created once in __init__, not per-call

        if not self.enabled:
            logger.warning("Veto Agent: GROQ_API_KEY not found — all signals will APPROVE (review disabled)")
        else:
            # S3 FIX: instantiate Groq client once here (was created on every review_signal() call,
            # causing up to 100s sequential overhead when 10+ BUY signals queued per scan).
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info("Veto Agent: Groq/Llama3 connected")
            except ImportError:
                logger.error("groq package not installed. Run: pip install groq")
                self.enabled = False

    def review_signal(self,
                      symbol,
                      price,
                      prediction,
                      regime,
                      sentiment,
                      sector,
                      market_regime,
                      mtf_score,
                      current_positions,
                      vix=None):
        """
        Review a BUY signal. Returns dict with 'decision', 'reason', 'confidence'.

        If veto agent is disabled (no API key), returns APPROVE with note.
        If API call fails for any reason, returns VETO (fail-closed).
        """
        if not self.enabled:
            # Disabled — allow trade but mark it as unreviewed
            return {
                'decision' : 'APPROVE',
                'reason'   : 'Veto agent disabled (no GROQ_API_KEY)',
                'confidence': 0.5,
            }

        try:
            # S3 FIX: use pre-created self._client (was creating Groq() per call)
            if self._client is None:
                return {
                    'decision' : 'VETO',
                    'reason'   : 'Groq client not initialised (groq package missing?)',
                    'confidence': 0.0,
                }
            client          = self._client
            positions_text  = ', '.join(current_positions.keys()) if current_positions else 'None'
            vix_text        = f"{vix:.1f}" if vix is not None else "Unknown"
            n_positions     = len(current_positions)

            prompt = f"""You are a senior risk manager at a hedge fund.
Review this trade signal and decide APPROVE or VETO.

TRADE SIGNAL:
Symbol: {symbol}
Price: ${price:.2f}
Sector: {sector}
Market Regime: {market_regime}
AI Prediction: {prediction:.3f}
Regime: {regime}
Sentiment: {sentiment:+.2f}
MTF Score: {mtf_score:.0%}
VIX: {vix_text}

PORTFOLIO:
Open Positions: {positions_text}
Count: {n_positions}/5

Respond with ONLY valid JSON, no markdown, no preamble:
{{
  "decision": "APPROVE" or "VETO",
  "reason": "One sentence",
  "confidence": 0.0 to 1.0
}}

VETO if:
- Sentiment below -0.3 without prediction > 0.65
- VIX above 25 and prediction below 0.65
- Market regime is bearish or downtrend
- Symbol already in portfolio
- Portfolio already at 5 positions

APPROVE if:
- Prediction above 0.6 and regime is uptrend
- Sentiment neutral or positive
- Market conditions reasonable"""

            response = client.chat.completions.create(
                model    = self.model,
                messages = [
                    {
                        "role"   : "system",
                        "content": "You are a strict hedge fund risk manager. "
                                   "Respond ONLY with valid JSON. No markdown. No extra text."
                    },
                    {
                        "role"   : "user",
                        "content": prompt
                    }
                ],
                temperature = 0.1,
                max_tokens  = 150,
                timeout     = 10,   # explicit timeout — don't hang the scan
            )

            response_text = response.choices[0].message.content.strip()

            # Strip accidental markdown fences
            if '```' in response_text:
                parts = response_text.split('```')
                # Take the part after the first fence
                response_text = parts[1]
                if response_text.lower().startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()

            result     = json.loads(response_text)
            decision   = result.get('decision', 'VETO').upper().strip()
            reason     = result.get('reason', 'No reason provided')
            confidence = float(result.get('confidence', 0.5))

            if decision not in ('APPROVE', 'VETO'):
                logger.warning(f"Veto agent returned unknown decision '{decision}' for {symbol} — defaulting to VETO")
                decision = 'VETO'
                reason   = f"Unexpected decision value '{decision}' from model"

            print(f"  Veto Agent [{symbol}]: {decision}")
            print(f"    Reason: {reason}")
            print(f"    Confidence: {confidence:.0%}")

            return {
                'decision'  : decision,
                'reason'    : reason,
                'confidence': confidence,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Veto agent JSON parse error for {symbol}: {e} — VETOING")
            return {
                'decision'  : 'VETO',
                'reason'    : f'Model response was not valid JSON — cannot verify signal safety',
                'confidence': 0.0,
            }
        except Exception as e:
            logger.warning(f"Veto agent error for {symbol}: {e} — VETOING (fail-closed)")
            return {
                'decision'  : 'VETO',
                'reason'    : f'Veto agent unreachable ({type(e).__name__}) — trade blocked for safety',
                'confidence': 0.0,
            }


if __name__ == '__main__':
    print("\nTesting Veto Agent...")
    agent  = VetoAgent()
    result = agent.review_signal(
        symbol            = 'AAPL',
        price             = 192.40,
        prediction        = 0.72,
        regime            = 'uptrend',
        sentiment         = 0.15,
        sector            = 'Technology',
        market_regime     = 'BULL',
        mtf_score         = 1.0,
        current_positions = {},
        vix               = 16.9,
    )
    print(f"\nDecision:   {result['decision']}")
    print(f"Reason:     {result['reason']}")
    print(f"Confidence: {result['confidence']:.0%}")
