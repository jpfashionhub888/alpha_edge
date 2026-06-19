# veto_agent.py
# ALPHAEDGE - AI Veto Agent (Groq/Llama3)
#
# FIX P2-3: reason string in the unexpected-decision branch was built
#   AFTER decision was overwritten to 'VETO', so the message always
#   said "Unexpected decision value 'VETO'" instead of the actual
#   bad value received from the model. Fixed by capturing the original
#   value first.

import os
import json
import logging
import time
from datetime import datetime


logger = logging.getLogger(__name__)


class VetoAgent:
    """
    AI-powered trade review agent.
    Uses Groq/Llama3 to review BUY signals.
    """

    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY', '')
        self.enabled = bool(self.api_key)
        self.model   = 'llama-3.3-70b-versatile'

        if not self.enabled:
            print("  Veto Agent: GROQ_API_KEY not found")
        else:
            print("  Veto Agent: Groq/Llama3 connected ✅")

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

        if not self.enabled:
            return {
                'decision' : 'APPROVE',
                'reason'   : 'Veto agent disabled',
                'confidence': 0.5,
            }

        try:
            # Lazy import — won't crash startup if groq not installed
            try:
                from groq import Groq
            except ImportError:
                logger.error('groq package not installed — run: pip install groq')
                return {
                    'decision'  : 'VETO',
                    'reason'    : 'groq package not installed — cannot review signal',
                    'confidence': 0.0,
                }
            client = Groq(api_key=self.api_key)

            positions_text = ', '.join(current_positions.keys()) \
                             if current_positions else 'None'
            vix_text = f"{vix:.1f}" if vix else "Unknown"

            prompt = f"""You are a senior risk manager at a hedge fund.
Review this trade signal and decide APPROVE or VETO.

TRADE SIGNAL:
Symbol:       {symbol}
Price:        ${price:.2f}
Sector:       {sector}
Market Regime:{market_regime}
AI Prediction:{prediction:.3f}
Regime:       {regime}
Sentiment:    {sentiment:+.2f}
MTF Score:    {mtf_score:.0%}
VIX:          {vix_text}

PORTFOLIO:
Open Positions:{positions_text}
Count:         {len(current_positions)}/5

Respond with ONLY this JSON:
{{
    "decision":   "APPROVE" or "VETO",
    "reason":     "One sentence",
    "confidence": 0.0 to 1.0
}}

VETO if:
- Sentiment below -0.3 without strong technicals
- VIX above 25 and prediction below 0.65
- Market regime is bearish
- Stock already in portfolio

APPROVE if:
- Prediction above 0.6
- Sentiment neutral or positive
- Market conditions reasonable"""

            # Fix 4.3: Exponential backoff retry before fail-closed VETO.
            # 3 attempts: wait 1s, 2s, 4s between retries.
            last_exc = None
            response = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model    = self.model,
                        messages = [
                            {
                                "role"   : "system",
                                "content": "You are a strict hedge fund risk manager. Respond only with valid JSON."
                            },
                            {
                                "role"   : "user",
                                "content": prompt
                            }
                        ],
                        temperature = 0.1,
                        max_tokens  = 150,
                        timeout     = 10,
                    )
                    break   # success — exit retry loop
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        wait = 2 ** attempt   # 1s, 2s
                        logger.warning(
                            f'Veto agent attempt {attempt+1}/3 failed for {symbol}: {exc} '
                            f'— retrying in {wait}s'
                        )
                        time.sleep(wait)
                    else:
                        raise   # exhausted retries → caught by outer except

            response_text = response.choices[0].message.content.strip()
            if '```' in response_text:
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]

            result     = json.loads(response_text)
            decision   = result.get('decision', 'APPROVE').upper()
            reason     = result.get('reason', 'No reason')
            confidence = float(result.get('confidence', 0.5))

            # FIX: capture original value BEFORE overwriting, default to VETO not APPROVE
            if decision not in ('APPROVE', 'VETO'):
                original_decision = decision
                decision = 'VETO'
                reason   = (
                    f"Unexpected decision value '{original_decision}' from model "
                    f"— defaulting to VETO for safety"
                )

            print(f"  Veto Agent [{symbol}]: {decision}")
            print(f"  Reason: {reason}")
            print(f"  Confidence: {confidence:.0%}")

            return {
                'decision'  : decision,
                'reason'    : reason,
                'confidence': confidence,
            }

        except json.JSONDecodeError as e:
            logger.warning(f'Veto agent JSON parse error for {symbol}: {e} — VETOING')
            return {
                'decision'  : 'VETO',
                'reason'    : 'Model response was not valid JSON — cannot verify signal safety',
                'confidence': 0.0,
            }
        except Exception as e:
            # FIX: fail-closed — any error (network, rate-limit, timeout) returns VETO.
            # The original code returned APPROVE here, making a broken veto agent
            # indistinguishable from a functioning one that approved the trade.
            logger.warning(f'Veto agent error for {symbol}: {e} — VETOING (fail-closed)')
            return {
                'decision'  : 'VETO',
                'reason'    : f'Veto agent unavailable ({type(e).__name__}) — trade blocked for safety',
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
    print(f"\nDecision: {result['decision']}")
    print(f"Reason:   {result['reason']}")
