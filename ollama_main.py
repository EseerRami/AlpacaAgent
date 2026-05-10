"""
Ollama-only AlpacaAgent — replaces the PydanticAI/Anthropic agent loop.

Why this exists:
  Original main.py used PydanticAI with strict structured output, which only
  reliably works with frontier models (Anthropic Haiku, GPT-4, etc.).
  Local Ollama models (llama3.1, qwen2.5) couldn't satisfy the strict schema
  and the agent failed every cycle with "Exceeded maximum retries (3) for
  output validation".

This is the deterministic-pipeline version (same pattern as OllamaCodeTrader):
  1. Snapshot Alpaca account + positions
  2. Fetch recent news for each watchlist symbol (Google News RSS, free)
  3. Send ONE call to Ollama with all context, get JSON decision back
  4. Parse + execute via Alpaca

Total external cost: $0. Just GPU/CPU cycles.
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(ROOT))

# Reuse existing Google News fetcher we wrote earlier (free, no key)
from integrations.tavily.search import web_search

STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)
DECISIONS_LOG = STATE_DIR / "decisions.jsonl"

# Defaults
WATCHLIST = [s.strip() for s in os.environ.get("WATCHLIST", "BTC,ETH,SOL").split(",") if s.strip()]
LOOP_SECONDS = int(os.environ.get("LOOP_SECONDS", "300"))  # 5min
ERROR_SLEEP_SECONDS = 900
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.65"))
MAX_POSITION_PCT = float(os.environ.get("MAX_POSITION_PCT", "0.05"))  # 5% per trade
ALLOW_TRADING = os.environ.get("ALLOW_TRADING", "true").lower() == "true"

# Map shorthand -> Alpaca crypto pair
CRYPTO_PAIRS = {
    "BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD",
    "DOGE": "DOGE/USD", "AVAX": "AVAX/USD", "LINK": "LINK/USD",
    "DOT": "DOT/USD", "MATIC": "MATIC/USD", "UNI": "UNI/USD",
    "AAVE": "AAVE/USD", "LTC": "LTC/USD", "BCH": "BCH/USD",
    "PEPE": "PEPE/USD", "SHIB": "SHIB/USD", "HYPE": "HYPE/USD",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(action: str, **fields):
    entry = {"ts": _now_iso(), "action": action, **fields}
    with DECISIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{entry['ts']}] {action} {json.dumps(fields)[:200]}")


# ---------------- Ollama call -------------------------------------------
def ollama_chat(system: str, user: str, max_tokens: int = 1500,
                temperature: float = 0.2) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i:j+1])
    raise ValueError(f"No JSON in: {text[:200]}")


# ---------------- Alpaca ------------------------------------------------
def load_alpaca():
    key = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise SystemExit("ALPACA_KEY / ALPACA_SECRET missing in .env")
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import CryptoHistoricalDataClient
    return TradingClient(key, secret, paper=True), CryptoHistoricalDataClient(key, secret)


def get_account_snapshot(tc) -> dict:
    acct = tc.get_account()
    positions = tc.get_all_positions()
    return {
        "equity": round(float(acct.equity), 2),
        "cash": round(float(acct.cash), 2),
        "buying_power": round(float(acct.buying_power), 2),
        "positions": [{
            "symbol": p.symbol, "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": round(float(p.market_value), 2),
            "unrealized_plpc": round(float(p.unrealized_plpc) * 100, 2),
        } for p in positions],
    }


def fetch_news_for(symbols: List[str]) -> Dict[str, List[dict]]:
    """Pull 3 recent headlines per symbol via free Google News RSS."""
    out = {}
    for sym in symbols:
        try:
            res = web_search(None, query=f"{sym} crypto", topic="news",
                             days=2, max_results=3)
            out[sym] = [{
                "title": r.get("title", "")[:140],
                "source": r.get("source", ""),
                "date": (r.get("published_date") or "")[:10],
            } for r in res.get("results", [])[:3]]
        except Exception as e:
            out[sym] = [{"error": str(e)[:80]}]
    return out


def place_buy(tc, pair: str, notional: float):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    req = MarketOrderRequest(
        symbol=pair, notional=round(notional, 2),
        side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
    )
    return tc.submit_order(order_data=req)


def place_sell(tc, symbol: str):
    """Close position by symbol."""
    return tc.close_position(symbol)


# ---------------- Decision pipeline -------------------------------------
SYSTEM_PROMPT = """You are a paper-trading agent for crypto markets via Alpaca.

You see: account snapshot, current positions, recent news headlines per watchlist symbol.
Your job: decide ONE action this cycle (buy / sell / hold).

Rules:
- Per-trade max: 5% of equity (notional)
- If conviction is weak, choose HOLD (don't force trades)
- For SELL: only if you currently hold the symbol and want to exit
- Cite news themes that informed your decision

Output ONLY valid JSON, no prose, no markdown:
{
  "action": "buy" | "sell" | "hold",
  "symbol": "BTC" | "ETH" | "SOL" | etc | null,
  "notional_usd": <number, only for buy>,
  "confidence": <0.0..1.0>,
  "reason": "<one short sentence>",
  "key_themes": ["theme1", "theme2"],
  "sleep_seconds": <60..3600, how long until next cycle>
}"""


def cycle(tc, _data_client):
    print(f"\n=== {_now_iso()} cycle ===")
    snap = get_account_snapshot(tc)
    print(f"[account] equity=${snap['equity']:,.2f} cash=${snap['cash']:,.2f} positions={len(snap['positions'])}")

    news = fetch_news_for(WATCHLIST)
    n_articles = sum(len(v) for v in news.values())
    print(f"[news] fetched {n_articles} headlines across {len(WATCHLIST)} symbols")

    user_msg = (
        f"Watchlist: {', '.join(WATCHLIST)}\n\n"
        f"Account snapshot:\n```json\n{json.dumps(snap, indent=2)}\n```\n\n"
        f"Recent news per symbol:\n```json\n{json.dumps(news, indent=2)}\n```\n\n"
        "Decide your next action and output JSON now."
    )
    raw = ollama_chat(SYSTEM_PROMPT, user_msg, max_tokens=1000)

    try:
        decision = parse_json(raw)
    except Exception as e:
        print(f"[parse] failed: {e}")
        _log("PARSE_FAIL", error=str(e), raw=raw[:300])
        return 600

    action = (decision.get("action") or "hold").lower()
    symbol = (decision.get("symbol") or "").upper()
    confidence = float(decision.get("confidence") or 0)
    reason = decision.get("reason", "")
    sleep_s = max(60, min(3600, int(decision.get("sleep_seconds") or 300)))

    print(f"[decision] {action.upper()} {symbol or '-'} conf={confidence:.2f} sleep={sleep_s}s")
    print(f"  reason: {reason}")

    if action == "hold" or confidence < MIN_CONFIDENCE:
        _log("HOLD", confidence=confidence, threshold=MIN_CONFIDENCE,
             reason=reason if action == "hold" else "low_confidence")
        return sleep_s

    if not ALLOW_TRADING:
        _log("DRY", action=action, symbol=symbol, confidence=confidence, reason=reason)
        return sleep_s

    if action == "buy":
        pair = CRYPTO_PAIRS.get(symbol)
        if not pair:
            print(f"[buy] {symbol} not in CRYPTO_PAIRS, skipping")
            _log("SKIP", reason=f"unknown symbol {symbol}")
            return sleep_s
        notional = float(decision.get("notional_usd") or 0)
        max_notional = snap["equity"] * MAX_POSITION_PCT
        if notional <= 0 or notional > max_notional:
            notional = max_notional  # clamp
            print(f"[buy] notional clamped to ${notional:.2f} ({MAX_POSITION_PCT*100:.0f}% cap)")
        if notional > snap["cash"]:
            notional = snap["cash"] * 0.95
        if notional < 5:
            _log("SKIP", reason=f"notional too small (${notional:.2f})")
            return sleep_s
        try:
            order = place_buy(tc, pair, notional)
            _log("BUY", symbol=symbol, pair=pair, notional=round(notional, 2),
                 order_id=str(order.id), confidence=confidence, reason=reason,
                 themes=decision.get("key_themes", []))
        except Exception as e:
            _log("BUY_FAIL", symbol=symbol, error=str(e))
        return sleep_s

    if action == "sell":
        # Find existing position
        pair = CRYPTO_PAIRS.get(symbol)
        held = next((p for p in snap["positions"]
                     if p["symbol"] == symbol or p["symbol"] == (pair or "").replace("/", "")),
                    None)
        if not held:
            _log("SKIP", reason=f"no open position in {symbol}")
            return sleep_s
        try:
            place_sell(tc, held["symbol"])
            _log("SELL", symbol=symbol, qty=held["qty"],
                 entry=held["avg_entry_price"], exit=held["current_price"],
                 pnl_pct=held["unrealized_plpc"], confidence=confidence, reason=reason)
        except Exception as e:
            _log("SELL_FAIL", symbol=symbol, error=str(e))
        return sleep_s

    return sleep_s


def main():
    print(f"[agent] starting · model={OLLAMA_MODEL} · watchlist={WATCHLIST} · "
          f"allow_trading={ALLOW_TRADING} · min_conf={MIN_CONFIDENCE}")
    tc, dc = load_alpaca()
    while True:
        try:
            sleep_s = cycle(tc, dc)
        except KeyboardInterrupt:
            print("[agent] stopped.")
            break
        except Exception as e:
            print(f"[agent] cycle error: {e!r}")
            traceback.print_exc()
            sleep_s = ERROR_SLEEP_SECONDS
        time.sleep(max(int(sleep_s), 1))


if __name__ == "__main__":
    main()
