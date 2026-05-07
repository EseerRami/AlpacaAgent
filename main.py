import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient

from core.agent import configure_model
from core.runner import run_once
from integrations.alpaca.config import load_config
from schemas.deps import Deps

# Load .env from this directory before reading env vars
load_dotenv(Path(__file__).parent / ".env", override=True)

# Model: Anthropic Claude Haiku (fast + cheap). Override with MODEL env var.
MODEL = os.environ.get("MODEL", "anthropic:claude-haiku-4-5")

# Absolute path so the dashboard can find the SQLite file regardless of CWD.
DB_PATH = str(Path(__file__).parent / "state" / "state.db")

# Identifier for this agent instance (lets you run multiple in parallel later).
STATE_KEY = os.environ.get("STATE_KEY", "btc-eth-agent")

# Prompt: scan multiple symbols. Update this if you want a different beat.
BASE_PROMPT = os.environ.get(
    "BASE_PROMPT",
    "Scan recent news and price context for BTC, ETH, and SOL. "
    "Recommend buy/sell/hold for the strongest opportunity (or wait if nothing compelling). "
    "Cite real news sources you used.",
)

# Trading mode: True = the agent can actually place orders on its Alpaca account.
# Per user request (option B): second paper account, real paper trades.
ALLOW_TRADING = os.environ.get("ALLOW_TRADING", "true").lower() == "true"


def main():
    if not os.environ.get("ALPACA_KEY") or not os.environ.get("ALPACA_SECRET"):
        print("ERROR: ALPACA_KEY / ALPACA_SECRET not set in .env", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("TAVILY_API_KEY"):
        print("ERROR: TAVILY_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    print(f"[agent] starting · model={MODEL} · state_key={STATE_KEY} · allow_trading={ALLOW_TRADING}")
    print(f"[agent] db={DB_PATH}")

    configure_model(MODEL, strict=True)

    deps = Deps(
        alpaca=load_config(),
        tavily=TavilyClient(api_key=os.environ["TAVILY_API_KEY"]),
        allow_trading=ALLOW_TRADING,
    )

    try:
        while True:
            try:
                sleep_s = run_once(
                    deps,
                    base_prompt=BASE_PROMPT,
                    conf_threshold=0.65,  # was 0.75 — too strict; LLM rarely hits it
                    poll_sleep_seconds=30,
                    db_path=DB_PATH,
                    state_key=STATE_KEY,
                )
                print(f"[agent] cycle complete · sleeping {sleep_s}s")
            except Exception as e:
                print(f"[agent] loop error: {e!r}")
                sleep_s = 900  # 15 min on error (was 60s; Tavily quota exhaust spammed logs)

            time.sleep(max(int(sleep_s), 1))
    except KeyboardInterrupt:
        print("[agent] stopped.")


if __name__ == "__main__":
    main()
