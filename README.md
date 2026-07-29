# Abhi MSS-5s NQ Bot

Market-Structure-Shift entry strategy on the 5-second timeframe for the NQ
E-mini, running on Abhi's TopstepX practice account.

## How it works
- Builds 5-second candles live from the 1-second tick feed.
- Tracks an 8-EMA on those 5s candles and detects swing highs/lows with a
  ZigZag filter (default 15 points).
- Enters on a structure shift only:
  - SHORT when the market makes successive lower highs (trend breaking down)
    and price is back below the 8-EMA.
  - LONG when it makes successive higher lows and price is back above the EMA.
- No 1-hour trend gate — it acts purely on the shift, as requested.
- Stop is placed at the invalidation swing, capped so a losing trade never
  risks more than ~$1,000.
- Target = 1.75x the risk on the trade; profit-lock trails runners.

## Files
- `mss5s.py` — the MSS shift tracker (EMA + ZigZag swing logic).
- `nq_signal_renko.py` — signal engine (routes to MSS mode for Abhi).
- `nq_signal_bot.py` — the live bot (feed handling, order placement, risk).

Config (account keys, sizing, mss settings) lives in a separate
`accounts.json` that is NOT part of this repo — no credentials are stored here.
