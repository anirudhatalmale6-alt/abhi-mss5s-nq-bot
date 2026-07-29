#!/usr/bin/env python3.12
"""
Renko Volume-Flip Bot  (TopstepX / NQ)

Strategy = the client's Pine "Volume Buy/Sell Signals", ported to a live Renko grid:

    volume_color = close[1] > close[2] ? green : red      # == the brick direction
    green_signal = colour turns green                     # == the brick flip
    red_signal   = colour turns red

  * Bricks are built from 1-SECOND BAR CLOSES (TradingView-style), NOT from raw
    ticks. Measured: this removes ~64% of the flips a tick-fed Renko produces.
    That is the whipsaw the client asked to avoid.
  * Always in the market. Long while bricks are green, short while red.
  * Stop = "the flip" (2 bricks against). Because the flip level is recomputed off
    the newest brick, the stop TRAILS automatically as the trend prints bricks.
    The client's stop and his trailing-profit request are the same mechanism.
  * Entries are MARKETABLE LIMIT with a hard slippage cap, never market orders.
    The Renko grid gives us the exact entry price in advance, so there is no
    excuse to pay the 0.79pt average slippage that has been bleeding the BB bot.
    If the book will not fill us inside the cap, we SKIP the trade.
  * Exits fall back to MARKET if the limit does not fill. Picky getting in,
    certain getting out - a position must never run away.

Config: accounts_renko.json
"""

import argparse, asyncio, json, os, signal, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from renko_volume_engine import RenkoVolumeEngine, viability
from nq_signal_core import NQSignal, in_session as nq_in_session
from nq_signal_renko import NQSignalRenko, RenkoFlip
from nq_levels_rl import LevelEntryRL, state_key

ET = ZoneInfo("America/New_York")
POINT_VALUE = 20.0          # NQ = $20 / point
TICK = 0.25

STATE_FILE   = "nq_signal_state.json"
TRADE_LOG    = "nq_signal_trades.jsonl"

# ---- watchdog -------------------------------------------------------------
# A bot that is WRONG is dangerous. A bot that is SILENTLY ASLEEP is merely
# useless - but it looks identical to a healthy one from the outside, which is
# how it went unnoticed for five hours on 13-Jul: the websocket died at 14:30 ET,
# the price read failed on every pass, the loop swallowed it, and systemd kept
# reporting "active" because the process never exited. Restart=always cannot save
# a process that refuses to die.
#
# So: if the feed goes quiet while the market is open, we do not try to heal in
# place (a wedged HTTP session and a dead SignalR socket are not reliably
# repairable from inside). We exit non-zero and let systemd rebuild us clean.
#
# Exiting is SAFE even holding a position: the protective stop is a real resting
# order on the exchange, not something we hold in memory, and reconcile() adopts
# any open position on the next boot.
STALE_TICK_S  = 120.0   # no usable price for this long, market open -> restart
MAX_API_FAILS = 15      # consecutive platform-read failures -> restart

# In-process connection recovery (added 16-Jul at the client's request, adapted
# from his bot_upgraded_v2 "connection that recovers instead of crashing").
#
# The original design deliberately chose exit-and-let-systemd-rebuild over
# healing in place, because a dead SignalR socket cannot be revived. But a fresh
# TradingSuite.create() does NOT revive the dead socket - it builds a brand-new
# one, with a new HTTP session: a real rebuild, the same thing systemd does, only
# without killing the process. That is safe to try FIRST, and it heals a
# transient drop with zero downtime and zero lost in-memory state. The exit path
# below is KEPT as the last resort: if in-process reconnects keep failing, we
# still bail out so systemd can rebuild us clean.
FROZEN_FEED_S           = 60.0   # feed quiet this long, market open -> reconnect
USER_HUB_DOWN_S         = 45.0   # order/account (user) hub down this long, market open -> reconnect
RECONNECT_COOLDOWN_BASE = 5.0    # backoff after the first failed reconnect
RECONNECT_COOLDOWN_MAX  = 120.0  # cap on the exponential backoff
MAX_RECONNECT_TRIES     = 6      # give up in-process after this -> hand to systemd

# The realtime price read (get_current_price) very occasionally hands back a
# stale/phantom quote that is 100+ points off the live market (seen 22-Jul on
# Abhi: it returned 29165 while the market - and every real fill - was at
# ~29060). Traded on, that phantom priced entries 100pt off and set the stop on
# the wrong side of the actual fill, so the platform flattened each trade
# instantly for a 1-3pt loss and re-armed: 68 tiny losses in a session. NQ
# simply cannot travel this far between two ~1s reads, so a jump larger than
# this is a bad tick, not a move. Reject isolated outliers; adopt a far level
# only after several reads agree (a genuine gap), so a real fast move still gets
# through. The Renko engine is fed from CLOSED bars, not this tick, so dropping
# a bad live tick never starves the bricks/EMA.
MAX_TICK_JUMP = 40.0             # points; > this vs last good price = phantom

# Broker refusals that no amount of retrying will fix. Restarting does not help
# either - the account itself is the problem. Stop, and say so.
FATAL_REJECTS = ("account is closed", "account is locked", "account not found",
                 "account is disabled", "not authorized", "trading is disabled",
                 "account is not active")


def market_open(now=None):
    """CME Globex NQ: Sun 18:00 ET -> Fri 17:00 ET, halted daily 17:00-18:00 ET.

    The watchdog must know this or it would restart-loop all weekend, when a
    silent feed is the correct and expected state.
    """
    n = now or datetime.now(ET)
    wd = n.weekday()                       # Mon=0 .. Sat=5, Sun=6
    mins = n.hour * 60 + n.minute
    if wd == 5:                            # Saturday: shut
        return False
    if wd == 6:                            # Sunday: reopens 18:00 ET
        return mins >= 18 * 60
    if wd == 4 and mins >= 17 * 60:        # Friday 17:00 ET -> weekend
        return False
    return not (17 * 60 <= mins < 18 * 60)  # daily maintenance halt


def patch_position_model():
    """THE most important six lines in this file.

    TopstepX now returns a `contractDisplayName` field on /Position/searchOpen.
    project_x_py's `Position` is a plain @dataclass with a fixed field list, so
    `Position(**data)` raises TypeError the moment the API sends anything new.

    That alone would be survivable. What makes it lethal is that
    `position_manager.get_all_positions()` CATCHES that TypeError, logs it, and
    returns an EMPTY LIST. So the library cheerfully reports "no open positions"
    at the exact moment a position IS open - and reports it correctly (flat) when
    we are genuinely flat. An always-wrong-in-one-direction position feed.

    Every downstream safety check we own - "is the platform flat before I enter?",
    "did my stop fire?", "am I still in this trade?" - was reading that lie. It
    booked fictional stop-outs, orphaned live stop orders on a flat account, and
    re-entered on top of positions it already held.

    Patched here rather than in site-packages so that a `pip install -U` cannot
    silently undo it. Filtering unknown keys (rather than adding the one field)
    means the next field TopstepX invents cannot break us either.
    """
    import dataclasses
    from project_x_py.models import Position
    if getattr(Position, "_tolerant_init", False):
        return
    allowed = {f.name for f in dataclasses.fields(Position)}
    original = Position.__init__

    def tolerant(self, *args, **kwargs):
        original(self, *args, **{k: v for k, v in kwargs.items() if k in allowed})

    Position.__init__ = tolerant
    Position._tolerant_init = True
    print("[PATCH] Position model made tolerant of unknown API fields "
          "(TopstepX sends contractDisplayName; the library would otherwise "
          "report FLAT while holding a position)")


def now_str():
    return datetime.now(ET).strftime("%H:%M:%S")


class FlowTracker:
    """Reconstructs ORDER FLOW from the quote stream.

    TopstepX does NOT give us Level 2. Verified on a live open market: 60 seconds
    produced 1,124 quote updates and exactly ZERO market-depth events. The quote
    payload is {bid, ask, last, volume} - top-of-book PRICES with no sizes, no
    ladder, no per-trade aggressor flag. Every DOM method in project_x_py
    (get_market_imbalance, detect_spoofing, ...) is fed by a market_depth event
    this broker never sends, so they all return zeros. There is no book to read.

    What we CAN do is rebuild the tape. `volume` is a running session counter, so
    the jump between two quotes is the size that just traded, and comparing `last`
    against the prevailing bid/ask tells us who crossed the spread (the Lee-Ready
    tick rule):

        traded at/above the ask  -> a buyer lifted the offer   -> +size
        traded at/below the bid  -> a seller hit the bid       -> -size
        in between               -> fall back to the tick test

    Summed over a window that is CUMULATIVE DELTA: real aggressor order flow.
    It is also the half of order flow that actually carries information - resting
    book depth is the half that gets spoofed.

    NOTE: this only RECORDS. Nothing here gates a trade. See the note in enter().
    """

    def __init__(self, windows=(30, 300)):
        self.windows = windows
        self.events = []                     # [(ts, signed_size)]
        self._bid = self._ask = self._last = None
        self._vol = None

    def on_quote(self, q, ts):
        bid, ask = q.get("bid"), q.get("ask")
        last, vol = q.get("last"), q.get("volume")
        if vol is not None and self._vol is not None and last is not None:
            dv = vol - self._vol
            if dv > 0:
                sign = 0
                if self._ask is not None and last >= self._ask:
                    sign = 1
                elif self._bid is not None and last <= self._bid:
                    sign = -1
                elif self._last is not None:
                    sign = 1 if last > self._last else (-1 if last < self._last else 0)
                if sign:
                    self.events.append((ts, sign * dv))
        if bid is not None:
            self._bid = bid
        if ask is not None:
            self._ask = ask
        if last is not None:
            self._last = last
        if vol is not None:
            self._vol = vol
        cutoff = ts - max(self.windows) - 5
        if len(self.events) > 4000:
            self.events = [e for e in self.events if e[0] >= cutoff]

    def snapshot(self, ts):
        """Delta + buy/sell volume over each window. Recorded with every trade."""
        out = {}
        for w in self.windows:
            cut = ts - w
            buys = sells = 0
            for t, s in reversed(self.events):
                if t < cut:
                    break
                if s > 0:
                    buys += s
                else:
                    sells += -s
            tot = buys + sells
            out[f"delta_{w}s"] = buys - sells
            out[f"buyvol_{w}s"] = buys
            out[f"sellvol_{w}s"] = sells
            # -1.0 = all selling, +1.0 = all buying, 0 = balanced
            out[f"imbalance_{w}s"] = round((buys - sells) / tot, 4) if tot else 0.0
        out["spread"] = (round(self._ask - self._bid, 2)
                         if self._bid is not None and self._ask is not None else None)
        return out


class RenkoVolAccount:
    def __init__(self, cfg):
        self.name         = cfg["name"]
        self.username     = cfg["username"]
        self.api_key      = cfg["api_key"]
        self.account_name = cfg["account_name"]
        self.symbol       = cfg.get("symbol", "NQ")
        # account_auto (client 22-Jul): adopt a reset/replaced practice account
        # AUTOMATICALLY so no manual config edit is needed each time TopStep resets
        # it. On connect we pick the newest TRADEABLE account whose name contains
        # account_match (default = the configured account's family prefix, i.e.
        # everything before the final "-<id>"), and NEVER one listed in
        # account_exclude (the funded / combine / eval account). Because the service
        # is Restart=always, this also recovers a reset LIVE: the old account
        # vanishes -> connect raises ACCOUNT_NOT_FOUND -> systemd restarts us -> we
        # auto-pick the new practice account and carry on.
        self.account_auto    = bool(cfg.get("account_auto", False))
        self.account_match   = cfg.get("account_match") or (
            self.account_name.rsplit("-", 1)[0]
            if (self.account_auto and "-" in self.account_name) else "")
        self.account_exclude = list(cfg.get("account_exclude", ["50KTC", "COMBINE", "EVAL"]))
        # limit_entry (client 22-Jul): instead of a marketable limit that SKIPS a bad
        # fill, rest a passive LIMIT at the brick level offset by limit_entry_offset_ticks
        # (default +1 tick = +0.25 "above the level") and let a small pullback fill us
        # at that exact price. A fresh signal cancels+replaces the resting order; if
        # price never pulls back it simply never fills. off is added to the anchor
        # (last confirmed brick close) for BOTH sides: LONG fills on a dip to it,
        # SHORT fills on a bounce up to it - i.e. always 0.25 above the level.
        self.limit_entry              = bool(cfg.get("limit_entry", False))
        self.limit_entry_offset_ticks = int(cfg.get("limit_entry_offset_ticks", 1))
        self.limit_entry_ttl_s        = float(cfg.get("limit_entry_ttl_s", 45.0))
        self._pending_entry           = None   # {order_id,dir,level,anchor,limit_price,sl_level,ts}
        # confirm_entry (client 22-Jul): "no falling-knife" entry, supersedes limit_entry.
        # On a signal, DON'T enter at the level. First wait for price to dip
        # confirm_pullback BELOW the entry brick (long) / ABOVE it (short) - the
        # small pullback. THEN watch the live 1-second price and only enter (at MARKET)
        # once it RESUMES confirm_resume back in the signal direction (a 0.50 brick our
        # way), confirming the move restarted. If price runs through the first-brick SL
        # before that, the setup is void. Same re-arm-on-new-signal / TTL as limit_entry.
        self.confirm_entry    = bool(cfg.get("confirm_entry", False))
        self.confirm_pullback = float(cfg.get("confirm_pullback", 0.25))
        self.confirm_resume   = float(cfg.get("confirm_resume", 0.50))
        self._confirm_state   = None   # {dir,level,anchor,sl_level,phase,extreme,ts}

        self.brick_size   = float(cfg.get("brick_size", 0.25))
        self.qty          = int(cfg.get("contracts", 1))
        # hard ceiling on what we will pay to get in, in TICKS
        self.max_entry_slip_ticks = int(cfg.get("max_entry_slip_ticks", 1))
        self.entry_timeout_s      = float(cfg.get("entry_timeout_s", 2.0))
        self.daily_loss_limit     = float(cfg.get("daily_loss_limit", 1000.0))
        self.max_trades_per_day   = int(cfg.get("max_trades_per_day", 200))
        self.enabled              = bool(cfg.get("enabled", True))

        self.engine = RenkoVolumeEngine(brick_size=self.brick_size, tick_size=TICK)

        self.suite = None
        self.ctx = None
        self.connected = False

        self.position = 0          # -1 short, 0 flat, +1 long
        self.entry_price = 0.0
        self.intended_entry = 0.0
        self.entry_time = 0.0
        self.contracts_held = 0

        # Per-trade excursion tracking - RECORDED ONLY, it gates nothing (same
        # spirit as the flow recorder). MFE = furthest the trade ran in our
        # favour; MAE = furthest it went against us, both in points, measured
        # tick-by-tick between entry and exit. This is the raw material for the
        # later RL risk-management work: "how far did it usually go against us
        # before it worked, and how much profit was on the table at the peak."
        self._mfe_pts = 0.0
        self._mae_pts = 0.0
        self._mfe_price = 0.0
        self._mae_price = 0.0

        self.live_pnl = 0.0
        self.daily_loss = 0.0
        self.trades_today = 0
        self.session_day = datetime.now(ET).date().isoformat()

        # The daily-loss breaker is the only thing standing between a bad day and
        # a locked account - so it must not be measured with the bot's own
        # bookkeeping. Its trade-by-trade tally drifts: it misses commissions, and
        # it loses trades across restarts (on 13-Jul it read +$315 while the broker
        # said -$72). The broker's BALANCE is the truth. Anchor to the balance at
        # the session open and measure against that.
        self.day_open_balance = None
        self._last_pnl_sync = 0.0

        self.skipped_slippage = 0   # trades we refused because the fill was too bad
        self.last_reject = None     # the broker's own words, when it refuses an order
        self._halted = False
        self._stop_order_id = None  # REAL protective stop resting on the platform
        self._stop_price = None
        self._stop_fill_price = None  # set when the stop EXECUTED instead of cancelling
        self._api_fails = 0         # consecutive platform-read failures

        # In-process reconnect bookkeeping (see FROZEN_FEED_S block up top).
        self.health = "HEALTHY"                       # HEALTHY|RECONNECTING|DEGRADED|UNSAFE
        self._last_reconnect = 0.0
        self._reconnect_tries = 0
        self._reconnect_cooldown = RECONNECT_COOLDOWN_BASE
        self._user_down_since = None  # when the order/account (user) hub first went missing

        # Watchdog inputs. last_tick_ts is the last time we got a USABLE price -
        # not the last time we asked for one. A dead feed still answers the call.
        self.last_tick_ts = time.time()
        self.bad_ticks = 0
        self._last_tick_gripe = 0.0
        self._last_good_px = None      # last accepted live price (phantom-tick guard)
        self._px_reject_streak = 0     # consecutive far reads (a real gap once >=3)

        # The position feed lags a fill. Never declare a position closed until it
        # has had time to appear, and never on a single reading.
        self.reconcile_grace_s = float(cfg.get("reconcile_grace_s", 10.0))
        self._flat_polls = 0
        self._recent_entries = []   # churn circuit-breaker

        # Order flow, RECORDED ONLY - it gates nothing. See FlowTracker.
        self.flow = FlowTracker()
        self.flow_at_entry = {}

        # ---- EMA-cross mode (the client's design) ---------------------------
        # ema_len = 0 -> plain flip: always in the market, reverse on every brick
        #               colour change.
        # ema_len > 0 -> run an EMA over the BRICK closes. A brick closing across
        #               the EMA "arms" a direction; we do not take that brick. We
        #               wait for the next brick to CONFIRM the direction and enter
        #               there ("break of the first crossing brick, so the second
        #               one in that direction"). Stop still trails at the flip
        #               level, and an opposite cross closes us.
        # Backtested on 20,000 real 1s bars: the cross RAISES profit factor at
        # every brick size (e.g. brick 6: PF 1.32 -> 1.98). It does not rescue
        # brick 1, which loses either way.
        self.ema_len = int(cfg.get("ema_len", 0))
        self._ema = None
        self._prev_above = None
        self._armed = None          # 'long' | 'short' | None
        # ---- 70% ghost-brick early entry (client 17-Jul) --------------------
        # Once a brick has closed across the EMA and armed a direction, DON'T
        # wait for the next brick to fully close to confirm. Enter as soon as
        # price pushes `entry_confirm_frac` of one brick in the armed direction
        # (0.70 = 70%). 0 or >=1 disables it and falls back to full-brick close.
        self.entry_confirm_frac = float(cfg.get("entry_confirm_frac", 0.70))
        # ---- reverse-signals mode (client 19-Jul) ---------------------------
        # When true, every entry the strategy generates is placed in the OPPOSITE
        # direction (a full mirror): the LONG signal sells, the SHORT signal buys.
        # Entries flip at enter(); the EMA-cross EXIT is evaluated on the original
        # (pre-flip) "logical" signal so the exit timing mirrors too. The
        # protective stop keys off the ACTUAL position, so it still lands on the
        # correct side automatically. Everything else is untouched.
        self.reverse = bool(cfg.get("reverse_signals", False))

        # ---- NQ Signal strategy (client's Pine port) ------------------------
        # This bot does NOT use Renko. It runs the client's TradingView "NQ
        # Signal" strategy: 5m breakout + trend oscillator gated by a 30m
        # Ichimoku/Bollinger filter, inside the 09:31-15:30 NY session. Higher-
        # timeframe bars are re-fetched from the platform every few seconds and
        # fed to the pure-logic core (nq_signal_core). Entries only; NO stop and
        # NO auto take-profit (client takes profit manually for now) - the bot
        # just records how far each trade ran (MFE/MAE) to build the dataset for
        # the later RL layer. One position at a time.
        # ---- Renko variant (Abhi A/B, client 20-Jul) ------------------------
        # renko_mode: run the SAME strategy on ONE Renko brick stream instead of
        # 5m/30m time candles. Individual bricks = fast/entry (breakout fires on
        # the UNCONFIRMED forming brick); every `renko_group` bricks aggregate into
        # the higher timeframe that carries the trend + Ichimoku/BB filter. Bricks
        # are built from 1-second closes (the engine buckets live ticks by second).
        # renko_flip: the client's 20-Jul redesign - a clean 5-MINUTE 2-point Renko
        # flip with a trailing stop. Direction = 2 same-colour bricks (2nd caught
        # early as a ghost). No oscillator, no Ichimoku - the Renko IS the signal.
        # A trailing stop locks profit (and caps the initial loss); a 2-opposite-
        # brick print flips the position. Both entry and flip are on the 5-min.
        self.renko_flip = bool(cfg.get("renko_flip", False))
        self.trail_pts  = float(cfg.get("trail_pts", 4.0))
        # trail_bricks (client 21-Jul, Rutvi): 0 = OFF (fixed first-brick stop only,
        # manual TP - Abhi). >0 = ratchet the RESTING stop up to lock profit once the
        # trade is up trail_bricks bricks, trailing trail_bricks brick(s) behind the
        # peak (in whole-brick steps). The profit is then "guaranteed by code" because
        # the locked level lives as a real resting stop, not a soft bot exit.
        self.trail_bricks = int(cfg.get("trail_bricks", 0))
        # trail_risk (client 22-Jul, 1:1 R:R): >0 = risk-based profit trail. R = the
        # initial stop distance (entry to first-brick stop). Once the trade is up
        # trail_risk x R, trail the stop trail_risk*R behind the peak. trail_risk=1 ->
        # "the TP trails the SAME distance as the SL": stop reaches breakeven at +R,
        # locks 1x the risk (true 1:1) at +2R, and keeps banking as it runs. Takes
        # precedence over trail_bricks when set.
        self.trail_risk = float(cfg.get("trail_risk", 0.0))
        self._risk_pts = 0.0        # R for the open trade, captured at entry
        # max_stop_pts (client 22-Jul): hard cap on the first-brick stop distance. A
        # late entry far from the brick structure otherwise makes an outsized stop
        # (one short lost $325 / 16pt because the brick was 15pt away). 0 = no cap.
        self.max_stop_pts = float(cfg.get("max_stop_pts", 0.0))
        # take_profit_usd (client 23-Jul): hard per-trade profit lock. When the OPEN
        # trade's unrealized profit reaches this many DOLLARS, cancel the resting stop
        # and close at market so the gain is banked instead of handed back (Abhi once
        # ran +$2k unrealized and gave it all back). 0 = off. Runs alongside the
        # trailing stop - whichever triggers first exits. Keyed on entry_time so it
        # self-resets for the next trade with no extra bookkeeping.
        self.take_profit_usd = float(cfg.get("take_profit_usd", 0.0))
        self._tp_fired_entry = 0.0     # entry_time of the trade we've already TP-closed
        self._last_tp_attempt = 0.0
        # profit_lock (client 29-Jul, Rutvi): DOLLAR trailing profit-lock, independent
        # of the 1:1 target. A reversal off a far swing carries a wide target (a 100pt
        # swing => a ~$2k target AND a ~$2k stop), so a trade can run most of the way to
        # +$2k then reverse and hand the whole gain back before hitting target or stop.
        # Once the OPEN trade's unrealized profit reaches profit_lock_arm_usd, arm a
        # floor at (peak - profit_lock_trail_usd) and ratchet it UP with the peak (never
        # down); if profit falls back to the floor, cancel the stop and close at market.
        # Guaranteed minimum banked once armed = arm - trail. 0 = off. Runs ahead of the
        # hard take_profit; whichever fires first exits.
        self.profit_lock_arm_usd   = float(cfg.get("profit_lock_arm_usd", 0.0))
        self.profit_lock_trail_usd = float(cfg.get("profit_lock_trail_usd", 0.0))
        self._plock_entry        = 0.0   # entry_time of the trade we're tracking
        self._plock_peak_usd     = 0.0   # best unrealized $ seen this trade
        self._plock_armed        = False
        self._plock_fired_entry  = 0.0
        self._last_plock_attempt = 0.0
        if self.profit_lock_arm_usd > 0:
            print(f"[{self.name}] profit-lock ON: arm +${self.profit_lock_arm_usd:.0f}, "
                  f"trail ${self.profit_lock_trail_usd:.0f} (guaranteed min banked "
                  f"+${self.profit_lock_arm_usd - self.profit_lock_trail_usd:.0f} once armed)", flush=True)
        # entry_tf (client 22-Jul): timeframe the ENTRY Renko is built from,
        # [interval, unit] (unit 1=seconds, 2=minutes). Default 5-min [5,2]; Rutvi now
        # 15-sec [15,1] to catch entries earlier, with the 5-min Renko as the trend
        # filter (trend_tf_mins=[5]). Fewer history days for sub-minute bars.
        _etf = cfg.get("entry_tf", [5, 2])
        self._etf_i, self._etf_u = int(_etf[0]), int(_etf[1])
        self._etf_feed_days = 1 if self._etf_u == 1 else 5
        self._etf_warm_days = 1 if self._etf_u == 1 else 15
        self._etf_label = f"{self._etf_i}s" if self._etf_u == 1 else f"{self._etf_i}m"
        self._peak = 0.0                    # best price since entry (trailing stop)
        self._reentry_block_bricks = 0      # after a trail-stop, wait for a NEW brick
        self.renko_mode = bool(cfg.get("renko_mode", False))
        # use_time_htf: read the trend + filter off REAL 5m/30m bars (matches the
        # client's actual 5min/30min charts). flip_mode: when in a position and the
        # 1-brick unconfirmed colour flips against us, REVERSE to that side (always
        # -in-market flip that cuts losses and rides the trend).
        self.use_time_htf = bool(cfg.get("use_time_htf", False))
        self.flip_mode    = bool(cfg.get("flip_mode", False))
        self.ghost_exit   = bool(cfg.get("ghost_exit", False))
        # churn breaker ceiling (entries in 60s). A 1-brick flip strategy trades
        # fast, so it needs a higher ceiling than the renko bots' default of 6.
        self.churn_max    = int(cfg.get("churn_max", 6))
        if self.renko_flip:
            self.sig = RenkoFlip(brick_size=float(cfg.get("brick_size", 2.0)),
                                 confirm=int(cfg.get("confirm_bricks", 2)),
                                 entry_frac=self.entry_confirm_frac,
                                 ema_len=self.ema_len)
        # ---- swing-level filter + RL entry decision (client 27-Jul, Abhi) -----
        # levels_filter: the strict support/resistance gate (only fade AT a level).
        # levels_rl:     let RL learn which level-setups to take vs skip on top of it.
        self.levels_filter = bool(cfg.get("levels_filter", False))
        self.levels_rl     = bool(cfg.get("levels_rl", False))
        # flip_signals (client 27-Jul): trade the OPPOSITE of what the strategy
        # signals - if fading into levels keeps hitting the stop, take the break /
        # continuation instead. Applied early in the entry path so the protective
        # stop, the RL state and the order all derive from the flipped side.
        self.flip_signals  = bool(cfg.get("flip_signals", False))
        self.level_tol_pts   = float(cfg.get("level_tol_pts", 6.0))
        self.level_break_pts = float(cfg.get("level_break_pts", 2.0))
        self.swing_min_run   = int(cfg.get("swing_min_run", 2))
        if self.renko_flip and self.levels_filter and hasattr(self.sig, "configure_levels"):
            self.sig.configure_levels(True, tol=self.level_tol_pts,
                                      brk=self.level_break_pts,
                                      min_run=self.swing_min_run,
                                      max_n=int(cfg.get("level_max", 40)))
        # wave_mode (client 27-Jul, Rutvi): Elliott wave-3 trend-riding entry. Uses
        # the engine's swing pivots to spot a wave-3 impulse (higher-high + higher-low
        # for longs, mirror for shorts) and enters with the BB in the TREND direction
        # - riding the move instead of fading it. It replaces the strict fade gate;
        # flipping the signal would invert the trend read, so flip is forced off here.
        self.wave_mode = bool(cfg.get("wave_mode", False))
        if self.renko_flip and self.wave_mode and hasattr(self.sig, "configure_wave"):
            self.sig.configure_wave(True, min_run=self.swing_min_run,
                                    max_pivots=int(cfg.get("wave_max_pivots", 60)))
            if self.flip_signals:
                self.flip_signals = False
                print(f"[{self.name}] wave_mode ON -> flip_signals disabled (structure sets direction)")
        # reversal_mode (client 28-Jul, Rutvi): 8-EMA trend REVERSAL. Two lower
        # swing highs in a row -> SHORT / two higher swing lows -> LONG, entered at
        # market on confirmation with a 1:1 risk/reward - the stop sits just past
        # the last swing point and the take-profit is sized to match it. Replaces
        # the wave-3 read; flip would invert the structure, so it is forced off.
        self.reversal_mode = bool(cfg.get("reversal_mode", False))
        # 1-hour trend gate (client 28-Jul "the signal should match the 1h trend"):
        # the 1h trend is coloured by an EMA-fast-vs-slow read on 1h closes; a
        # reversal only fires when its direction matches. Refreshed on a slow throttle.
        self.h1_fast = int(cfg.get("h1_fast", 8))
        self.h1_slow = int(cfg.get("h1_slow", 21))
        self.h1_days = int(cfg.get("h1_days", 5))
        self.h1_refresh_s = float(cfg.get("h1_refresh_s", 60.0))
        self._h1_trend = None
        self._h1_last_fetch = 0.0
        if self.renko_flip and self.reversal_mode and hasattr(self.sig, "configure_reversal"):
            self.sig.configure_reversal(True, stop_buf=float(cfg.get("rev_stop_buf_pts", 2.0)))
            if self.wave_mode:
                self.wave_mode = False
                if hasattr(self.sig, "configure_wave"):
                    self.sig.configure_wave(False)
            self.flip_signals = False
            print(f"[{self.name}] reversal_mode ON -> wave/flip disabled "
                  f"(2 EMA-breaks off lower highs=short / higher lows=long, "
                  f"1:1 RR, must match 1h trend {self.h1_fast}/{self.h1_slow})")
        # ---- MSS-on-5s mode (client 29-Jul, Abhi ONLY) ----------------------
        # The shift strategy on his chart: 8-EMA on 5-SECOND candles, wait for the
        # structure shift (higher highs -> lower high(s) -> short as price rolls back
        # under the EMA; mirror for longs), NO 1h gate, structural stop capped to a
        # $ max-loss, wider R-multiple target. Own tick path (_tick_mss) that reuses
        # the enter/protective-stop/take-profit plumbing. Config-gated so Rutvi
        # (mss_mode absent) is completely unaffected.
        self.mss_mode    = bool(cfg.get("mss_mode", False))
        self.mss_tp_r    = float(cfg.get("mss_tp_r", 1.5))
        self.mss_cap_usd = float(cfg.get("mss_cap_usd", 1000.0))
        self.mss_cool_s  = float(cfg.get("mss_cooldown_s", 12.0))
        self._mss_fetch_s   = float(cfg.get("mss_fetch_s", 3.0))
        self._last_mss_fetch = 0.0
        self._mss_last_ts    = None
        # 5-second candles are built LIVE from the 1-second price feed (like the
        # bricks) - the REST get_bars(5s) endpoint returns stale/cached data and
        # can't drive a real-time entry. get_bars is used ONLY to seed history.
        self._mss_bkt   = None      # current 5s bucket start (unix seconds)
        self._mss_o = self._mss_h = self._mss_l = self._mss_c = 0.0
        if self.mss_mode and self.renko_flip and hasattr(self.sig, "configure_mss"):
            self.sig.configure_mss(True,
                ema_len=int(cfg.get("ema_len", 8)),
                stop_buf=float(cfg.get("rev_stop_buf_pts", 2.0)),
                confirm=int(cfg.get("mss_confirm", 1)),
                zz_thresh=float(cfg.get("mss_zz_pts", 15.0)))
            _cap_pts = self.mss_cap_usd / (POINT_VALUE * self.qty)
            print(f"[{self.name}] MSS-5s ON (Abhi shift strategy): 8-EMA shift on 5s "
                  f"candles, NO 1h gate, zz={cfg.get('mss_zz_pts', 15.0)}pt "
                  f"confirm={cfg.get('mss_confirm', 1)} | stop=structural capped "
                  f"${self.mss_cap_usd:.0f} (~{_cap_pts:.0f}pt) | target={self.mss_tp_r}x risk",
                  flush=True)
        self._rl = LevelEntryRL(
            cfg.get("rl_levels_file", f"rl_levels_{self.name}.json"),
            enabled=self.levels_rl,
            min_samples=int(cfg.get("rl_min_samples", 8)),
            skip_margin=float(cfg.get("rl_skip_margin", 0.0)),
            probe_every=int(cfg.get("rl_probe_every", 6)))
        self._rl_pending = None       # (state) of the currently-open RL-taken trade
        self._entry_level_ctx = None  # level snapshot captured at entry, for the log
        # trend_tf_mins (client 22-Jul): higher-timeframe Renko trend filter. Only
        # enter when EVERY listed higher-TF Renko (e.g. 15m + 30m) points the same way
        # as the 5m signal - so the bot plays with the trend instead of shorting into
        # small pullbacks. [] = off. Each gets its own RenkoFlip engine, same brick.
        self.trend_tf_mins = [int(m) for m in cfg.get("trend_tf_mins", [])]
        self._trend_sigs = {}
        self._trend_last_ts = {}
        self._last_trend_fetch = 0.0
        if self.renko_flip:
            for m in self.trend_tf_mins:
                self._trend_sigs[m] = RenkoFlip(
                    brick_size=float(cfg.get("brick_size", 2.0)),
                    confirm=int(cfg.get("confirm_bricks", 2)),
                    entry_frac=self.entry_confirm_frac)
                self._trend_last_ts[m] = None
        elif self.renko_mode:
            self.sig = NQSignalRenko(
                brick_size=float(cfg.get("brick_size", 1.0)),
                group=int(cfg.get("renko_group", 6)),
                confirm_frac=float(cfg.get("confirm_frac", 0.70)),
                trend_on_htf=bool(cfg.get("trend_on_htf", True)),
                use_time_htf=self.use_time_htf)
        else:
            self.sig = NQSignal()
        self._conf5 = []            # confirmed 5-min bars, oldest -> newest
        self._conf30 = []           # confirmed 30-min bars
        self._last5_ts = None       # newest confirmed 5m bar start-time seen
        self._last30_ts = None
        self._last_htf_fetch = 0.0  # throttle the 5m/30m refetch
        self._htf_interval_s = float(cfg.get("htf_refresh_s", 5.0))
        self._live_price = 0.0

        # ---- account-sync (master -> follower mirror) -----------------------
        # Two accounts under two DIFFERENT logins can only run as two separate
        # processes, and two independent signal engines drift apart: they latch
        # the breakout a second apart, one skips a signal on slippage the other
        # takes, they reconnect at different moments. Those little differences
        # compound, so by end of day the two accounts have different trades and
        # different P&L. To keep them matched we nominate ONE account the master:
        # it alone runs the strategy and writes every ENTRY/EXIT to a shared bus
        # file. The follower runs NO strategy of its own - it just mirrors the
        # bus: same side, same instant (its own fill, market-backed so it never
        # misses the trade), and it closes when the master closes.
        #   sync_role: "master" | "follower" | absent (standalone, old behaviour)
        self.sync_role = cfg.get("sync_role")
        self.sync_bus  = cfg.get("sync_bus")
        self._sync_seq = 0          # highest bus seq emitted (master) / consumed (follower)
        self._last_sig_log = 0.0    # throttle for the "why no signal" diagnostic

    async def broker_balance(self):
        """Realised account balance, straight from the platform. None if unknown."""
        try:
            accts = await asyncio.wait_for(self.suite.client.list_accounts(), timeout=10.0)
        except Exception as e:
            print(f"[{self.name}] balance read failed: {e!r}")
            return None
        for ac in (accts or []):
            if getattr(ac, "name", None) == self.account_name:
                return float(ac.balance)
        return None

    async def sync_daily_pnl(self):
        """Reconcile the day's P&L to the broker. The broker always wins.

        The running tally in exit() stays - it reacts instantly, with no API call,
        so the breaker still bites the moment a trade goes bad. But whenever the
        platform will tell us the truth, we take the truth.
        """
        bal = await self.broker_balance()
        if bal is None:
            return
        if self.day_open_balance is None:
            # No anchor for today. Do NOT reconstruct one from self.daily_loss -
            # that number is exactly the thing we distrust, and seeding from it
            # would enshrine the drift instead of correcting it. Anchor here and
            # count the day from this point on. Once the anchor is persisted, every
            # later restart is exact.
            self.day_open_balance = bal
            self.daily_loss = 0.0
            print(f"[{self.name}] no day anchor - anchoring to broker balance "
                  f"${bal:,.2f} and counting today's P&L from here")
            return
        truth = bal - self.day_open_balance
        if abs(truth - self.daily_loss) > 1.0:
            print(f"[{now_str()}] [{self.name}] day P&L corrected to the broker: "
                  f"${self.daily_loss:+.0f} (mine) -> ${truth:+.0f} (theirs)")
        self.daily_loss = truth

    def user_hub_down(self):
        """True when the order/account (SignalR user) hub is DOWN while market
        data may still be flowing.

        This is the failure the price-feed watchdog misses: the market hub keeps
        ticking (get_current_price works, last_tick_ts stays fresh) so the feed
        looks alive, but the user hub - the channel that places and tracks orders
        - has silently dropped, so the bot cannot trade. Seen live 17-Jul on Abhi
        ("User hub disconnected" + P&L desync) with the market feed still up.

        Read defensively: an unknown/missing status is NOT treated as down, so a
        library change can only make this no-op, never false-trigger a reconnect.
        """
        if not self.suite:
            return False
        rc = getattr(self.suite, "realtime_client", None) or getattr(self.suite, "realtime", None)
        if rc is None:
            return False
        uc = getattr(rc, "user_connected", None)
        if uc is None:
            return False
        return not uc

    def note_bad_tick(self, why):
        """A price read that failed. Loud, but not 10x/second loud.

        This used to be `except Exception: continue` - no message, no counter,
        no trace. The bot spun on it for five hours and nobody could tell.
        """
        self.bad_ticks += 1
        now = time.time()
        if now - self._last_tick_gripe > 30.0:
            self._last_tick_gripe = now
            quiet = now - self.last_tick_ts
            print(f"[{now_str()}] [{self.name}] no price ({self.bad_ticks} reads failed, "
                  f"last good one {quiet:.0f}s ago): {why}", flush=True)

    def live_price_is_sane(self, px):
        """Reject a phantom/stale realtime quote before it reaches the strategy.

        The feed occasionally returns a price 100+ points off the real market
        (see MAX_TICK_JUMP note). Such a tick, traded on, sets the stop on the
        wrong side of the ACTUAL fill and the platform flattens instantly -
        that was the 22-Jul '$40 SL' churn. An isolated far read is dropped and
        the last good price is kept; only when several reads in a row agree at
        the far level do we accept it as a genuine gap and resync. Fail-open on
        the very first read (no reference yet)."""
        ref = self._last_good_px
        if ref is None:
            self._last_good_px = px
            self._px_reject_streak = 0
            return True
        if abs(px - ref) <= MAX_TICK_JUMP:
            self._last_good_px = px
            self._px_reject_streak = 0
            return True
        self._px_reject_streak += 1
        if self._px_reject_streak >= 3:
            print(f"[{now_str()}] [{self.name}] price resync {ref:.2f} -> "
                  f"{px:.2f} after {self._px_reject_streak} agreeing reads "
                  f"(treating as a real gap)", flush=True)
            self._last_good_px = px
            self._px_reject_streak = 0
            return True
        self.note_bad_tick(f"phantom quote {px:.2f} vs last good {ref:.2f} "
                           f"(|delta|={abs(px - ref):.1f} > {MAX_TICK_JUMP:.0f}pt)")
        return False

    async def _resolve_account(self):
        """When account_auto is on, resolve self.account_name off the LIVE login so
        a reset / replaced practice account is adopted automatically. Keeps the
        configured account if it is still present and tradeable; otherwise picks the
        NEWEST tradeable account whose name contains account_match and is not in
        account_exclude. Never raises - on any failure it leaves account_name as-is
        so the normal connect path (and its error) is unchanged."""
        if not self.account_auto:
            return
        import project_x_py as px
        os.environ["PROJECT_X_USERNAME"] = self.username
        os.environ["PROJECT_X_API_KEY"]  = self.api_key
        try:
            async with px.ProjectX.from_env() as c:
                await asyncio.wait_for(c.authenticate(), timeout=15.0)
                accts = await asyncio.wait_for(c.list_accounts(), timeout=15.0)
        except Exception as e:
            print(f"[{self.name}] account auto-resolve failed ({e!r}) - "
                  f"using configured {self.account_name}")
            return
        allnames = [getattr(a, "name", "") or "" for a in (accts or [])]

        def tradeable(a):
            n = getattr(a, "name", "") or ""
            if self.account_match and self.account_match not in n:
                return False
            if any(x in n for x in self.account_exclude):
                return False
            return getattr(a, "canTrade", None) is not False   # None/True both ok

        cands = [a for a in (accts or []) if tradeable(a)]
        if self.account_name in [getattr(a, "name", "") for a in cands]:
            print(f"[{self.name}] account auto: configured {self.account_name} still valid "
                  f"(of {allnames})")
            return
        if not cands:
            print(f"[{self.name}] account auto: NO tradeable match for "
                  f"'{self.account_match}' (excl {self.account_exclude}) among "
                  f"{allnames} - keeping {self.account_name}")
            return

        def idnum(a):
            tail = (getattr(a, "name", "") or "").rsplit("-", 1)[-1]
            return int(tail) if tail.isdigit() else (getattr(a, "id", 0) or 0)

        chosen = sorted(cands, key=idnum)[-1]   # highest id suffix = newest
        old = self.account_name
        self.account_name = getattr(chosen, "name")
        print(f"[{self.name}] account AUTO-RESOLVED: {old} -> {self.account_name} "
              f"(newest tradeable of {[getattr(a, 'name', '') for a in cands]}; "
              f"all {allnames})")

    # ------------------------------------------------------------------ conn
    async def connect(self):
        patch_position_model()      # MUST run before any position is read
        from project_x_py import TradingSuite
        await self._resolve_account()   # adopt a reset/replaced account before we bind it
        os.environ["PROJECT_X_USERNAME"]     = self.username
        os.environ["PROJECT_X_API_KEY"]      = self.api_key
        os.environ["PROJECT_X_ACCOUNT_NAME"] = self.account_name
        self.suite = await TradingSuite.create(
            instruments=[self.symbol], timeframes=["1sec"], initial_days=1)
        self.ctx = self.suite[self.symbol]
        self.connected = True

        # Feed the flow tracker straight off the quote stream. Wrapped so that a
        # bad quote can never take the trading loop down with it.
        from project_x_py import EventType

        async def _on_quote(evt):
            try:
                self.flow.on_quote(getattr(evt, "data", evt) or {}, time.time())
            except Exception:
                pass

        try:
            await self.suite.on(EventType.QUOTE_UPDATE, _on_quote)
            print(f"[{self.name}] order-flow recorder attached (quote tape -> "
                  f"aggressor delta). RECORD ONLY - it does not gate any trade.")
        except Exception as e:
            print(f"[{self.name}] flow recorder not attached: {e!r}")
        v = viability(self.brick_size, 0.79, point_value=POINT_VALUE)
        print(f"[{self.name}] Connected: {self.account_name} | contract {self.ctx.instrument_info.id}")
        print(f"[{self.name}] Renko brick={self.brick_size} | flip-stop={v['stop_pts']}pts "
              f"(${v['stop_dollars']:.0f}) | entry cap={self.max_entry_slip_ticks} tick(s) "
              f"(${self.max_entry_slip_ticks*TICK*POINT_VALUE:.2f})")
        if not v["ok"]:
            print(f"[{self.name}] *** WARNING: at market-order slippage (0.79pt) this brick size "
                  f"costs {v['cost_as_pct_of_stop']:.0f}% of the stop. Only the "
                  f"{self.max_entry_slip_ticks}-tick entry cap makes it survivable. ***")

        # Rebuild the chart from real history so we never start cold (see method).
        await self.warmup_from_history()

    async def reconnect(self):
        """Rebuild this account's SDK connection IN-PROCESS, without crashing.

        This is the client's "connection that recovers instead of restarting".
        When the data feed dies (SignalR drop, HTTP session wedge) we tear down
        the old suite and build a brand-new one via connect() - fresh socket,
        fresh HTTP session - all inside the running process.

        Why this is SAFE while holding a position:
          * the protective stop is a REAL resting order on the exchange; it
            guards the position for the entire time we are disconnected;
          * the process never dies, so self.position / entry_price /
            contracts_held / the stop order id all stay intact in memory - we do
            NOT re-adopt from a (possibly lying) position feed, and we do NOT
            run _sync_flat, so there is no chance of sweeping a live stop;
          * connect() re-runs warmup so the brick/EMA chart is rebuilt to the
            exact same state, and the run loop's reconcile() re-verifies the
            position against the platform on the fresh connection within 2s.

        Returns True on success. On a locked/closed account (a thing no reconnect
        can fix) it sets health UNSAFE and returns False without hammering.
        """
        self.connected = False
        self.health = "RECONNECTING"
        if self.suite:
            try:
                await asyncio.wait_for(self.suite.disconnect(), timeout=5.0)
            except Exception:
                pass
        self.suite = None
        self.ctx = None
        try:
            await self.connect()          # fresh suite + warmup; position untouched
        except Exception as e:
            self._reconnect_tries += 1
            msg = str(e).lower()
            if any(k in msg for k in FATAL_REJECTS):
                print(f"[{now_str()}] [{self.name}] *** reconnect refused: {e}. That "
                      f"is an account-side lock/close, not a feed drop - retrying "
                      f"cannot fix it, so I am not retrying. Clear it on TopstepX "
                      f"and restart me. ***", flush=True)
                self.health = "UNSAFE"
                return False
            print(f"[{now_str()}] [{self.name}] reconnect attempt {self._reconnect_tries} "
                  f"failed: {e!r}", flush=True)
            self.health = "DEGRADED"
            return False
        # connect() set self.connected = True and re-registered the quote handler.
        self._api_fails = 0
        self.bad_ticks = 0
        self.last_tick_ts = time.time()
        self._reconnect_tries = 0
        self._reconnect_cooldown = RECONNECT_COOLDOWN_BASE
        self._user_down_since = None
        self.health = "HEALTHY"
        held = ("flat" if self.position == 0
                else f"{'LONG' if self.position > 0 else 'SHORT'} @ {self.entry_price}")
        print(f"[{now_str()}] [{self.name}] reconnected in-process - no restart. "
              f"Position kept ({held}); reconcile will re-verify it on the new feed.",
              flush=True)
        return True

    # --------------------------------------------------------------- helpers
    def _session_open_dt(self, now=None):
        """Most recent 6pm ET at or before `now` - the open of the current
        overnight session. Warmup replays from here so both accounts, and every
        restart, rebuild the identical brick/EMA chart from one fixed point."""
        now = now or datetime.now(ET)
        open_today = now.replace(hour=18, minute=0, second=0, microsecond=0)
        return open_today if now >= open_today else open_today - timedelta(days=1)

    def _bars_to_dicts(self, df):
        """Normalise a get_bars dataframe to a list of {ts(NY),open,high,low,close}
        sorted oldest->newest. Returns [] on anything unexpected."""
        if df is None or len(df) == 0:
            return []
        recs = df.to_dicts() if hasattr(df, "to_dicts") else df.to_dict("records")
        out = []
        for r in recs:
            t = r["timestamp"]
            out.append({"ts": t.astimezone(ET),
                        "open": float(r["open"]), "high": float(r["high"]),
                        "low": float(r["low"]), "close": float(r["close"])})
        return sorted(out, key=lambda r: r["ts"])

    async def _get_bars(self, interval, unit, days):
        """Fetch bars, reusing the live suite client, falling back to from_env. EVERY
        path is timeout-bounded and returns [] on total failure - a slow/flaky feed
        must NEVER freeze the trading loop. (Unbounded fallback awaits froze the bot
        on 22-Jul once the trend filter tripled the _get_bars traffic.)"""
        try:
            return await asyncio.wait_for(
                self.suite.client.get_bars(self.symbol, days=days,
                                           interval=interval, unit=unit), timeout=15.0)
        except Exception as e:
            print(f"[{self.name}] get_bars({interval},{unit}) primary failed: {e!r}")
        try:
            import project_x_py as px
            async with px.ProjectX.from_env() as client:
                await asyncio.wait_for(client.authenticate(), timeout=10.0)
                return await asyncio.wait_for(
                    client.get_bars(self.symbol, days=days,
                                    interval=interval, unit=unit), timeout=15.0)
        except Exception as e:
            print(f"[{self.name}] get_bars({interval},{unit}) fallback failed: {e!r}")
            return []

    def _feed_htf(self, bars5, bars30, live_price):
        """Feed any NEWLY CONFIRMED 5m/30m bars into the strategy core. A bar is
        confirmed once the next one has started (we only ever act on closed bars,
        never the forming one - the request.security(lookahead_off) rule)."""
        # drop the still-forming last bar of each series before confirming
        c5 = bars5[:-1] if bars5 else []
        c30 = bars30[:-1] if bars30 else []
        for b in c5:
            if self._last5_ts is None or b["ts"] > self._last5_ts:
                self._conf5.append(b)
                self._last5_ts = b["ts"]
                self.sig.on_new_5m(self._conf5, live_price)
        for b in c30:
            if self._last30_ts is None or b["ts"] > self._last30_ts:
                self._conf30.append(b)
                self._last30_ts = b["ts"]
                self.sig.on_new_30m(self._conf30)

    async def _refresh_htf(self, live_price):
        """Throttled re-fetch of the 5m/30m bars and feed of confirmed ones."""
        now = time.time()
        if now - self._last_htf_fetch < self._htf_interval_s:
            return
        self._last_htf_fetch = now
        try:
            b5 = self._bars_to_dicts(await self._get_bars(5, 2, 5))
            b30 = self._bars_to_dicts(await self._get_bars(30, 2, 15))
            self._feed_htf(b5, b30, live_price)
        except Exception as e:
            print(f"[{self.name}] HTF refresh failed: {e!r}")

    async def _refresh_htf_renko(self, live_price):
        """Renko + real higher timeframe: throttled refetch of the ACTUAL 5m/30m
        bars, feeding only the trend + filter (the brick stream drives the
        breakout/flip). Confirmed bars only - drop the still-forming last one."""
        now = time.time()
        if now - self._last_htf_fetch < self._htf_interval_s:
            return
        self._last_htf_fetch = now
        try:
            b5 = self._bars_to_dicts(await self._get_bars(5, 2, 5))
            b30 = self._bars_to_dicts(await self._get_bars(30, 2, 15))
            self.sig.set_htf(b5[:-1] if b5 else [], b30[:-1] if b30 else [])
        except Exception as e:
            print(f"[{self.name}] renko HTF refresh failed: {e!r}")

    async def _feed_renkoflip(self, price, ts):
        """Feed newly-CLOSED 5-minute bars into the RenkoFlip engine (throttled).
        The forming 5m bar is dropped; the ghost brick is read off the live price."""
        now = time.time()
        if now - self._last_htf_fetch < self._htf_interval_s:
            return
        self._last_htf_fetch = now
        try:
            b5 = self._bars_to_dicts(await self._get_bars(
                self._etf_i, self._etf_u, self._etf_feed_days))
            for b in (b5[:-1] if b5 else []):
                if self._last5_ts is None or b["ts"] > self._last5_ts:
                    self.sig.on_bar(b["close"], b["ts"].timestamp())
                    self._last5_ts = b["ts"]
        except Exception as e:
            print(f"[{self.name}] renkoflip {self._etf_label} feed failed: {e!r}")

    async def _feed_trend(self, price, ts):
        """Feed each higher-timeframe trend Renko its newly-CLOSED bars (throttled
        to ~30s - these bars only change every 15/30/60 min)."""
        if not self._trend_sigs:
            return
        now = time.time()
        if now - self._last_trend_fetch < max(30.0, self._htf_interval_s):
            return
        self._last_trend_fetch = now
        for m in self.trend_tf_mins:
            try:
                bars = self._bars_to_dicts(await self._get_bars(m, 2, 3))
                for b in (bars[:-1] if bars else []):
                    if self._trend_last_ts[m] is None or b["ts"] > self._trend_last_ts[m]:
                        self._trend_sigs[m].on_bar(b["close"], b["ts"].timestamp())
                        self._trend_last_ts[m] = b["ts"]
            except Exception as e:
                print(f"[{self.name}] trend {m}m feed failed: {e!r}")

    def _ema_color(self, closes, fast, slow):
        """Colour a close series by fast-EMA vs slow-EMA: green when fast>slow (up
        trend), red when fast<slow. None until there are enough bars for the slow
        EMA so we never read a half-warmed trend."""
        if not closes or len(closes) < slow:
            return None
        kf, ks = 2.0 / (fast + 1.0), 2.0 / (slow + 1.0)
        ef = es = closes[0]
        for c in closes[1:]:
            ef = c * kf + ef * (1.0 - kf)
            es = c * ks + es * (1.0 - ks)
        return "green" if ef >= es else "red"

    async def _refresh_h1_trend(self):
        """Throttled 1-hour trend read for the reversal gate: fetch 1h bars, colour
        them by the fast/slow EMA, and push the colour into the engine. Keeps the
        last good value on a fetch error so a flaky feed never flips the gate."""
        if not getattr(self, "reversal_mode", False):
            return
        now = time.time()
        if now - self._h1_last_fetch < self.h1_refresh_s:
            return
        self._h1_last_fetch = now
        try:
            bars = self._bars_to_dicts(await self._get_bars(60, 2, self.h1_days))
            closes = [b["close"] for b in (bars[:-1] if bars else [])]   # confirmed 1h bars only
            col = self._ema_color(closes, self.h1_fast, self.h1_slow)
            if col is not None:
                self._h1_trend = col
            self.sig.set_h1_trend(self._h1_trend)
        except Exception as e:
            print(f"[{self.name}] 1h trend refresh failed: {e!r}")

    def _tf_dir(self, eng, price):
        """Current Renko direction of one higher-TF engine: the forming ghost if
        there is one, else the last confirmed brick. 'green' | 'red' | None."""
        d = eng._ghost_color(price)
        if not d and eng.eng.bricks:
            d = eng.eng.bricks[-1]["direction"]
        return d

    def _trend_ok(self, sig, price):
        """True if EVERY higher-TF trend Renko agrees with the 5m signal direction.
        No trend TFs configured -> always True (filter off). A TF with no data yet
        blocks (we do not trade a direction the trend can't confirm)."""
        if not self._trend_sigs:
            return True
        want = "green" if sig == "LONG" else "red"
        return all(self._tf_dir(e, price) == want for e in self._trend_sigs.values())

    def _trend_str(self, price):
        return ",".join(f"{m}={self._tf_dir(e, price)}"
                        for m, e in self._trend_sigs.items()) or "off"

    async def _warmup_renkoflip(self):
        """Seed the 5-min Renko bricks from real 5-minute history so the flip is
        warm at startup. NO orders, NO position."""
        try:
            b5 = self._bars_to_dicts(await self._get_bars(
                self._etf_i, self._etf_u, self._etf_warm_days))
            conf = b5[:-1] if b5 else []
            for b in conf:
                self.sig.on_bar(b["close"], b["ts"].timestamp())
                self._last5_ts = b["ts"]
            # Warm the 15-second Bollinger Bands from real 15s history so the entry
            # trigger is live at startup instead of blank for the first ~5 minutes.
            if getattr(self.sig, "ema_len", 0) > 0 and hasattr(self.sig, "push_bb_close"):
                try:
                    bb = self._bars_to_dicts(await self._get_bars(15, 1, 1))
                    for b in (bb[:-1] if bb else []):
                        self.sig.push_bb_close(b["open"], b["high"], b["low"], b["close"])
                    print(f"[{self.name}] warmup (RENKO-FLIP): fed {len(bb[:-1]) if bb else 0} "
                          f"15s bars into BB{self.sig.bb_len}/{self.sig.bb_std}sd")
                except Exception as e:
                    print(f"[{self.name}] 15s BB warmup failed: {e!r}")
            # prime the 1h trend gate so the reversal is aligned from the first tick
            if getattr(self, "reversal_mode", False):
                self._h1_last_fetch = 0.0
                await self._refresh_h1_trend()
                # fold all the historical bricks into the reversal tracker WITHOUT
                # arming, then open the gate so live breaks fire (even batched ones)
                if hasattr(self.sig, "mark_reversal_warm"):
                    self.sig.mark_reversal_warm()
                print(f"[{self.name}] reversal 1h-trend primed: {self._h1_trend} | "
                      f"reversal armed-live from now")
            print(f"[{self.name}] warmup (RENKO-FLIP): fed {len(conf)} {self._etf_label} bars | "
                  f"{self.sig.status()} | entry_frac={self.entry_confirm_frac} SL=first-brick exit={'ghost-flip' if self.ghost_exit else 'stop/trail'} | "
                  f"session 18:00-16:00 ET (6pm-4pm)")
            for m in self.trend_tf_mins:
                bars = self._bars_to_dicts(await self._get_bars(m, 2, 10))
                conf = bars[:-1] if bars else []
                for b in conf:
                    self._trend_sigs[m].on_bar(b["close"], b["ts"].timestamp())
                    self._trend_last_ts[m] = b["ts"]
                print(f"[{self.name}] trend warmup {m}m: fed {len(conf)} bars | "
                      f"{self._trend_sigs[m].status()}")
        except Exception as e:
            print(f"[{self.name}] renkoflip warmup FAILED ({e!r}) - starting cold.")

    def _entry_gates_ok(self, sig, price, nbricks):
        """The same guards the market-entry path applies: tradeable, a NEW brick
        since the last stop-out, the 12s re-entry throttle, and trend agreement."""
        return (bool(sig) and self._can_trade()
                and nbricks > self._reentry_block_bricks
                and (time.time() - getattr(self, "_last_entry_ts", 0.0)) >= 12.0
                and self._trend_ok(sig, price))

    async def _cancel_pending_entry(self, why=""):
        """Cancel and forget the resting entry limit, if any."""
        pend = self._pending_entry
        self._pending_entry = None
        if not pend:
            return
        try:
            await asyncio.wait_for(self.ctx.orders.cancel_order(pend["order_id"]), timeout=8.0)
        except Exception as e:
            print(f"[{self.name}] cancel pending entry {pend['order_id']} failed: {e!r}")
        else:
            print(f"[{now_str()}] [{self.name}] pending {pend['dir']} limit "
                  f"@ {pend['limit_price']:.2f} cancelled{(' - ' + why) if why else ''}")

    def _die_if_account_dead(self, err):
        """A reset/closed account rejects EVERY order forever ("Account is closed").
        The client resets these practice accounts constantly, and account_auto only
        runs at connect - so a mid-run reset strands us on a dead account, and any
        per-tick order attempt then floods the book with rejects (the "million signals"
        bug, client 22-Jul). Exit hard: systemd Restart=always brings us back and
        connect()'s account_auto re-resolves to the new live account."""
        m = str(err).lower()
        if any(k in m for k in ("account is closed", "account not found",
                                "account is not active", "account not active",
                                "account is disabled", "account disabled")):
            print(f"[{now_str()}] [{self.name}] ACCOUNT DEAD ({err!r}) - exiting for "
                  f"systemd restart + account_auto re-resolve.", flush=True)
            os._exit(1)

    async def _place_limit_entry(self, sig, price):
        """Rest a passive limit at anchor + offset. Records self._pending_entry and
        self._sl_level so a fill (adopted here or by reconcile) is protected correctly."""
        # Throttle: never re-attempt a failing placement every tick (order-spam guard).
        if time.time() - getattr(self, "_last_entry_attempt", 0.0) < 2.0:
            return
        self._last_entry_attempt = time.time()
        try:
            brick = self.sig.eng.bricks[-1]
        except Exception:
            return
        anchor = float(brick["close"])
        sl_level = float(brick["open"])
        off = self.limit_entry_offset_ticks * TICK
        # Directional (client 22-Jul): rest the limit 0.25 ABOVE the brick for a LONG,
        # 0.25 BELOW it for a SHORT - a shallow pullback into the brick fills us.
        raw = (anchor + off) if sig == "LONG" else (anchor - off)
        limit_price = round(raw / TICK) * TICK
        side = 0 if sig == "LONG" else 1               # 0=BUY 1=SELL
        try:
            resp = await asyncio.wait_for(self.ctx.orders.place_limit_order(
                contract_id=self.ctx.instrument_info.id,
                side=side, size=self.qty, limit_price=limit_price), timeout=10.0)
        except Exception as e:
            self._die_if_account_dead(e)
            print(f"[{self.name}] entry limit place FAILED: {e!r}")
            return
        oid = getattr(resp, "orderId", None) if resp else None
        if not (resp and getattr(resp, "success", False) and oid):
            why = getattr(resp, "errorMessage", None) if resp else "no response"
            print(f"[{self.name}] entry limit rejected ({why})")
            return
        self._sl_level = sl_level
        self._pending_entry = {"order_id": oid, "dir": sig, "level": anchor,
                               "anchor": anchor, "limit_price": limit_price,
                               "sl_level": sl_level, "ts": time.time()}
        print(f"[{now_str()}] [{self.name}] RESTING {sig} limit @ {limit_price:.2f} "
              f"(brick {anchor:.2f} {'+' if sig == 'LONG' else '-'}{off:.2f}, "
              f"SL {sl_level:.2f}) - waiting for pullback")

    async def _adopt_limit_fill(self, truth):
        """Our resting entry limit filled (platform now holds `truth`). Set the
        position up from the pending record - correct entry price, first-brick SL,
        protective stop - then clear the pending order."""
        pend = self._pending_entry
        self._pending_entry = None
        self.position = 1 if truth > 0 else -1
        self.contracts_held = abs(truth)
        self.intended_entry = pend["level"] if pend else None
        self.entry_price = pend["limit_price"] if pend else float(
            await self.ctx.data.get_current_price())
        self.entry_time = time.time()
        if pend:
            self._sl_level = pend["sl_level"]
        self._mfe_pts = self._mae_pts = 0.0
        self._mfe_price = self._mae_price = self.entry_price
        self._peak = self.entry_price
        self.trades_today += 1
        self._last_entry_ts = time.time()
        try:
            self.flow_at_entry = self.flow.snapshot(time.time())
        except Exception:
            self.flow_at_entry = {}
        self._stop_order_id = None
        self._stop_price = None
        await self._sync_protective_stop()
        d = "LONG" if self.position == 1 else "SHORT"
        print(f"[{now_str()}] [{self.name}] >>> {d} x{self.contracts_held} LIMIT-FILLED "
              f"@ {self.entry_price:.2f} | flip-stop @ "
              f"{self._stop_price if self._stop_price else 0:.2f} | "
              f"trade {self.trades_today}/{self.max_trades_per_day}")
        self._sync_emit("ENTRY", d)

    async def _manage_limit_entry(self, price, ts, nbricks):
        """Passive-limit entry loop (runs only while FLAT). Places ONE resting limit on
        a signal, adopts it when it fills, replaces it when a new signal supersedes it,
        and drops it after a TTL if it never fills. Fill is detected by the platform
        position, NOT by search_open_orders - a just-placed order lags the working-order
        feed by ~1s, and re-checking it there made the bot spam duplicate orders."""
        sig = self.sig.signal(price)
        pend = self._pending_entry

        if pend is not None:
            truth = await self.platform_pos()
            if truth is None:
                return True                             # can't tell - leave it resting
            if truth != 0:
                await self._adopt_limit_fill(truth)     # FILLED
                return True
            # A genuinely NEW signal (direction flip, or the level moved a whole brick)
            # supersedes the resting order.
            try:
                cur_anchor = float(self.sig.eng.bricks[-1]["close"])
            except Exception:
                cur_anchor = pend["anchor"]
            if sig and self._entry_gates_ok(sig, price, nbricks) and (
                    sig != pend["dir"] or abs(cur_anchor - pend["anchor"]) >= self.brick_size):
                await self._cancel_pending_entry("superseded by new signal")
                await self._place_limit_entry(sig, price)
                return True
            # Never filled and no new signal within the TTL -> cancel and let a fresh
            # signal re-arm. Guards against a stale limit resting forever.
            if (time.time() - pend["ts"]) > self.limit_entry_ttl_s:
                await self._cancel_pending_entry("TTL expired, no fill")
            return True

        # no resting order -> place one on a fresh, gated signal
        if sig and self._entry_gates_ok(sig, price, nbricks):
            await self._place_limit_entry(sig, price)
        return True

    async def _arm_confirm(self, sig, price):
        """Start a confirm-entry watch for a fresh signal: record the entry brick
        level + first-brick SL, wait for the pullback then the resume."""
        try:
            brick = self.sig.eng.bricks[-1]
        except Exception:
            return
        level = float(brick["close"])
        sl = float(brick["open"])
        # Pullback is measured from the ARM price (where price actually is when the
        # signal fires), NOT the brick close - a green ghost sits ~0.6 brick ABOVE the
        # close, so "0.25 below the close" was really a ~3pt drop that almost never
        # happened and the bot never entered. 0.25 below the arm price is the small
        # pullback the client means ("dip .25 from entry").
        self._confirm_state = {"dir": sig, "level": level, "anchor": level,
                               "arm_price": float(price), "sl_level": sl,
                               "phase": "PULLBACK", "extreme": None, "ts": time.time()}
        side = "below" if sig == "LONG" else "above"
        print(f"[{now_str()}] [{self.name}] ARMED {sig} confirm-entry @ {price:.2f} "
              f"(brick {level:.2f}) - wait {self.confirm_pullback:.2f} dip {side}, then "
              f"{self.confirm_resume:.2f} resume on 1s (SL {sl:.2f})")

    async def _confirm_market_enter(self, cs, price):
        """The move resumed after the pullback: enter at MARKET. SL is recomputed from
        the CURRENT first brick (the protective-side guard + max_stop cap keep it sane)."""
        try:
            self._sl_level = float(self.sig.eng.bricks[-1]["open"])
        except Exception:
            self._sl_level = cs["sl_level"]
        self._confirm_state = None
        print(f"[{now_str()}] [{self.name}] confirm-entry {cs['dir']} CONFIRMED "
              f"(resumed {self.confirm_resume:.2f} from {cs['extreme']:.2f}) -> MARKET")
        await self.enter(cs["dir"], price, price, market_ok=True)
        if self.position != 0:
            self._peak = self.entry_price or price

    async def _manage_confirm_entry(self, price, ts, nbricks):
        """No-falling-knife entry (client 22-Jul). Arms on a signal, waits for the
        pullback, then a 1-second resume in the signal direction, then MARKET-enters.
        Handles the whole flat state itself."""
        sig = self.sig.signal(price)
        cs = self._confirm_state

        if cs is not None:
            try:
                cur_anchor = float(self.sig.eng.bricks[-1]["close"])
            except Exception:
                cur_anchor = cs["anchor"]
            # a genuinely NEW signal (flip, or level moved a whole brick) re-arms
            if sig and self._entry_gates_ok(sig, price, nbricks) and (
                    sig != cs["dir"] or abs(cur_anchor - cs["anchor"]) >= self.brick_size):
                await self._arm_confirm(sig, price)
                return True
            # NOTE (client 22-Jul): do NOT void just because price dipped through the
            # first-brick stop. A deeper pullback that still agrees with the conditions
            # (trend aligned) is fine - it just means a BETTER entry. We only require
            # the trend to still agree at the moment we confirm+enter (checked below).
            if (time.time() - cs["ts"]) > self.limit_entry_ttl_s:
                self._confirm_state = None
                return True
            if cs["phase"] == "PULLBACK":
                ref = cs.get("arm_price", cs["level"])
                hit = ((price <= ref - self.confirm_pullback) if cs["dir"] == "LONG"
                       else (price >= ref + self.confirm_pullback))
                if hit:
                    cs["phase"] = "CONFIRM"
                    cs["extreme"] = price
                    print(f"[{now_str()}] [{self.name}] confirm-entry {cs['dir']} PULLBACK "
                          f"hit @ {price:.2f} - watching 1s for {self.confirm_resume:.2f} resume")
                return True
            # CONFIRM phase: track the extreme, enter when price resumes our way -
            # but ONLY while the trend still agrees (client: "as long as the 5 min is
            # aligning"). If the trend has flipped, hold; a new signal will re-arm.
            if cs["dir"] == "LONG":
                cs["extreme"] = min(cs["extreme"], price)
                if price >= cs["extreme"] + self.confirm_resume and self._trend_ok("LONG", price):
                    await self._confirm_market_enter(cs, price)
            else:
                cs["extreme"] = max(cs["extreme"], price)
                if price <= cs["extreme"] - self.confirm_resume and self._trend_ok("SHORT", price):
                    await self._confirm_market_enter(cs, price)
            return True

        # no active watch -> arm one on a fresh gated signal
        if sig and self._entry_gates_ok(sig, price, nbricks):
            await self._arm_confirm(sig, price)
        return True

    async def _on_price_renkoflip(self, price, ts):
        """5-min Renko flip tick: feed 5m bars, then act. From flat: enter on a
        2-brick signal (in session). In a position: FLIP on a 2-opposite-brick
        signal, else exit on the trailing stop (which also caps the initial loss)."""
        await self._feed_renkoflip(price, ts)
        await self._feed_trend(price, ts)
        await self._refresh_h1_trend()   # reversal 1h-trend gate (throttled)
        # 15s Bollinger-Band entry trigger (client 27-Jul): bucket the live price
        # into 15-second candles so signal() can fade an armed cross off the bands.
        if getattr(self.sig, "ema_len", 0) > 0 and hasattr(self.sig, "feed_bb_tick"):
            self.sig.feed_bb_tick(price, ts)
        nbricks = self.sig.eng._total_bricks   # monotonic; bricks[] caps at 1000 so len() would
                                               # pin the re-entry gate and lock out all entries.

        if time.time() - self._last_sig_log > 60:
            self._last_sig_log = time.time()
            # NB: do NOT call self.sig.signal(price) here - in EMA/BB mode signal()
            # consumes the armed cross when it fires, so logging it would eat a real
            # entry. status() already shows armed direction + band state.
            print(f"[{now_str()}] [{self.name}] renkoflip: {self.sig.status()} | "
                  f"ghost={self.sig._ghost_color(price)} | "
                  f"pos={self.position} | entry_frac={self.entry_confirm_frac} | "
                  f"trail={('%.1fR' % self.trail_risk) if self.trail_risk > 0 else ('%db' % self.trail_bricks)} | "
                  f"trend[{self._trend_str(price)}] | "
                  f"SL@{(self._stop_price if self._stop_price else 0):.2f}")

        if self.position == 0:
            # Confirm-entry mode (client 22-Jul): wait for the pullback + 1s resume,
            # then MARKET-enter. Takes precedence over limit_entry.
            if self.confirm_entry:
                await self._manage_confirm_entry(price, ts, nbricks)
                return
            # Passive-limit entry mode (client 22-Jul): rest a limit at the level and
            # let the pullback fill it, cancel/replace on a new signal. Handles the
            # whole flat state itself.
            if self.limit_entry:
                await self._manage_limit_entry(price, ts, nbricks)
                return
            sig = self.sig.signal(price)
            # only re-enter once a NEW confirmed brick has printed since the last
            # stop-out, so we never re-fire on the same stale trend instantly.
            if sig and self._can_trade() and nbricks > self._reentry_block_bricks \
                    and (time.time() - getattr(self, "_last_entry_ts", 0.0)) >= 12.0 \
                    and self._trend_ok(sig, price):
                # Flip-signals experiment (client 27-Jul): invert direction FIRST so
                # the stop, RL state and order all follow the traded (opposite) side.
                if self.flip_signals:
                    sig = "SHORT" if sig == "LONG" else "LONG"
                # The engine's signal() already applied the STRICT level filter (only
                # fades sitting AT a support/resistance survive). RL now makes the
                # final take/skip call on that approved setup and learns from the P&L.
                if not self._rl_gate(sig, price):
                    return
                # SL. In 9-EMA-cross mode (client 22-Jul) the stop is the brick BEFORE
                # the crossing brick - that gives the trade room to breathe. Otherwise
                # SL = the OPEN of the "first brick" (the last CONFIRMED brick that set
                # the direction): if price comes back to it, that first brick has
                # disappeared -> we are out. confirm=2 guarantees this brick matches our
                # side, so the stop is always on the correct (protective) side.
                try:
                    if getattr(self, "reversal_mode", False):
                        # Reversal (client 28-Jul): the stop sits just past the last
                        # swing point (the lower high we shorted / the higher low we
                        # bought) and the take-profit is sized to a 1:1 risk/reward.
                        rc = getattr(self.sig, "_last_rev_ctx", None) or {}
                        stop_px = rc.get("stop")
                        if stop_px is not None:
                            risk = abs(float(stop_px) - price)
                            if risk < TICK:
                                risk = self.brick_size
                            self._sl_level = float(stop_px)
                            self.max_stop_pts = round(risk + 0.5, 2)   # let the structural stop stand
                            self.take_profit_usd = round(risk * POINT_VALUE * self.qty, 2)  # 1:1
                        else:
                            _sp = self.max_stop_pts if self.max_stop_pts > 0 else (self.brick_size * 4.0)
                            self._sl_level = (price - _sp) if sig == "LONG" else (price + _sp)
                    elif getattr(self.sig, "ema_len", 0) > 0:
                        # BB-fade entry (client 27-Jul): we enter AGAINST the last
                        # 15s move (buy a dip below the lower band / sell a pop above
                        # the upper band), so the pre-cross brick stop is on the wrong
                        # side. Use a clean fixed protective baseline = max_stop_pts
                        # away from entry (8pt / ~$160). RL will own the take-profit.
                        _sp = self.max_stop_pts if self.max_stop_pts > 0 else (self.brick_size * 4.0)
                        self._sl_level = (price - _sp) if sig == "LONG" else (price + _sp)
                    else:
                        self._sl_level = float(self.sig.eng.bricks[-1]["open"])
                except Exception:
                    self._sl_level = None
                # 9-EMA-cross is a BREAKOUT: take the entry at market so we are
                # actually in on the ghost of the 2nd brick ("enter following ghost
                # candle", client 22-Jul). The confirm-N mode keeps the no-chase rule
                # (fill within the 1pt slip cap or skip) to avoid bad fills.
                _mkt = getattr(self.sig, "ema_len", 0) > 0
                await self.enter("LONG" if sig == "LONG" else "SHORT", price, price,
                                 market_ok=_mkt)
                self._peak = self.entry_price or price
                # R (initial stop distance for the 1:1 trail) is captured inside
                # _sync_protective_stop from the ACTUAL protective stop level.
            return

        # In a position: ONE fixed protective stop rests on the platform at the
        # first-brick open (placed at entry). There is NO flip, NO trailing stop and
        # NO auto take-profit - the client takes profit manually. If the stop is ever
        # missing (e.g. a position adopted after a restart), re-rest it; otherwise
        # leave the trade alone and let reconcile() book the exit when the stop, or
        # the client's manual close, fills.
        if self.position == 1:
            self._peak = max(self._peak, price)
        else:
            self._peak = min(self._peak, price)
        # Ghost-flip exit (client 24-Jul): the PRIMARY exit - close the moment the
        # forming (unconfirmed) brick flips to the opposite colour. Checked ahead of
        # TP and the trail; the resting protective stop stays as a hard backstop.
        if await self._maybe_ghost_exit(price):
            return
        # Dollar trailing profit-lock (client 29-Jul): protect a big winner - once up
        # profit_lock_arm_usd, bank it if it gives back more than profit_lock_trail_usd
        # from its peak. Checked ahead of the hard TP so it can lock in early.
        if await self._maybe_profit_lock(price):
            return
        # Hard per-trade profit lock (client 23-Jul): bank the gain at +take_profit_usd
        # before the market can hand it back. Checked every tick, ahead of the trail.
        if await self._maybe_take_profit(price):
            return
        if self._stop_order_id is None and self._stop_price is None:
            await self._sync_protective_stop()
        elif self.trail_bricks > 0 or self.trail_risk > 0:
            await self._maybe_trail_stop()

    async def _warmup_renko(self):
        """Renko variant: rebuild the brick chart from real 1-second history, and
        (if use_time_htf) warm the trend + filter from real 5m/30m history, so we
        start warm not cold. NO orders, NO position."""
        try:
            b1 = self._bars_to_dicts(await self._get_bars(1, 1, 1))   # 1-second bars, ~1 day (capped)
            for b in b1:
                self.sig.on_bar(b["close"], b["ts"].timestamp(), b["close"])
            if self.use_time_htf:
                b5 = self._bars_to_dicts(await self._get_bars(5, 2, 15))
                b30 = self._bars_to_dicts(await self._get_bars(30, 2, 15))
                self.sig.set_htf(b5[:-1] if b5 else [], b30[:-1] if b30 else [])
            u = "off" if self.sig.up is None else f"{self.sig.up - self.sig.down:+.2f}"
            fset = "ready" if self.sig.filt else "warming"
            flip = "FLIP-on-brick" if self.flip_mode else "manual TP"
            htf = "real 5m/30m" if self.use_time_htf else "brick-groups"
            print(f"[{self.name}] warmup (RENKO): fed {len(b1)} 1s bars | {self.sig.status()} "
                  f"| trend({htf})={u} | filter={fset} | session 18:00-16:00 ET "
                  f"(6pm-4pm) | exit={flip}")
        except Exception as e:
            print(f"[{self.name}] renko warmup FAILED ({e!r}) - starting cold, no harm.")

    async def _warmup_mss(self):
        """Seed the MSS tracker from real 5-second history so the 8-EMA + swing
        structure are warm at startup. NO orders, NO position. Only the LIVE breaks
        after mark_warm may arm an entry, so startup history never fires a stale trade."""
        try:
            bars = self._bars_to_dicts(await self._get_bars(5, 1, 1))
        except Exception as e:
            print(f"[{self.name}] MSS warmup fetch failed: {e!r}")
            bars = []
        conf = bars[:-1] if bars else []
        for b in conf:
            self.sig.on_5s(b["open"], b["high"], b["low"], b["close"])
            self._mss_last_ts = b["ts"]
        self.sig.mss_mark_warm()
        print(f"[{self.name}] warmup (MSS-5s): fed {len(conf)} 5s candles | "
              f"ctx={self.sig.mss_context()}", flush=True)

    def _feed_mss(self, price, ts):
        """Build 5-second candles from the LIVE 1-second price feed and fold each
        one into the MSS tracker as it CLOSES. Using the live feed (not polling the
        stale REST 5s endpoint) is how the bricks are built too, so the entry reacts
        in real time. Returns True when a candle just closed (a signal may be ready)."""
        bkt = int(ts // 5) * 5
        if self._mss_bkt is None:
            self._mss_bkt = bkt
            self._mss_o = self._mss_h = self._mss_l = self._mss_c = price
            return False
        if bkt == self._mss_bkt:
            if price > self._mss_h: self._mss_h = price
            if price < self._mss_l: self._mss_l = price
            self._mss_c = price
            return False
        # a new bucket started -> the previous 5s candle is CLOSED, fold it
        closed = False
        if bkt > self._mss_bkt:
            self.sig.on_5s(self._mss_o, self._mss_h, self._mss_l, self._mss_c)
            closed = True
        self._mss_bkt = bkt
        self._mss_o = self._mss_h = self._mss_l = self._mss_c = price
        return closed

    async def _tick_mss(self, price, ts):
        """MSS-on-5s tick (client 29-Jul, Abhi). Feed 5s candles, then: from FLAT,
        enter at market on a shift with a structural stop capped to the $ max-loss and
        a wider R-multiple target; in a position, run the profit-lock + hard take-profit
        (the target) with the resting protective stop as the hard backstop. No 1h gate,
        no reverse - exit to flat and wait for the next fresh shift."""
        self._feed_mss(price, ts)

        if time.time() - self._last_sig_log > 60:
            self._last_sig_log = time.time()
            mc = self.sig.mss_context() if hasattr(self.sig, "mss_context") else {}
            print(f"[{now_str()}] [{self.name}] MSS-5s: {mc} | px={price:.2f} | "
                  f"pos={self.position} | SL@{(self._stop_price if self._stop_price else 0):.2f} | "
                  f"TP=${self.take_profit_usd:.0f}", flush=True)

        if self.position == 0:
            sig = self.sig.signal(price)
            if sig and self._can_trade() \
                    and (time.time() - getattr(self, "_last_entry_ts", 0.0)) >= self.mss_cool_s:
                cap_pts = self.mss_cap_usd / (POINT_VALUE * self.qty)
                rc = getattr(self.sig, "_last_rev_ctx", None) or {}
                stop_px = rc.get("stop")
                if stop_px is not None:
                    risk = abs(float(stop_px) - price)
                    if risk < TICK:
                        risk = self.brick_size
                    if risk > cap_pts:                 # hard $ max-loss cap
                        self._sl_level = (price - cap_pts) if sig == "LONG" else (price + cap_pts)
                        eff = cap_pts
                    else:
                        self._sl_level = float(stop_px)
                        eff = risk
                else:
                    self._sl_level = (price - cap_pts) if sig == "LONG" else (price + cap_pts)
                    eff = cap_pts
                self.max_stop_pts    = round(eff + 0.5, 2)
                self.take_profit_usd = round(eff * POINT_VALUE * self.qty * self.mss_tp_r, 2)
                await self.enter(sig, price, price, market_ok=True)
                self._peak = self.entry_price or price
            return

        # In a position: same exit stack as the reversal, minus the ghost/trail.
        if self.position == 1:
            self._peak = max(self._peak, price)
        else:
            self._peak = min(self._peak, price)
        if await self._maybe_profit_lock(price):
            return
        if await self._maybe_take_profit(price):
            return
        if self._stop_order_id is None and self._stop_price is None:
            await self._sync_protective_stop()

    async def warmup_from_history(self):
        """Seed the strategy from real 5m/30m history so the trend oscillator,
        the 5m breakout colour and the 30m Ichimoku/Bollinger filter are all warm
        the moment we start - never cold. Places NO orders, touches NO position."""
        if getattr(self, "mss_mode", False):
            return await self._warmup_mss()
        if self.renko_flip:
            return await self._warmup_renkoflip()
        if self.renko_mode:
            return await self._warmup_renko()
        try:
            b5 = self._bars_to_dicts(await self._get_bars(5, 2, 15))
            b30 = self._bars_to_dicts(await self._get_bars(30, 2, 15))
            live = b5[-1]["close"] if b5 else 0.0
            self._feed_htf(b5, b30, live)
            u = "off" if self.sig.up is None else f"{self.sig.up - self.sig.down:+.2f}"
            fset = "ready" if self.sig.filt else "warming"
            print(f"[{self.name}] warmup: 5m bars={len(self._conf5)} "
                  f"30m bars={len(self._conf30)} | trend(up-down)={u} | "
                  f"30m filter={fset} | session 18:00-16:00 ET (6pm-4pm) | NO stop / manual TP")
        except Exception as e:
            print(f"[{self.name}] warmup FAILED ({e!r}) - starting cold, no harm.")

    def _roll_session(self):
        today = datetime.now(ET).date().isoformat()
        if today != self.session_day:
            print(f"[{self.name}] --- new session {today}: resetting daily loss / trade count ---")
            self.session_day = today
            self.daily_loss = 0.0
            self.trades_today = 0
            self._halted = False
            # Drop yesterday's balance anchor. sync_daily_pnl() re-anchors to the
            # broker on its next pass; leaving it would measure today's loss
            # against yesterday's opening balance.
            self.day_open_balance = None

    def _in_session(self, now=None):
        """The client's trading window: 6:00pm ET through 4:00pm ET the next day.

        Until 14 Jul there was NO clock rule in this bot at all. It simply traded
        until the exchange itself shut at 5pm ET, which is why trades appeared at
        4:05pm and 4:07pm. The client wants a hard 4pm stop, so here it is.

        NQ runs Sunday 6pm ET to Friday 5pm ET, halting 5-6pm each day. So:
          Sat            - shut all day
          Sun            - only from the 6pm reopen
          Fri            - trade the overnight, then STOP at 4pm for the weekend
          Mon-Thu        - 6pm onwards, and up to 4pm the following day
        """
        t = now or datetime.now(ET)
        wd = t.weekday()                       # Mon=0 ... Sat=5, Sun=6
        mins = t.hour * 60 + t.minute
        OPEN, STOP = 18 * 60, 16 * 60          # 6:00pm, 4:00pm
        if wd == 5:                            # Saturday
            return False
        if wd == 6:                            # Sunday - the week reopens at 6pm
            return mins >= OPEN
        if wd == 4:                            # Friday - no evening session after
            return mins < STOP
        return mins >= OPEN or mins < STOP

    def _can_trade(self):
        if not self.enabled or self._halted:
            return False
        # Hard 4pm ET stop. NOT a halt - _halted would keep it dead through the 6pm
        # reopen. This gate simply closes and re-opens with the session.
        if not self._in_session():
            return False
        # Churn breaker. At brick 8 this strategy trades ~25 times an HOUR. If it
        # ever fires 6 times in a minute, something is wrong with the bot, not with
        # the market - stop before it grinds the account down in slippage.
        now = time.time()
        self._recent_entries = [t for t in self._recent_entries if now - t < 60]
        if len(self._recent_entries) >= self.churn_max:
            print(f"[{now_str()}] [{self.name}] *** CHURN BREAKER: "
                  f"{len(self._recent_entries)} entries in 60s. That is not the "
                  f"strategy, that is a bug. Halting. ***")
            self._halted = True
            return False
        if self.daily_loss <= -abs(self.daily_loss_limit):
            if not self._halted:
                print(f"[{now_str()}] [{self.name}] DAILY LOSS LIMIT hit "
                      f"(${self.daily_loss:.0f}). Halting for the day.")
                self._halted = True
            return False
        if self.trades_today >= self.max_trades_per_day:
            if not self._halted:
                print(f"[{now_str()}] [{self.name}] max_trades_per_day "
                      f"({self.max_trades_per_day}) reached. Halting for the day.")
                self._halted = True
            return False
        return True

    # ------------------------------------------------- platform = the truth
    async def platform_pos(self):
        """Signed contract count the PLATFORM holds. None = we could not find out,
        and the caller must NOT guess.

        Talks to /Position/searchOpen RAW and parses the JSON itself, deliberately
        bypassing project_x_py's model layer. Both of the library's position calls
        are unusable against the live TopstepX API:

          * client.search_open_positions()  -> RAISES TypeError(contractDisplayName)
          * ctx.positions.get_all_positions() -> catches that same TypeError,
            logs it, and returns an EMPTY LIST

        Both break for one reason: `Position` is a fixed-field @dataclass and the
        API now sends a field it does not declare. The second is the dangerous one,
        because it does not fail - it LIES. It reports "flat" precisely when a
        position is open, and reports "flat" correctly when we are flat, so it
        looks perfect right up until it costs you money. It made this bot book
        fictional stop-outs, orphan live stop orders on a flat account, and
        re-enter on top of positions it already held.

        The raw endpoint returns clean JSON and has never once misbehaved. No
        model, no dataclass, no surprises.
        """
        try:
            r = await asyncio.wait_for(
                self.suite.client._make_request(
                    "POST", "/Position/searchOpen",
                    data={"accountId": self.suite.client.account_info.id}),
                timeout=10.0)
        except Exception as e:
            self._api_fails += 1
            print(f"[{self.name}] platform_pos failed ({self._api_fails}): {e!r}")
            return None
        if not isinstance(r, dict) or not r.get("success"):
            self._api_fails += 1
            print(f"[{self.name}] platform_pos bad response: {str(r)[:160]}")
            return None
        self._api_fails = 0
        cid = self.ctx.instrument_info.id
        for p in (r.get("positions") or []):
            if p.get("contractId") != cid:
                continue
            sz = int(p.get("size") or 0)
            if sz == 0:
                continue
            return -abs(sz) if p.get("type") == 2 else abs(sz)   # 1=LONG 2=SHORT
        return 0

    # ---------------------------------------------------------------- orders
    async def _fill_limit(self, side, size, limit_price, allow_market_fallback):
        """Marketable limit with a hard price cap. side 0=BUY 1=SELL.
        Returns fill price, or None if we could not fill inside the cap.

        On the way out it records WHY it failed in self.last_reject. A broker
        refusal and a thin book both return None, but they are not the same
        problem, and the caller must not confuse them."""
        self.last_reject = None
        try:
            resp = await asyncio.wait_for(
                self.ctx.orders.place_limit_order(
                    contract_id=self.ctx.instrument_info.id,
                    side=side, size=size, limit_price=limit_price),
                timeout=10.0)
        except Exception as e:
            self.last_reject = str(e)
            print(f"[{self.name}] limit order error: {e!r}")
            resp = None

        # A response that came back unsuccessful is ALSO a refusal - the library
        # does not always raise for one.
        if resp is not None and not getattr(resp, "success", False):
            self.last_reject = (getattr(resp, "errorMessage", None)
                                or self.last_reject or "order rejected, no reason given")

        oid = getattr(resp, "orderId", None) if resp else None
        if resp and getattr(resp, "success", False) and oid:
            deadline = time.time() + self.entry_timeout_s
            while time.time() < deadline:
                await asyncio.sleep(0.15)
                try:
                    if await self.ctx.orders.is_order_filled(oid):
                        return limit_price
                except Exception:
                    pass
            # did not fill inside the cap -> pull it
            try:
                await self.ctx.orders.cancel_order(oid)
            except Exception:
                pass

        if not allow_market_fallback:
            return None

        # EXITS ONLY: never leave a position hanging.
        print(f"[{self.name}] limit unfilled -> MARKET fallback (exit must complete)")
        try:
            r = await asyncio.wait_for(
                self.ctx.orders.place_market_order(
                    contract_id=self.ctx.instrument_info.id, side=side, size=size),
                timeout=10.0)
            if r and getattr(r, "success", False):
                # The order is placed. Getting a price is a SEPARATE call that can
                # fail on its own - and if we let that failure propagate we would
                # report "exit failed" for an exit that actually completed, and go
                # on believing we hold a position we no longer have. That desync is
                # what tried to open a naked reverse on 13 Jul. Never again: fall
                # back to the limit price rather than lose the fill.
                try:
                    px = await asyncio.wait_for(self.ctx.data.get_current_price(),
                                                timeout=5.0)
                    return float(px)
                except Exception as e:
                    print(f"[{self.name}] market order FILLED but price read failed "
                          f"({e!r}); booking at {limit_price}")
                    return float(limit_price)
        except Exception as e:
            print(f"[{self.name}] MARKET fallback failed: {e!r}")
        return None

    async def _sync_protective_stop(self):
        """RenkoFlip SL-only mode (client 21-Jul): rest ONE FIXED stop at the
        "first brick" open - the level where, if price returns to it, the first
        brick has effectively disappeared. No trailing, no take-profit: the client
        sets his own TP manually on the platform. The bot's only job here is to keep
        that single protective stop resting under the position.

        For the plain NQ Signal bot (renko_flip off) this stays a no-op - that
        strategy runs with a manual stop AND a manual TP.
        """
        if self.position == 0:
            await self._cancel_protective_stop()
            return
        if not self.renko_flip:
            return
        # Never move a stop that is already resting - it is FIXED by design.
        if self._stop_order_id is not None and self._stop_price is not None:
            return
        want = getattr(self, "_sl_level", None)
        room = self.brick_size * (1.0 + self.entry_confirm_frac)
        if want is None and self.entry_price:
            # Adopted a position after a restart and lost the original brick level:
            # fall back to the same geometry (one brick + the early-entry fraction
            # behind entry) so the position is never left unprotected.
            want = (self.entry_price - room) if self.position == 1 else (self.entry_price + room)
        if want is None:
            return
        # A stop MUST be protective: below entry for a long, above for a short. The
        # first-brick open can land on the WRONG side when price re-entered below (long)
        # / above (short) the brick structure - a late entry off stale confirmed bricks.
        # The broker then rejects it ("price outside allowed range - set below best
        # ask"). Clamp to the one-brick fallback so the stop is always on the losing
        # side of entry (and so the risk-based trail measures a sane R).
        if self.entry_price:
            if self.position == 1 and want >= self.entry_price - TICK:
                want = self.entry_price - room
            elif self.position == -1 and want <= self.entry_price + TICK:
                want = self.entry_price + room
        # Cap the stop distance so a far entry (price ran well past the brick before
        # the signal) can't create an outsized risk. Without this a "1 brick + 60%"
        # setup lost $325 (16pt) because the first-brick open was 15pt from entry.
        if self.max_stop_pts > 0 and self.entry_price:
            if self.position == 1 and want < self.entry_price - self.max_stop_pts:
                want = self.entry_price - self.max_stop_pts
            elif self.position == -1 and want > self.entry_price + self.max_stop_pts:
                want = self.entry_price + self.max_stop_pts
        # round to the tick grid so the broker accepts it
        want = round(want / TICK) * TICK
        # R = the true initial stop distance, used by the risk-based (1:1) trail.
        if self.entry_price:
            self._risk_pts = abs(self.entry_price - want)
        side = 1 if self.position == 1 else 0        # opposite side closes us
        # Throttle retries: a persistently-failing placement must NEVER be re-sent every
        # tick - that floods the book with rejected orders (the "million signals" bug,
        # client 22-Jul). At most one attempt every 2s.
        if time.time() - getattr(self, "_last_stop_attempt", 0.0) < 2.0:
            return
        self._last_stop_attempt = time.time()
        try:
            r = await asyncio.wait_for(
                self.ctx.orders.place_stop_order(
                    contract_id=self.ctx.instrument_info.id,
                    side=side, size=self.contracts_held, stop_price=want),
                timeout=8.0)
            if r and getattr(r, "success", False):
                self._stop_order_id = getattr(r, "orderId", None)
            self._stop_price = want
            print(f"[{now_str()}] [{self.name}] SL rested @ {want:.2f} "
                  f"(first-brick stop, fixed)")
        except Exception as e:
            self._die_if_account_dead(e)           # reset/closed account -> restart+re-resolve
            msg = str(e).lower()
            # "outside allowed range" / "above best bid" / "below best ask": the market
            # has ALREADY traded past where the stop would sit, so the position is beyond
            # its stop and the broker will reject the order forever. Don't retry it (spam)
            # and don't sit unprotected - CLOSE at market. That is the stop doing its job.
            if any(k in msg for k in ("outside allowed", "best bid", "best ask", "allowed range")):
                truth = await self.platform_pos()
                if truth is not None and truth != 0:
                    print(f"[{now_str()}] [{self.name}] stop {want:.2f} is past the market "
                          f"- position {truth} is beyond its stop -> FLATTEN at market")
                    try:
                        await asyncio.wait_for(self.ctx.orders.place_market_order(
                            contract_id=self.ctx.instrument_info.id,
                            side=(1 if truth > 0 else 0), size=abs(truth)), timeout=10.0)
                    except Exception as e2:
                        print(f"[{self.name}] market flatten failed: {e2!r}")
                # Mark the stop 'handled' so the per-tick retry gate stops firing;
                # reconcile() books the exit and resets us to flat.
                self._stop_price = want
                return
            print(f"[{self.name}] protective-stop place FAILED (position may be "
                  f"unprotected at {want:.2f}): {e!r}")

    async def _maybe_take_profit(self, price):
        """Hard per-trade profit lock (client 23-Jul). If the open trade's unrealized
        profit reaches take_profit_usd, cancel the resting stop and close at market so
        the gain is banked instead of given back. One market close per trade (keyed on
        entry_time); the 2s throttle stops a failed attempt from flooding the book.
        Returns True if it fired the close."""
        if self.take_profit_usd <= 0 or self.position == 0 or not self.entry_price:
            return False
        if self.entry_time and self._tp_fired_entry == self.entry_time:
            return False                                   # already closed this trade
        held = self.contracts_held or self.qty or 1
        pts = ((price - self.entry_price) if self.position == 1
               else (self.entry_price - price))
        profit = pts * POINT_VALUE * held
        if profit < self.take_profit_usd:
            return False
        if time.time() - self._last_tp_attempt < 2.0:
            return False
        self._last_tp_attempt = time.time()
        # confirm we really hold it on the platform before sending a closing order
        truth = await self.platform_pos()
        if truth is None or truth == 0:
            return False
        print(f"[{now_str()}] [{self.name}] TAKE-PROFIT: +${profit:.0f} >= "
              f"${self.take_profit_usd:.0f} ({pts:+.2f}pt @ {price:.2f}) -> cancel "
              f"stop + close at market", flush=True)
        await self._cancel_protective_stop()
        try:
            await asyncio.wait_for(self.ctx.orders.place_market_order(
                contract_id=self.ctx.instrument_info.id,
                side=(1 if truth > 0 else 0), size=abs(truth)), timeout=10.0)
        except Exception as e:
            self._die_if_account_dead(e)
            print(f"[{self.name}] take-profit market close FAILED "
                  f"(will retry next tick): {e!r}", flush=True)
            return False
        self._tp_fired_entry = self.entry_time     # book it so we don't re-send
        return True

    async def _maybe_profit_lock(self, price):
        """Dollar trailing profit-lock (client 29-Jul). Tracks the peak unrealized $
        of the open trade; once it reaches profit_lock_arm_usd the lock arms, and if
        profit later falls back to (peak - profit_lock_trail_usd) - a floor that only
        ever ratchets UP - it cancels the resting stop and closes at market so the
        winner is banked instead of round-tripping. One close per trade (keyed on
        entry_time), 2s throttle, platform-confirmed. Returns True if it fired."""
        if self.profit_lock_arm_usd <= 0 or self.position == 0 or not self.entry_price:
            return False
        if self.entry_time and self._plock_fired_entry == self.entry_time:
            return False                                   # already banked this trade
        held = self.contracts_held or self.qty or 1
        pts = ((price - self.entry_price) if self.position == 1
               else (self.entry_price - price))
        profit = pts * POINT_VALUE * held
        # new trade -> reset the peak/arm tracking
        if self._plock_entry != self.entry_time:
            self._plock_entry = self.entry_time
            self._plock_peak_usd = profit
            self._plock_armed = False
        if profit > self._plock_peak_usd:
            self._plock_peak_usd = profit
        if not self._plock_armed:
            if self._plock_peak_usd >= self.profit_lock_arm_usd:
                self._plock_armed = True
                print(f"[{now_str()}] [{self.name}] PROFIT-LOCK armed: peak +"
                      f"${self._plock_peak_usd:.0f} >= ${self.profit_lock_arm_usd:.0f} "
                      f"-> floor +${self._plock_peak_usd - self.profit_lock_trail_usd:.0f}",
                      flush=True)
            else:
                return False
        floor = self._plock_peak_usd - self.profit_lock_trail_usd
        if profit > floor:
            return False                                   # still above the trailing floor
        if time.time() - self._last_plock_attempt < 2.0:
            return False
        self._last_plock_attempt = time.time()
        truth = await self.platform_pos()
        if truth is None or truth == 0:
            return False
        print(f"[{now_str()}] [{self.name}] PROFIT-LOCK: banked +${profit:.0f} "
              f"(peak +${self._plock_peak_usd:.0f}, floor +${floor:.0f}, "
              f"{pts:+.2f}pt @ {price:.2f}) -> cancel stop + close at market", flush=True)
        await self._cancel_protective_stop()
        try:
            await asyncio.wait_for(self.ctx.orders.place_market_order(
                contract_id=self.ctx.instrument_info.id,
                side=(1 if truth > 0 else 0), size=abs(truth)), timeout=10.0)
        except Exception as e:
            self._die_if_account_dead(e)
            print(f"[{self.name}] profit-lock market close FAILED "
                  f"(will retry next tick): {e!r}", flush=True)
            return False
        self._plock_fired_entry = self.entry_time     # book it so we don't re-send
        return True

    async def _maybe_ghost_exit(self, price):
        """Ghost-flip exit (client 24-Jul): the moment the forming (unconfirmed) brick
        prints the OPPOSITE colour to our position, cancel the stop and close at market.
        Renko's 2x-flip level keeps us in through same-direction bricks (hold winners)
        and only fires on a genuine reversal (cut false breaks). The resting protective
        stop stays as a hard disconnect/gap backstop. Returns True if it fired."""
        if not self.ghost_exit or self.position == 0:
            return False
        if self.entry_time and getattr(self, "_ghost_fired_entry", 0.0) == self.entry_time:
            return False
        try:
            gc = self.sig._ghost_color(price)
        except Exception:
            return False
        want_opp = "red" if self.position == 1 else "green"
        if gc != want_opp:
            return False
        if time.time() - getattr(self, "_last_ghost_attempt", 0.0) < 2.0:
            return False
        self._last_ghost_attempt = time.time()
        truth = await self.platform_pos()
        if truth is None or truth == 0:
            return False
        print(f"[{now_str()}] [{self.name}] GHOST-FLIP exit: opposite {gc} ghost vs "
              f"pos={self.position} @ {price:.2f} -> cancel stop + close at market", flush=True)
        await self._cancel_protective_stop()
        try:
            await asyncio.wait_for(self.ctx.orders.place_market_order(
                contract_id=self.ctx.instrument_info.id,
                side=(1 if truth > 0 else 0), size=abs(truth)), timeout=10.0)
        except Exception as e:
            self._die_if_account_dead(e)
            print(f"[{self.name}] ghost-flip market close FAILED (retry next tick): {e!r}", flush=True)
            return False
        self._ghost_fired_entry = self.entry_time
        # block re-entry until a NEW confirmed brick prints (no instant re-fire)
        self._reentry_block_bricks = len(self.sig.eng.bricks)
        self._last_entry_ts = time.time()
        return True

    async def _maybe_trail_stop(self):
        """Ratchet the RESTING protective stop toward profit, never against us. It
        edits the resting order (not a soft exit), so the locked profit is held on the
        platform. Two modes, risk takes precedence:

         - trail_risk > 0 (client 22-Jul, 1:1): trail trail_risk x R behind the peak
           once the trade is up trail_risk x R, where R = the initial stop distance.
           trail_risk=1 -> the TP trails the SAME distance as the SL: breakeven at +R,
           locks 1x risk (true 1:1) at +2R, banks more as it runs. Ratchets in ~1pt
           steps to avoid hammering modify_order.
         - trail_bricks > 0: whole-brick trail (breakeven at +1 brick, +$100 at +2...).
        """
        if self.position == 0 or self._stop_order_id is None:
            return
        info = ""
        if self.trail_risk > 0 and self._risk_pts > 0:
            gap = self.trail_risk * self._risk_pts
            if self.position == 1:
                if (self._peak - self.entry_price) < gap:
                    return                                      # not up 1x risk yet
                want = self._peak - gap
                if self._stop_price is not None and want <= self._stop_price + 0.99:
                    return                                      # ratchet up, ~1pt steps
            else:
                if (self.entry_price - self._peak) < gap:
                    return
                want = self._peak + gap
                if self._stop_price is not None and want >= self._stop_price - 0.99:
                    return
            lock = ((want - self.entry_price) if self.position == 1
                    else (self.entry_price - want))
            info = f"risk {self._risk_pts:.2f}pt, lock {lock:+.2f}pt"
        elif self.trail_bricks > 0:
            bs = self.brick_size
            if self.position == 1:
                n = int((self._peak - self.entry_price) // bs)  # full profit-bricks
                if n < self.trail_bricks:
                    return
                want = self.entry_price + (n - self.trail_bricks) * bs
                if self._stop_price is not None and want <= self._stop_price + 1e-9:
                    return
            else:
                n = int((self.entry_price - self._peak) // bs)
                if n < self.trail_bricks:
                    return
                want = self.entry_price - (n - self.trail_bricks) * bs
                if self._stop_price is not None and want >= self._stop_price - 1e-9:
                    return
            info = f"locked {n - self.trail_bricks} brick"
        else:
            return
        want = round(want / TICK) * TICK
        try:
            await asyncio.wait_for(self.ctx.orders.modify_order(
                order_id=self._stop_order_id, stop_price=want), timeout=8.0)
            old = self._stop_price
            self._stop_price = want
            print(f"[{now_str()}] [{self.name}] TRAIL stop {old}->{want:.2f} "
                  f"({info}, peak {self._peak:.2f})")
        except Exception as e:
            print(f"[{self.name}] trail modify FAILED (stop stays at "
                  f"{self._stop_price}): {e!r}")

    async def _sync_protective_stop_DISABLED(self):
        if self.position == 0:
            await self._cancel_protective_stop()
            return
        lv = self.engine.next_levels()
        if lv:
            want = lv["flip"]
        else:
            # The engine has no grid yet (fresh start, or we just adopted an orphan
            # position from the platform). We are NOT allowed to sit here holding
            # size with no stop just because the bricks have not warmed up. Fall
            # back to the same geometry a flip would give us: 2 bricks against.
            if not self.entry_price:
                return
            gap = 2 * self.brick_size
            want = (self.entry_price - gap) if self.position == 1 else (self.entry_price + gap)
            print(f"[{self.name}] no brick grid yet - placing fallback stop "
                  f"{gap:.2f}pts against at {want:.2f}")
        # never let the stop move against us
        if self._stop_price is not None:
            if self.position == 1 and want <= self._stop_price:
                return
            if self.position == -1 and want >= self._stop_price:
                return
        side = 1 if self.position == 1 else 0        # opposite side closes us
        try:
            if self._stop_order_id is not None:
                await asyncio.wait_for(
                    self.ctx.orders.modify_order(order_id=self._stop_order_id,
                                                 stop_price=want), timeout=8.0)
            else:
                r = await asyncio.wait_for(
                    self.ctx.orders.place_stop_order(
                        contract_id=self.ctx.instrument_info.id,
                        side=side, size=self.contracts_held, stop_price=want),
                    timeout=8.0)
                if r and getattr(r, "success", False):
                    self._stop_order_id = getattr(r, "orderId", None)
            self._stop_price = want
        except Exception as e:
            # A failed sync means the stop is NOT where we think it is. Say so
            # loudly - a silent failure here leaves a position unprotected.
            print(f"[{self.name}] protective-stop sync FAILED (position may be "
                  f"unprotected at {want:.2f}): {e!r}")

    async def _cancel_protective_stop(self):
        """Pull the resting stop and establish WHAT HAPPENED TO IT.

        Returns one of:
          "cancelled" - the stop is gone and we are STILL holding the position.
          "filled"    - the stop EXECUTED. We are ALREADY FLAT. Do not send a close.
          False       - could not confirm. Halted.

        The old version swallowed a failed cancel and cleared the order id anyway.
        That is how you orphan a live stop: the bot forgets it, the position closes,
        and the stop is still sitting on the exchange. When price reaches it, it
        fires against a FLAT account - which does not protect anything, it OPENS a
        brand new position on the wrong side.

        That is exactly what happened at 13:54 on 13 Jul: we exited a short, the
        stale BUY-stop at 29490 fired, and the bot found itself unexpectedly long.

        And here is the 14 Jul bug, which is the SAME mistake wearing a new hat:
        `_order_gone` cannot tell a CANCELLED order from a FILLED one - both simply
        vanish from the working-order list. So when price hit the flip level, the
        stop executed, and this function shrugged and reported "gone". exit() then
        sent its closing order into an account that was already flat. A closing
        order with nothing to close is a NAKED order: it opened a brand new position
        on the opposite side. It fired 17 times on 14 Jul (69 bot entries, 83 trades
        on the platform).

        CANCELLED and FILLED are opposite outcomes. Never collapse them again.
        """
        if self._stop_order_id is None:
            return "cancelled"
        oid = self._stop_order_id
        self._stop_fill_price = None
        for attempt in range(3):
            try:
                await asyncio.wait_for(self.ctx.orders.cancel_order(oid), timeout=8.0)
            except Exception as e:
                print(f"[{self.name}] cancel stop {oid} attempt {attempt+1}: {e!r}")
            if await self._order_gone(oid):
                # Gone. But gone HOW? Ask - do not infer.
                if await self._stop_did_fill(oid):
                    self._stop_fill_price = self._stop_price
                    px = self._stop_price if self._stop_price else 0.0
                    print(f"[{now_str()}] [{self.name}] stop {oid} had ALREADY FILLED "
                          f"@ {px:.2f} - we are flat. NOT stacking a close order on "
                          f"top of it.")
                    self._stop_order_id = None
                    self._stop_price = None
                    return "filled"
                self._stop_order_id = None
                self._stop_price = None
                return "cancelled"
            await asyncio.sleep(0.4)
        print(f"[{now_str()}] [{self.name}] *** COULD NOT CANCEL STOP {oid}. It is "
              f"still live on the platform. Halting rather than risk it firing "
              f"against a flat account. ***")
        self._halted = True
        return False

    async def _stop_did_fill(self, oid):
        """Did `oid` leave the book by EXECUTING, rather than by being cancelled?

        If the platform will not answer, fall back to asking what we actually hold.
        A position that is genuinely flat can only have got that way one road: the
        stop filled.
        """
        try:
            return bool(await asyncio.wait_for(
                self.ctx.orders.is_order_filled(oid), timeout=8.0))
        except Exception as e:
            print(f"[{self.name}] could not read fill status of stop {oid}: {e!r}")
        # The position feed lags a fill by a second or two, so this reading is only
        # safe because we are long past the entry by the time we are exiting.
        try:
            return (await self.platform_pos()) == 0
        except Exception:
            return False                       # cannot tell -> assume still in

    async def _order_gone(self, oid):
        """True once `oid` no longer appears in the platform's working orders.

        NOTE: "gone" means cancelled OR filled. It does NOT tell you which. Anyone
        who needs to know must call _stop_did_fill().
        """
        try:
            oo = await asyncio.wait_for(self.ctx.orders.search_open_orders(),
                                        timeout=8.0)
        except Exception:
            return False                       # cannot verify -> assume still there
        return not any(getattr(o, "id", None) == oid for o in (oo or []))

    async def sweep_orphan_orders(self):
        """INVARIANT: flat means NO working orders. Anything resting on a flat
        account is a landmine - it can only ever open a position, never close one.

        Cheap to enforce, and it catches every future variant of this bug, not just
        the one I already know about.
        """
        try:
            oo = await asyncio.wait_for(self.ctx.orders.search_open_orders(),
                                        timeout=8.0)
        except Exception as e:
            # NEVER swallow this. A silent failure here means the sweep quietly
            # stops running and the invariant it exists to protect is gone. Three
            # separate bugs today hid behind a bare `except: return`.
            print(f"[{self.name}] sweep could not read open orders: {e!r}")
            return
        # Deliberately NOT filtered by contractId: this account trades exactly one
        # contract, and ANY working order on a flat account is wrong regardless of
        # what it claims to be attached to.
        # EXCEPTION: a passive entry limit we placed on purpose (limit_entry mode) is
        # SUPPOSED to rest while flat - it is how we get in. Never sweep our own.
        keep = self._pending_entry["order_id"] if self._pending_entry else None
        for o in (oo or []):
            oid = getattr(o, "id", None)
            if keep is not None and oid == keep:
                continue
            print(f"[{now_str()}] [{self.name}] ORPHAN order {oid} resting while "
                  f"FLAT (type={getattr(o, 'type', '?')} "
                  f"stop={getattr(o, 'stopPrice', None)}) - cancelling")
            try:
                await asyncio.wait_for(self.ctx.orders.cancel_order(oid), timeout=8.0)
                if await self._order_gone(oid):
                    print(f"[{self.name}] orphan {oid} confirmed gone")
                else:
                    print(f"[{self.name}] *** orphan {oid} STILL LIVE after cancel ***")
            except Exception as e:
                print(f"[{self.name}] could not cancel orphan {oid}: {e!r}")

    async def enter(self, direction, level, price, market_ok=False):
        """direction 'LONG'|'SHORT'. `level` = the exact brick price that triggered.
        market_ok: the follower mirrors with a market backstop so it never misses
        the master's trade; the master keeps its strict no-chase limit (False)."""
        if not self._can_trade():
            return
        # Reverse-signals mode: mirror the whole trade. Flip here, once, so side,
        # marketable limit, self.position and the protective stop all derive from
        # the actual (opposite) direction with no other change. `level` is the
        # live trigger price, so a flipped fill is still marketable.
        if self.reverse:
            direction = "SHORT" if direction == "LONG" else "LONG"
        # Never open on top of something. If the platform is not flat, or will not
        # tell us, we do not send an entry - that is how you end up double-sized.
        truth = await self.platform_pos()
        if truth is None:
            print(f"[{now_str()}] [{self.name}] SKIP {direction} - platform not "
                  f"answering, refusing to trade blind")
            return
        if truth != 0:
            print(f"[{now_str()}] [{self.name}] SKIP {direction} - platform still "
                  f"holds {truth}, not stacking on top of it")
            return
        cap = self.max_entry_slip_ticks * TICK
        side = 0 if direction == "LONG" else 1
        # marketable limit: we will pay up to `cap` beyond the brick level, no more
        limit = level + cap if direction == "LONG" else level - cap
        fill = await self._fill_limit(side, self.qty, limit, allow_market_fallback=market_ok)
        if fill is None:
            # Two very different failures land here. Do not conflate them: on
            # 14 Jul the account had been CLOSED and every order was refused, but
            # the bot reported it as a routine slippage skip 15 times in a row.
            # It looked quiet. It was broken.
            if self.last_reject:
                why = self.last_reject
                if any(k in why.lower() for k in FATAL_REJECTS):
                    print(f"[{now_str()}] [{self.name}] *** BROKER REFUSED THE ORDER: "
                          f"{why} *** This is an account problem, not a market one - "
                          f"no order will ever fill. Halting so it stops pretending "
                          f"to trade. Fix the account, then restart me.", flush=True)
                    self._halted = True
                else:
                    print(f"[{now_str()}] [{self.name}] REJECTED {direction} @ "
                          f"{level:.2f} - broker said: {why}", flush=True)
                return
            self.skipped_slippage += 1
            print(f"[{now_str()}] [{self.name}] SKIP {direction} @ {level:.2f} - "
                  f"book would not fill inside {cap:.2f}pt cap "
                  f"(skipped {self.skipped_slippage} today)")
            return
        self.position = 1 if direction == "LONG" else -1
        self.contracts_held = self.qty
        self.intended_entry = level
        self.entry_price = fill
        self.entry_time = time.time()
        # fresh excursion window for this trade
        self._mfe_pts = 0.0
        self._mae_pts = 0.0
        self._mfe_price = fill
        self._mae_price = fill
        self.trades_today += 1
        self._recent_entries.append(self.entry_time)
        self._flat_polls = 0
        self._last_entry_ts = time.time()   # 1-contract lock: throttle re-entry
        slip = (fill - level) if direction == "LONG" else (level - fill)

        # Stamp the order flow as it stood the instant we entered. We do NOT act on
        # it. In a week we replay these against the actual results and ask, on his
        # own trades: "would refusing the ones that fought the flow have helped?"
        # That is a question worth answering with data, not with a theory.
        try:
            self.flow_at_entry = self.flow.snapshot(time.time())
        except Exception:
            self.flow_at_entry = {}

        await self._sync_protective_stop()
        imb = self.flow_at_entry.get("imbalance_30s")
        fs = f" | flow30s {imb:+.2f}" if imb is not None else ""
        print(f"[{now_str()}] [{self.name}] >>> {direction} x{self.qty} @ {fill:.2f} "
              f"(brick {level:.2f}, slip {slip:+.2f}pt) | flip-stop @ "
              f"{self._stop_price if self._stop_price else 0:.2f}{fs} | "
              f"trade {self.trades_today}/{self.max_trades_per_day}")
        # Master: tell the follower to take the same side right now.
        self._sync_emit("ENTRY", "LONG" if self.position == 1 else "SHORT")

    async def _flip(self, new_direction, price):
        """Renko flip: REVERSE the position in a single order (close what we hold +
        open qty the other way, size = held + qty). Books the closed leg as a FLIP
        trade with its MFE/MAE, then holds the new side. This is the always-in-the-
        market flip: it cuts the loser and rides the new direction."""
        if self.position == 0:
            return
        old_dir = "LONG" if self.position == 1 else "SHORT"
        # The platform is the truth about what we actually hold before reversing.
        truth = await self.platform_pos()
        if truth is None:
            print(f"[{now_str()}] [{self.name}] FLIP {old_dir}->{new_direction} "
                  f"skipped - platform not answering")
            return
        if truth == 0:
            # already flat (client closed it) - just drop our stale position
            print(f"[{now_str()}] [{self.name}] wanted to flip but platform is FLAT "
                  f"- clearing stale {old_dir}")
            self.position = 0
            self.contracts_held = 0
            return
        held = abs(truth)
        side = 0 if new_direction == "LONG" else 1     # BUY to become long, SELL to become short
        size = held + self.qty                          # close held + open qty, one order
        cap = self.max_entry_slip_ticks * TICK
        limit = price + cap if new_direction == "LONG" else price - cap
        fill = await self._fill_limit(side, size, limit, allow_market_fallback=True)
        if fill is None:
            print(f"[{now_str()}] [{self.name}] FLIP {old_dir}->{new_direction} "
                  f"could not fill - will retry next tick")
            return
        # book the P&L of the leg we just closed (the OLD position)
        pnl_pts = (fill - self.entry_price) if self.position == 1 else (self.entry_price - fill)
        pnl = pnl_pts * POINT_VALUE * held
        self.live_pnl += pnl
        self.daily_loss += pnl
        rec = {
            "ts": datetime.now(ET).isoformat(), "account": self.name,
            "brick_size": self.brick_size,
            "direction": old_dir, "intended_entry": self.intended_entry,
            "entry": self.entry_price, "exit": fill, "reason": "FLIP",
            "pnl_pts": round(pnl_pts, 2), "pnl": round(pnl, 2),
            "time_in_trade_s": round(time.time() - self.entry_time, 1),
            "daily_loss": round(self.daily_loss, 2)}
        rec.update({f"flow_{k}": v for k, v in (self.flow_at_entry or {}).items()})
        rec.update(self._excursion_fields())
        rec.update(self._level_fields())
        self._rl_finalize(pnl)
        try:
            with open(TRADE_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        print(f"[{now_str()}] [{self.name}] <<>> FLIP {old_dir}->{new_direction} @ "
              f"{fill:.2f} | closed {old_dir} {pnl_pts:+.2f}pt = ${pnl:+.0f} | "
              f"day ${self.daily_loss:.0f} | total ${self.live_pnl:+.0f}")
        # now holding qty in the NEW direction
        self.position = 1 if new_direction == "LONG" else -1
        self.contracts_held = self.qty
        self.intended_entry = price
        self.entry_price = fill
        self.entry_time = time.time()
        self._mfe_pts = 0.0
        self._mae_pts = 0.0
        self._mfe_price = fill
        self._mae_price = fill
        self.trades_today += 1
        self._recent_entries.append(self.entry_time)
        try:
            self.flow_at_entry = self.flow.snapshot(time.time())
        except Exception:
            self.flow_at_entry = {}

    async def exit(self, price, reason):
        if self.position == 0:
            return None
        # pull the resting stop FIRST, or it can fire against the flat position we
        # are about to create and silently open a new one
        outcome = await self._cancel_protective_stop()
        if outcome is False:
            # The stop is still live and we could not kill it. _cancel_protective_stop
            # has already halted us. Closing now would leave that stop resting on a
            # flat account, where it can only ever OPEN a position.
            return None

        if outcome == "filled":
            # Our own stop beat us to the exit. The position is ALREADY closed, so
            # there is nothing left to sell - and a close order with nothing to close
            # is a naked order that OPENS the opposite position. Book the stop-out and
            # send nothing. This is the 14 Jul bug; see _cancel_protective_stop.
            fill = float(self._stop_fill_price or price)
            reason = "PROTECTIVE-STOP"
        else:
            side = 1 if self.position == 1 else 0     # opposite to close
            cap = self.max_entry_slip_ticks * TICK
            limit = price - cap if self.position == 1 else price + cap
            fill = await self._fill_limit(side, self.contracts_held, limit,
                                          allow_market_fallback=True)
            if fill is None:
                # Do NOT assume we are still in. The order may well have filled and
                # only the confirmation read failed. Ask the platform.
                truth = await self.platform_pos()
                if truth == 0:
                    print(f"[{self.name}] exit reported failure but platform is FLAT - "
                          f"the fill went through. Booking at {price:.2f}.")
                    fill = float(price)
                else:
                    print(f"[{self.name}] *** EXIT FAILED and platform still shows "
                          f"{truth} - position IS open, will retry next tick ***")
                    return None
        pnl_pts = (fill - self.entry_price) if self.position == 1 else (self.entry_price - fill)
        pnl = pnl_pts * POINT_VALUE * self.contracts_held
        self.live_pnl += pnl
        # NET for the day, not just the losses. Summing only the losers made the
        # daily stop fire while the account was actually up.
        self.daily_loss += pnl
        rec = {
            "ts": datetime.now(ET).isoformat(),
            "account": self.name, "brick_size": self.brick_size,
            "direction": "LONG" if self.position == 1 else "SHORT",
            "intended_entry": self.intended_entry, "entry": self.entry_price,
            "exit": fill, "reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl": round(pnl, 2),
            "time_in_trade_s": round(time.time() - self.entry_time, 1),
            "daily_loss": round(self.daily_loss, 2),
        }
        rec.update({f"flow_{k}": v for k, v in (self.flow_at_entry or {}).items()})
        rec.update(self._excursion_fields())
        rec.update(self._level_fields())
        self._rl_finalize(pnl)
        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{now_str()}] [{self.name}] <<< EXIT {rec['direction']} @ {fill:.2f} "
              f"| {reason} | {pnl_pts:+.2f}pt = ${pnl:+.0f} | day ${self.daily_loss:.0f} "
              f"| total ${self.live_pnl:+.0f}")
        self.position = 0
        self.contracts_held = 0
        # Master: tell the follower to close too, so the accounts stay matched.
        self._sync_emit("EXIT")
        return pnl

    async def _actual_close_pnl(self):
        """Read the REAL closing fill from TopStep (/Trade/search) instead of
        assuming the exit landed on the stop level. A triggered stop is a market
        order, so the true fill - and the realized P&L - carries slippage. The
        closing fill is the one that carries a non-null profitAndLoss (the opening
        fill's is null); we take the most recent such fill for this contract.

        Returns {"price","pnl","fees"} (pnl = platform realized P&L, gross of the
        separate commission+fees) or None if the API can't be reached.
        """
        try:
            acct = self.suite.client.account_info.id
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=30)
            r = await asyncio.wait_for(self.suite.client._make_request(
                "POST", "/Trade/search",
                data={"accountId": acct,
                      "startTimestamp": start.isoformat(),
                      "endTimestamp": end.isoformat()}), timeout=8.0)
            trades = (r or {}).get("trades", []) if isinstance(r, dict) else []
            def _ts(t):
                try:
                    return datetime.fromisoformat(t.get("creationTimestamp")).timestamp()
                except Exception:
                    return 0.0
            # Only THIS trade's close: a realized (non-null P&L) fill on our contract
            # that happened at/after we entered - so a lagged feed can't hand us the
            # previous trade's number.
            closers = [t for t in trades
                       if t.get("profitAndLoss") is not None
                       and t.get("contractId") == self.ctx.instrument_info.id
                       and _ts(t) >= self.entry_time - 3.0]
            if not closers:
                return None
            closers.sort(key=_ts)
            t = closers[-1]
            return {"price": float(t.get("price")),
                    "pnl": float(t.get("profitAndLoss")),
                    "fees": float(t.get("fees") or 0.0) + float(t.get("commissions") or 0.0)}
        except Exception as e:
            print(f"[{self.name}] trade-history read failed ({e!r}) - using estimate")
            return None

    async def reconcile(self):
        """Believe the platform, always - in BOTH directions.

        The old version only ran when we THOUGHT we were in a position, and it
        swallowed its exception and returned. So when the TopstepX API browned out
        on 13 Jul (every /Position call timing out for ~20 minutes) it simply gave
        up, and the bot went on believing it was long after the platform had
        already closed it. Had a brick flipped, the bot would have "exited" a long
        it did not hold - which is a naked SELL, i.e. it would have OPENED a short.
        """
        truth = await self.platform_pos()
        if truth is None:
            return                                  # API down: do not guess
        # HARD 1-CONTRACT LOCK. On a fast timeframe a re-entry can slip in before
        # the position feed catches up and the size stacks (seen: 11-13 lots). If
        # the platform ever holds more than one contract, flatten the whole thing
        # to zero and wait for a fresh brick - never adopt-and-flip a stack.
        if abs(truth) > self.qty:
            print(f"[{now_str()}] [{self.name}] OVERSIZED {truth} contracts -> "
                  f"FLATTEN to 0 (1-contract lock)")
            try:
                await asyncio.wait_for(self.ctx.orders.place_market_order(
                    contract_id=self.ctx.instrument_info.id,
                    side=(1 if truth > 0 else 0), size=abs(truth)), timeout=10.0)
            except Exception as e:
                print(f"[{self.name}] flatten failed: {e!r}")
            await self._cancel_protective_stop()
            self._pending_entry = None
            self.position = 0
            self.contracts_held = 0
            self._stop_order_id = None
            self._stop_price = None
            self._last_entry_ts = time.time()
            _eng = getattr(self.sig, "eng", None)
            if _eng is not None and hasattr(_eng, "_total_bricks"):
                self._reentry_block_bricks = _eng._total_bricks
            return
        mine = self.position * self.contracts_held

        if truth == mine:
            # Flat and in agreement? Then nothing should be resting on the account.
            # Enforce it every cycle - a stop that outlives its position is the one
            # order that can open a trade nobody asked for.
            if truth == 0 and self._stop_order_id is None:
                self._sweep_tick = getattr(self, "_sweep_tick", 0) + 1
                if self._sweep_tick % 5 == 0:       # ~every 10s, not every 2s
                    await self.sweep_orphan_orders()
            return                                  # agreed

        if truth != 0 and self.position == 0:
            # If this is our own passive entry limit filling (limit_entry mode), adopt
            # it with the EXACT limit price + first-brick SL rather than the generic
            # market-price adoption below.
            if self._pending_entry is not None:
                await self._adopt_limit_fill(truth)
                return
            # platform holds something we know nothing about - adopt it so that at
            # minimum it is protected and can be closed
            self.position = 1 if truth > 0 else -1
            self.contracts_held = abs(truth)
            if not self.entry_price:
                self.entry_price = float(await self.ctx.data.get_current_price())
                self.entry_time = time.time()
                self._mfe_pts = 0.0
                self._mae_pts = 0.0
                self._mfe_price = self.entry_price
                self._mae_price = self.entry_price
            # Any stop we are still holding a reference to belonged to a PREVIOUS
            # trade. Keeping it would trip the never-move-backwards guard and this
            # position would end up with no stop at all.
            self._stop_order_id = None
            self._stop_price = None
            print(f"[{now_str()}] [{self.name}] platform holds {truth} that we did "
                  f"not know about - ADOPTING so it gets a stop")
            await self._sync_protective_stop()
            return

        if truth != 0 and self.position != 0:
            print(f"[{now_str()}] [{self.name}] size mismatch (we {mine}, platform "
                  f"{truth}) - adopting platform")
            self.position = 1 if truth > 0 else -1
            self.contracts_held = abs(truth)
            return

        # ---- platform says FLAT while we think we hold something ----------------
        #
        # DO NOT trust a single flat read here. The position feed lags a fill by a
        # second or two, so polling right after an entry returns "flat" for a
        # position that absolutely exists. Acting on that read booked four
        # fictional stop-outs in 60 seconds on 13 Jul, and - far worse - it made the
        # bot forget the REAL stop order it had just placed, leaving live stop
        # orders resting on a flat account. Each orphan is a landmine: price trades
        # through it and it opens a naked position.
        #
        # So: wait out a grace period, demand two consecutive flat reads, and then
        # ASK whether the stop actually filled rather than inferring it.
        if time.time() - self.entry_time < self.reconcile_grace_s:
            return
        self._flat_polls += 1
        if self._flat_polls < 2:
            return
        self._flat_polls = 0

        filled = False
        if self._stop_order_id is not None:
            try:
                filled = await asyncio.wait_for(
                    self.ctx.orders.is_order_filled(self._stop_order_id), timeout=8.0)
            except Exception as e:
                print(f"[{self.name}] could not read stop status: {e!r}")

        if filled and self._stop_price:
            px, reason = self._stop_price, "PROTECTIVE-STOP"
        else:
            # Something closed us that was not our stop. Book at the live price and
            # say so - do not quietly pretend the stop did it.
            try:
                px = float(await asyncio.wait_for(self.ctx.data.get_current_price(),
                                                  timeout=5.0))
            except Exception:
                px = self._stop_price or self.entry_price
            reason = "CLOSED-EXTERNALLY"
            print(f"[{now_str()}] [{self.name}] platform went flat but our stop did "
                  f"NOT fill - position closed by something else")

        # Whatever happened, the resting stop must not survive us going flat.
        await self._cancel_protective_stop()
        # Book the REAL result from TopStep, not the assumed stop level. A stop fills
        # at market once triggered, so a "locked $100" can actually land at $95/$105
        # (slippage). Pull the true closing fill + realized P&L; fall back to the
        # geometric estimate only if the API is unreachable.
        fees = 0.0
        actual = await self._actual_close_pnl()
        if actual is not None:
            px = actual["price"]                          # true exit fill price
            pnl = actual["pnl"]                           # true realized P&L (platform)
            fees = actual["fees"]
            pnl_pts = (pnl / (POINT_VALUE * self.contracts_held)
                       if self.contracts_held else 0.0)
            reason = reason + "/API"
        else:
            pnl_pts = (px - self.entry_price) if self.position == 1 else (self.entry_price - px)
            pnl = pnl_pts * POINT_VALUE * self.contracts_held
        self.live_pnl += pnl
        self.daily_loss += pnl              # net for the day, winners included
        print(f"[{now_str()}] [{self.name}] <<< {reason} @ {px:.2f} "
              f"| {pnl_pts:+.2f}pt = ${pnl:+.0f} (fees ${fees:.2f}) | day ${self.daily_loss:.0f}")
        rec = {
            "ts": datetime.now(ET).isoformat(), "account": self.name,
            "brick_size": self.brick_size,
            "direction": "LONG" if self.position == 1 else "SHORT",
            "intended_entry": self.intended_entry, "entry": self.entry_price,
            "exit": px, "reason": reason,
            "pnl_pts": round(pnl_pts, 2), "pnl": round(pnl, 2), "fees": round(fees, 2),
            "time_in_trade_s": round(time.time() - self.entry_time, 1),
            "daily_loss": round(self.daily_loss, 2)}
        rec.update({f"flow_{k}": v for k, v in (self.flow_at_entry or {}).items()})
        rec.update(self._excursion_fields())
        rec.update(self._level_fields())
        self._rl_finalize(pnl)
        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.position = 0
        self.contracts_held = 0
        self._stop_order_id = None
        self._stop_price = None
        # After ANY external/stop close, wait for a genuinely NEW brick before
        # taking the next setup. Without this the persistent signal re-fires the
        # instant we go flat, and in the fill-settlement window that rapid
        # re-entry can stack a second contract. Same discipline as TRAIL-PROFIT.
        _eng = getattr(self.sig, 'eng', None)
        if _eng is not None and hasattr(_eng, '_total_bricks'):
            self._reentry_block_bricks = _eng._total_bricks
        # Master: the client's manual close on the lead account lands here
        # (CLOSED-EXTERNALLY) - mirror it to the follower so it closes too.
        self._sync_emit("EXIT")

    # ------------------------------------------------------------------ tick
    def _update_ema(self, bricks):
        """EMA over BRICK closes. A brick closing across it arms a direction.

        Note we update the EMA on EVERY brick (indicators are free), but we only
        ever ACT once per 1-second bar, on the last brick. That distinction is not
        cosmetic: when one fast bar prints three bricks we cannot trade the first
        one's price - by the time we know it formed, price is already at the third.
        Backtesting as if we could turned a losing strategy into a fake +$107k one.
        """
        k = 2.0 / (self.ema_len + 1.0)
        for b in bricks:
            c = b["close"]
            self._ema = c if self._ema is None else self._ema + k * (c - self._ema)
            above = c > self._ema
            if self._prev_above is None:
                self._prev_above = above
                continue
            if above and not self._prev_above:
                self._armed = "long"
            elif (not above) and self._prev_above:
                self._armed = "short"
            self._prev_above = above

    def _rl_gate(self, sig, price):
        """Take/skip decision for a fade the strict level filter already approved.
        Returns True to enter. Always True when RL is off, so RL can only ever REDUCE
        entries, never invent one. On a skip it logs the passed-on setup so the record
        shows what RL avoided. On a take it stashes the state to reward on exit."""
        # Reversal mode: the two-lower-highs / two-higher-lows structure IS the gate
        # and the 1:1 stop/target is fixed by the swing point - no RL take/skip on top.
        if getattr(self, "reversal_mode", False):
            self._entry_level_ctx = getattr(self.sig, "_last_rev_ctx", None)
            self._rl_pending = None
            return True
        # Wave-3 mode: no support/resistance touches to bucket on - RL learns purely
        # by direction + session part on top of the wave-3 structural gate.
        if getattr(self, "wave_mode", False):
            self._entry_level_ctx = getattr(self.sig, "_last_wave_ctx", None)
            if not self.levels_rl:
                self._rl_pending = None
                return True
            st = state_key(sig, 1, False, datetime.now(ET).hour)
            action, reason = self._rl.decide(st)
            if action == "skip":
                self._rl_pending = None
                print(f"[{now_str()}] [{self.name}] RL-SKIP {sig} @ {price:.2f} ({reason}) | wave3")
                try:
                    rec = {"ts": datetime.now(ET).isoformat(), "account": self.name,
                           "event": "rl_skip", "direction": sig, "price": round(price, 2),
                           "rl_state": st, "rl_reason": reason}
                    if self._entry_level_ctx:
                        rec.update({f"lvl_{k}": v for k, v in self._entry_level_ctx.items()})
                    with open(TRADE_LOG, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
                return False
            self._rl_pending = st
            return True
        ctx = getattr(self.sig, "_last_level_ctx", None) if getattr(self, "levels_filter", False) else None
        self._entry_level_ctx = ctx
        if not self.levels_rl:
            self._rl_pending = None
            return True
        if sig == "LONG":
            touches = (ctx or {}).get("S_touches", 0)
            flipped = (ctx or {}).get("S_flipped", False)
        else:
            touches = (ctx or {}).get("R_touches", 0)
            flipped = (ctx or {}).get("R_flipped", False)
        st = state_key(sig, touches, flipped, datetime.now(ET).hour)
        action, reason = self._rl.decide(st)
        if action == "skip":
            self._rl_pending = None
            print(f"[{now_str()}] [{self.name}] RL-SKIP {sig} @ {price:.2f} ({reason}) "
                  f"| {'support' if sig=='LONG' else 'resistance'} touches={touches} flip={flipped}")
            try:
                rec = {"ts": datetime.now(ET).isoformat(), "account": self.name,
                       "event": "rl_skip", "direction": sig, "price": round(price, 2),
                       "rl_state": st, "rl_reason": reason}
                if ctx:
                    rec.update({f"lvl_{k}": v for k, v in ctx.items()})
                with open(TRADE_LOG, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass
            return False
        self._rl_pending = st
        return True

    def _level_fields(self):
        """Level context captured at entry, for the trade record."""
        ctx = getattr(self, "_entry_level_ctx", None)
        return {f"lvl_{k}": v for k, v in ctx.items()} if ctx else {}

    def _rl_finalize(self, pnl):
        """Fold the just-closed trade's realized $ P&L into RL, exactly once."""
        st = getattr(self, "_rl_pending", None)
        self._rl_pending = None
        if st and self.levels_rl:
            try:
                self._rl.reward(st, float(pnl))
                print(f"[{now_str()}] [{self.name}] RL-LEARN {st} <- ${float(pnl):+.0f}")
            except Exception:
                pass
        self._entry_level_ctx = None

    def _track_excursion(self, price):
        """Tick-by-tick MFE/MAE for the open trade. Records only; gates nothing."""
        if self.position == 0 or not self.entry_price:
            return
        try:
            p = float(price)
        except (TypeError, ValueError):
            return
        fav = (p - self.entry_price) if self.position == 1 else (self.entry_price - p)
        adv = -fav
        if fav > self._mfe_pts:
            self._mfe_pts = fav
            self._mfe_price = p
        if adv > self._mae_pts:
            self._mae_pts = adv
            self._mae_price = p

    def _excursion_fields(self):
        """Excursion summary for the trade record (points and dollars)."""
        mult = POINT_VALUE * (self.contracts_held or self.qty)
        return {
            "mfe_pts": round(self._mfe_pts, 2),
            "mae_pts": round(self._mae_pts, 2),
            "mfe_usd": round(self._mfe_pts * mult, 2),
            "mae_usd": round(self._mae_pts * mult, 2),
            "mfe_price": round(self._mfe_price, 2),
            "mae_price": round(self._mae_price, 2),
        }

    # ---------------------------------------------------- account sync (bus)
    def _sync_init(self):
        """Point our seq cursor at the CURRENT end of the bus so a restart never
        replays the day's earlier events. A follower replaying a stale ENTRY would
        open a position nobody signalled right now - so we only ever act on events
        written AFTER we came up."""
        if not self.sync_role or not self.sync_bus:
            return
        mx = 0
        try:
            if os.path.exists(self.sync_bus):
                with open(self.sync_bus) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            mx = max(mx, int(json.loads(line).get("seq", 0)))
                        except Exception:
                            pass
        except Exception as e:
            print(f"[{self.name}] sync init read failed: {e!r}")
        self._sync_seq = mx
        print(f"[{self.name}] sync {self.sync_role} - bus at seq {mx}")

    def _sync_emit(self, event, side=None):
        """MASTER only: publish an ENTRY/EXIT to the shared bus for the follower."""
        if self.sync_role != "master" or not self.sync_bus:
            return
        try:
            self._sync_seq += 1
            rec = {"seq": self._sync_seq, "ts": time.time(), "event": event,
                   "side": side, "from": self.name}
            os.makedirs(os.path.dirname(self.sync_bus), exist_ok=True)
            with open(self.sync_bus, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{now_str()}] [{self.name}] sync -> {event} {side or ''} "
                  f"(seq {self._sync_seq})")
        except Exception as e:
            print(f"[{self.name}] sync emit failed: {e!r}")

    async def _sync_consume(self, price, ts):
        """FOLLOWER only: mirror new master events. ENTRY -> take the same side
        (market-backed so we never miss it); EXIT -> flatten."""
        if self.sync_role != "follower" or not self.sync_bus:
            return
        if not os.path.exists(self.sync_bus):
            return
        try:
            with open(self.sync_bus) as f:
                lines = f.readlines()
        except Exception:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            seq = int(ev.get("seq", 0))
            if seq <= self._sync_seq:
                continue
            self._sync_seq = seq
            event = ev.get("event")
            if event == "ENTRY" and self.position == 0 and self._can_trade():
                direction = "LONG" if ev.get("side") == "LONG" else "SHORT"
                print(f"[{now_str()}] [{self.name}] sync <- ENTRY {direction} "
                      f"(seq {seq}) - mirroring")
                await self.enter(direction, price, price, market_ok=True)
            elif event == "EXIT" and self.position != 0:
                print(f"[{now_str()}] [{self.name}] sync <- EXIT (seq {seq}) - closing")
                await self.exit(price, "MIRROR-EXIT")

    def _log_signal_state(self, price, ts_ny):
        """Explain, once a minute, exactly which of the strategy's conditions are
        (not) met right now - so 'why hasn't it traded?' has a factual answer
        instead of a shrug. Pure read of the live signal state; changes nothing."""
        s = self.sig
        insess = nq_in_session(ts_ny)
        if s.up is None or s.down is None or s.filt is None:
            print(f"[{now_str()}] [{self.name}] why-no-signal: warming up "
                  f"(5m/30m not ready yet) | in_session={insess}")
            return
        f = s.filt
        bull = s.up > s.down
        # distances that the 30m filter blocks a BUY / SELL on (within 5pt)
        def near(level):
            return abs(price - level)
        blocks = []
        for nm, lv in (("tenkan", f["tenkan"]), ("kijun", f["kijun"]), ("bbmid", f["bb_mid"])):
            if near(lv) < 5.0:
                blocks.append(f"{nm} {price-lv:+.1f}pt")
        if f["bb_upper"] - price < 5.0:
            blocks.append(f"bbUpper {price-f['bb_upper']:+.1f}pt")
        if price - f["bb_lower"] < 5.0:
            blocks.append(f"bbLower {price-f['bb_lower']:+.1f}pt")
        blocktxt = ("BLOCKED near " + ", ".join(blocks)) if blocks else "clear"
        print(f"[{now_str()}] [{self.name}] why-no-signal: in_session={insess} | "
              f"trend={'BULL' if bull else 'BEAR'}({s.up - s.down:+.2f}) | "
              f"30m={'GREEN' if f['is_green'] else ('RED' if f['is_red'] else 'doji')} | "
              f"pending_buy={s.pending_buy}(brk {s.break_high}) "
              f"pending_sell={s.pending_sell}(brk {s.break_low}) | "
              f"px={price:.2f} | 30m-filter={blocktxt}")

    async def on_price(self, price, ts):
        self._roll_session()
        self._live_price = price
        # record how far the OPEN trade runs for/against us (MFE/MAE) every tick.
        # With no stop and manual TP, this excursion record IS the deliverable:
        # the average TP/SL per entry that the RL layer will learn from later.
        self._track_excursion(price)

        # Follower: run NO strategy of its own - just mirror the master's bus so
        # both accounts take the same trade at the same instant.
        if self.sync_role == "follower":
            await self._sync_consume(price, ts)
            return

        # MSS-on-5s (client 29-Jul, Abhi): its own tick handler off 5-second candles.
        if getattr(self, "mss_mode", False):
            await self._tick_mss(price, ts)
            return

        # Renko-flip (5-min 2-point Renko + trailing stop) has its own tick handler.
        if self.renko_flip:
            await self._on_price_renkoflip(price, ts)
            return

        # Update the strategy inputs. Renko: feed the live price to the brick
        # stream (buckets to 1s closes internally); if use_time_htf, also refresh
        # the trend/filter off the REAL 5m/30m bars. Time-candle: refetch 5m/30m.
        if self.renko_mode:
            self.sig.on_tick(price, ts, price)
            if self.use_time_htf:
                await self._refresh_htf_renko(price)
        else:
            await self._refresh_htf(price)

        # Once a minute, say WHY no trade fired (which condition is missing).
        if time.time() - self._last_sig_log > 60:
            self._last_sig_log = time.time()
            self._log_signal_state(price, datetime.fromtimestamp(ts, tz=ET))

        ts_ny = datetime.fromtimestamp(ts, tz=ET)

        # Renko FLIP mode: while in a position, if the 1-brick unconfirmed colour
        # flips against us, REVERSE to that side (always-in-market flip - cuts the
        # loss, rides the trend). This is Abhi's exit; it replaces manual TP.
        if self.renko_mode and self.flip_mode and self.position != 0:
            flip = self.sig.flip_signal(self.position)
            if flip and self._can_trade():
                await self._flip(flip, price)
            return

        # Otherwise: entries only, one at a time. (Time-candle Abhi/Rutvi close
        # manually - reconcile() books the exit with MFE/MAE when flat.)
        decision = self.sig.decide(price, ts_ny)
        if decision and self.position == 0 and self._can_trade():
            await self.enter("LONG" if decision == "BUY" else "SHORT", price, price)

    # ----------------------------------------------------------------- state
    def save(self):
        return {
            "position": self.position, "entry_price": self.entry_price,
            "live_pnl": self.live_pnl, "daily_loss": self.daily_loss,
            "trades_today": self.trades_today, "session_day": self.session_day,
            "skipped_slippage": self.skipped_slippage, "saved_at": time.time(),
            "day_open_balance": self.day_open_balance,
        }

    def restore(self, s):
        self.live_pnl     = s.get("live_pnl", 0.0)
        self.daily_loss   = s.get("daily_loss", 0.0)
        self.trades_today = s.get("trades_today", 0)
        self.skipped_slippage = s.get("skipped_slippage", 0)
        saved_day = s.get("session_day", self.session_day)
        # The balance anchor only means anything for the day it was taken on. On a
        # new day it is stale and must be dropped, or we would measure today's loss
        # against yesterday's opening balance.
        if saved_day == self.session_day:
            self.day_open_balance = s.get("day_open_balance")
        else:
            self.day_open_balance = None
            self.daily_loss = 0.0
            self.trades_today = 0
        self.session_day = saved_day if saved_day == self.session_day else self.session_day
        # deliberately do NOT restore an open position - we re-sync from the
        # platform instead. A stale position is how bots orphan live trades.


class Bot:
    def __init__(self, cfg_path):
        cfg = json.load(open(cfg_path))
        self.accounts = [RenkoVolAccount(a) for a in cfg["accounts"] if a.get("enabled", True)]
        self.running = True
        self.watchdog_tripped = False

    async def start(self):
        for a in self.accounts:
            await a.connect()
        if os.path.exists(STATE_FILE):
            try:
                st = json.load(open(STATE_FILE))
                for a in self.accounts:
                    if a.name in st.get("accounts", {}):
                        a.restore(st["accounts"][a.name])
                        print(f"[{a.name}] restored: day ${a.daily_loss:.0f}, "
                              f"{a.trades_today} trades, total ${a.live_pnl:+.0f}")
            except Exception as e:
                print(f"[STATE] restore failed: {e}")
        # make sure we agree with the platform about being flat
        for a in self.accounts:
            a._sync_init()
            await self._sync_flat(a)

    async def _sync_flat(self, a):
        sz = await a.platform_pos()
        if sz is None:
            print(f"[{a.name}] *** could not reach the platform at startup. "
                  f"Refusing to trade until it answers. ***")
            a.enabled = False
            return
        if sz != 0:
            a.position = 1 if sz > 0 else -1
            a.contracts_held = abs(sz)
            a.entry_price = float(await a.ctx.data.get_current_price())
            a.entry_time = time.time()
            print(f"[{a.name}] platform says OPEN: {sz} - adopting it")
        else:
            a.position = 0
            a.contracts_held = 0
            print(f"[{a.name}] platform says FLAT")
            # Anything resting on a flat account at boot is a leftover from a
            # previous process and can only ever OPEN a position. Clear it before
            # we trade a single contract.
            await a.sweep_orphan_orders()

    def persist(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({"accounts": {a.name: a.save() for a in self.accounts}}, f, indent=1)
        except Exception:
            pass

    async def heal_feeds(self):
        """Keep every account's data feed alive - reconnect first, exit last.

        The two ways this bot goes quietly deaf, both seen live on 13-Jul:
          * the market-data socket dies (SignalR 502 on the session rollover) and
            never comes back -> get_current_price() fails forever;
          * the HTTP session wedges -> every /Position/searchOpen times out, so
            enter() correctly refuses to trade blind... for as long as we let it.
        Neither raises. Neither exits. Both look 100% healthy to systemd.

        NEW behaviour (client's connection-recovery request): when a feed goes
        bad we first try to rebuild it IN-PROCESS (a.reconnect() -> fresh suite),
        with an exponential backoff between attempts. That heals a transient drop
        with no restart and no lost state. Only when the in-process reconnect
        keeps failing do we fall back to the original exit-and-let-systemd-rebuild
        path - so the last-resort safety net is unchanged. Returns True ONLY when
        the process must exit.
        """
        if not market_open():
            for a in self.accounts:         # silence is correct while shut
                a.last_tick_ts = time.time()
            return False
        now = time.time()
        must_exit = False
        for a in self.accounts:
            if not a.connected:
                continue
            quiet = now - a.last_tick_ts
            feed_dead = quiet > FROZEN_FEED_S
            session_wedged = a._api_fails >= MAX_API_FAILS
            # Third failure mode (Abhi, 17-Jul): the order/account (user) hub
            # drops while the market hub keeps ticking - the price feed looks
            # perfectly alive, so feed_dead stays False and the raw /Position
            # reads keep succeeding, so session_wedged stays False too, yet the
            # bot can no longer place or track orders. Watch the user hub flag
            # directly and rebuild once the outage is sustained (not a blip).
            if a.user_hub_down():
                if a._user_down_since is None:
                    a._user_down_since = now
            else:
                a._user_down_since = None
            user_hub_dead = (a._user_down_since is not None
                             and now - a._user_down_since > USER_HUB_DOWN_S)
            if not (feed_dead or session_wedged or user_hub_dead):
                continue
            # Back off between attempts so we don't spin on a still-dead endpoint.
            if now - a._last_reconnect < a._reconnect_cooldown:
                continue
            why = (f"no usable price for {quiet:.0f}s" if feed_dead
                   else f"{a._api_fails} platform reads timed out" if session_wedged
                   else f"order/account hub down {now - a._user_down_since:.0f}s")
            print(f"[{now_str()}] [{a.name}] *** feed unhealthy ({why}) while the "
                  f"market is OPEN. Rebuilding the connection in-process "
                  f"(attempt {a._reconnect_tries + 1}/{MAX_RECONNECT_TRIES}). Any open "
                  f"position keeps its stop - that order rests on the exchange. ***",
                  flush=True)
            a._last_reconnect = now
            ok = await a.reconnect()
            if ok:
                continue
            if a.health == "UNSAFE":
                # Locked/closed account. Restarting only crash-loops into the same
                # wall, so DON'T exit - disable this account and keep the process
                # (and any sibling account) alive. Client clears the lock, restarts.
                a.enabled = False
                continue
            a._reconnect_cooldown = min(a._reconnect_cooldown * 2, RECONNECT_COOLDOWN_MAX)
            if a._reconnect_tries >= MAX_RECONNECT_TRIES:
                print(f"[{now_str()}] [{a.name}] *** {a._reconnect_tries} in-process "
                      f"reconnects failed. Falling back to a clean systemd rebuild. ***",
                      flush=True)
                must_exit = True
        return must_exit

    async def run(self):
        await self.start()
        print(f"[BOT] running. Ctrl-C to stop.")
        # connect() can take half a minute. Start the watchdog's clock now, not
        # at __init__, or a slow login would look like a dead feed.
        for a in self.accounts:
            a.last_tick_ts = time.time()
        last_save = time.time()
        last_reconcile = time.time()
        while self.running:
            for a in self.accounts:
                if not a.connected:
                    continue
                try:
                    px = await asyncio.wait_for(a.ctx.data.get_current_price(), timeout=5.0)
                except Exception as e:
                    a.note_bad_tick(repr(e))
                    continue
                if not px or px <= 0:
                    a.note_bad_tick(f"price={px!r}")
                    continue
                # Drop phantom/stale quotes (100pt-off reads) before they can
                # price an entry or a stop. The engine is bar-fed, so this never
                # starves the bricks.
                if not a.live_price_is_sane(float(px)):
                    continue
                a.last_tick_ts = time.time()
                a.bad_ticks = 0
                try:
                    await a.on_price(float(px), time.time())
                except Exception as e:
                    print(f"[{a.name}] tick error: {e}")

            # Re-anchor the day's P&L to the broker. Cheap (one call / 30s) and it
            # keeps the daily-loss breaker honest even if our own tally drifts.
            for a in self.accounts:
                if a.connected and time.time() - a._last_pnl_sync > 30:
                    a._last_pnl_sync = time.time()
                    try:
                        await a.sync_daily_pnl()
                    except Exception as e:
                        print(f"[{a.name}] daily P&L sync error: {e!r}")

            if await self.heal_feeds():
                self.watchdog_tripped = True
                return
            # The resting stop can fire at any moment and the platform is the only
            # truth. Reconcile ALWAYS - including when we believe we are flat,
            # because "we think we are flat but we are not" is the dangerous half.
            if time.time() - last_reconcile > 2:
                for a in self.accounts:
                    if not a.connected:
                        continue
                    try:
                        await a.reconcile()
                    except Exception as e:
                        print(f"[{a.name}] reconcile error: {e!r}")
                last_reconcile = time.time()
            if time.time() - last_save > 5:
                self.persist()
                last_save = time.time()
            await asyncio.sleep(0.1)

    async def shutdown(self):
        self.running = False
        self.persist()
        for a in self.accounts:
            if a.suite:
                try:
                    await asyncio.wait_for(a.suite.disconnect(), timeout=5.0)
                except Exception:
                    pass
        print("[BOT] stopped.")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="accounts_renko.json")
    args = ap.parse_args()
    bot = Bot(args.config)
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, lambda: asyncio.create_task(bot.shutdown()))
    try:
        await bot.run()
    finally:
        await bot.shutdown()
    # Non-zero so the restart is visible in `systemctl status` as a fault rather
    # than a clean stop. systemd's Restart=always brings us back in 10s.
    return 1 if bot.watchdog_tripped else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
