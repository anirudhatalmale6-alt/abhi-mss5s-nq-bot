"""Renko variant of the NQ Signal strategy (Abhi A/B, 20-Jul-2026).

Same strategy as nq_signal_core, but built on ONE Renko brick stream instead of
time candles, so the client can compare identical logic WITH vs WITHOUT Renko
(Rutvi = time candles, Abhi = Renko). Per the client's spec:

  - ONE brick size.
  - Individual bricks = the fast / entry timeframe: the breakout, and the entry,
    fire off the UNCONFIRMED (still-forming) brick so it gets in fast - it does
    not wait for the brick to fully close.
  - Higher timeframe = every `group` bricks aggregated into one bar: the trend
    read and the Ichimoku/Bollinger filter live here ("smaller timeframe for
    entry, trend at the higher"). group=6 mirrors 30min = 6x 5min.

Everything else is identical to the time-candle version: same trend maths, same
Ichimoku/BB filter with the 5-point buffer, same 6pm-4pm session gate, no stop /
manual TP. We reuse the pure functions in nq_signal_core (calc_trend, calc_filter)
and an NQSignal object to hold the pending-breakout state and run decide().
"""

from renko_volume_engine import RenkoVolumeEngine
from nq_signal_core import NQSignal, calc_trend, calc_filter, in_session  # noqa: F401
from mss5s import MSS5s   # market-structure-shift on 5s candles (client 29-Jul, Abhi)


class RenkoFlip:
    """Clean 5-minute Renko flip strategy (Abhi, client's 20-Jul redesign).

    Everything on ONE timeframe (5-minute) and ONE brick size (2 points). No
    trend oscillator, no Ichimoku/BB - the direction IS the Renko:

      direction confirmed by `confirm` consecutive same-colour bricks. We catch
      the LAST brick EARLY as a "ghost": the moment live price crosses the level
      where the next brick would complete, we count it, so a 2-brick signal does
      not wait a full 5 minutes for the second brick to close.

      signal(price) -> 'LONG' | 'SHORT' | None
        LONG  when the last `confirm` bricks (confirmed + ghost) are all green
        SHORT when they are all red

    The bricks are fed from 5-MINUTE bar CLOSES (on_bar); the ghost is read off
    the live price. The trailing stop lives in the bot, not here.
    """

    def __init__(self, brick_size=2.0, confirm=2, tick_size=0.25, entry_frac=1.0,
                 ema_len=0):
        self.brick_size = float(brick_size)
        self.confirm = int(confirm)
        self.tick_size = float(tick_size)
        # entry_frac: how far into the FORMING (2nd) brick the live price must push
        # before we count it as a ghost of the continuation colour. 1.0 = wait for a
        # full brick (original behaviour); 0.10 = enter as soon as the 2nd brick is
        # ~10% formed ("1 brick + 10% of the second", client 21-Jul). Only the
        # CONTINUATION side is early; a colour REVERSAL still needs the real 2x-brick
        # flip level so a fractional wobble can never fake a reversal.
        self.entry_frac = float(entry_frac)
        self.eng = RenkoVolumeEngine(brick_size=self.brick_size, tick_size=tick_size)
        # ---- 9-EMA cross mode (client 22-Jul) -------------------------------
        # ema_len > 0 -> run an EMA over the BRICK CLOSES. The signal is no longer
        # "N same-colour bricks" but "a brick just CROSSED the EMA" (breakout of the
        # 9-EMA). The crossing brick arms a direction; we enter on the SECOND brick
        # forming that way (the ghost). The stop is the brick BEFORE the crossing
        # brick, which gives the trade room. 0 = off -> plain confirm-N logic.
        self.ema_len = int(ema_len)
        self.ema = None
        self._ema_side = None    # +1 above EMA / -1 below, at the last completed brick
        self._ema_seen = 0       # monotonic bricks already folded into the EMA
        self._prev_brick = None  # the brick before the one being processed (for the stop)
        self._cross_dir = None   # 'green' | 'red' armed by a brick crossing the EMA
        self._cross_stop = None  # stop level = extreme of the brick BEFORE the cross
        # ---- 15-second Bollinger-Band entry trigger (client 27-Jul) ----------
        # The 9-EMA brick cross ARMS a direction (above). Instead of entering on the
        # ghost, we now WAIT for a 15-second Bollinger-Band (20, 1.5sd) close to fade
        # back the other way (reversal): armed LONG enters when a 15s candle CLOSES
        # BELOW the lower band; armed SHORT enters when a 15s candle CLOSES ABOVE the
        # upper band. One entry per armed cross. Bands are built from 15-second
        # candles (bucketed from the live 1-second price feed, warmed from history).
        self.bb_len = 20
        self.bb_std = 1.5
        self.bb_tf_s = 15
        self._bb_closes = []       # last bb_len completed 15s closes
        self._bb_mid = None
        self._bb_up = None
        self._bb_lo = None
        self._last_bb = None       # {'open','high','low','close'} of last COMPLETED 15s candle
        self._bb_count = 0         # monotonic count of completed 15s candles
        self._arm_bb_count = 0     # _bb_count captured at the moment the cross armed
        self._bb_bucket = None     # current 15s bucket id (floor(ts/bb_tf_s))
        self._bb_o = self._bb_h = self._bb_l = self._bb_c = None
        # ---- swing High/Low support & resistance levels (client 27-Jul, Abhi) --
        # The bot builds its own levels from the PRICE ACTION: a run of same-colour
        # bricks that ENDS (colour flips) leaves a pivot - a green run ending is a
        # HIGH that got rejected (resistance), a red run ending is a LOW that
        # bounced (support). Those are the "previous points of bounces / rejections".
        # STRICT mode then only lets a fade fire when price is sitting AT a valid
        # level (short a pop into resistance, long a dip into support) and skips
        # everything in open space or breaking clean through - "don't enter when
        # you're not supposed to". A level is BROKEN (and flips R<->S) once a brick
        # closes decisively through it. Off unless the bot enables it.
        self.levels_on = False
        self.level_tol_pts = 6.0     # price within this of a level counts as "at" it
        self.level_break_pts = 2.0   # brick close beyond a level by this = broken
        self.swing_min_run = 2       # min same-colour bricks in a run to make a pivot
        self.level_max = 40          # keep the most-recent N levels
        self._levels = []            # [{price, side:'R'/'S', born, touches, flipped}]
        self._lv_seen = 0            # monotonic bricks folded into the level tracker
        self._run_dir = None         # colour of the current brick run
        self._run_ext = None         # extreme of the current run (high green / low red)
        self._run_len = 0
        self._last_level_ctx = None  # last level snapshot, for the bot to log
        # ---- Elliott wave-3 trend-riding mode (client 27-Jul, Rutvi) ----------
        # Reads the SAME swing pivots the levels use (green-run-end = swing HIGH,
        # red-run-end = swing LOW) but instead of fading, it counts the impulse legs
        # and only trades WAVE 3 - the strong continuation leg. Up-impulse wave 3:
        # a low, a high, then a HIGHER low, and price breaks back above the wave-1
        # high (higher-high + higher-low structure). Down-impulse is the mirror.
        # Once wave 3 is confirmed it ARMS a direction and enters on the next 15s
        # BB signal in that direction (buy the dip into the lower band in an
        # up-impulse / sell the rip into the upper band in a down-impulse). This
        # RIDES the trend rather than fading it. Off unless the bot enables it.
        self.wave_on = False
        self.wave_max_pivots = 60
        self._pivots = []            # chronological [{price, kind:'H'/'L', idx}]
        self._wave_dir = None        # 'green'(long) / 'red'(short) armed by a wave 3
        self._wave_arm_bb = 0        # _bb_count captured when the wave armed
        self._wave_leg_id = None     # id of the armed wave-3 leg (avoid re-arming it)
        self._last_wave_ctx = None   # last wave snapshot, for the bot to log
        # ---- 8-EMA trend-REVERSAL mode (client 28-Jul, Rutvi; refined 28-Jul) --
        # Counts 8-EMA BREAKS, exactly as the client marked on his chart. In an
        # up-move price rides above the 8 EMA, then BREAKS below it - that first
        # break is count 1. Price rallies back but makes a LOWER high and breaks
        # below the 8 EMA AGAIN - that second break is count 2 -> SHORT (the trend
        # reversal). Each counted break must come off a LOWER high than the one
        # before (else the count resets - a higher high means the up-trend is still
        # intact). LONG is the mirror: two breaks ABOVE the 8 EMA off successively
        # HIGHER lows. Every entry must also agree with the 1-HOUR trend (set from
        # the bot) - "the signal should match the 1h trend, that's key". Entry is at
        # market on the second break; the stop sits just past that last swing (the
        # lower high we shorted / the higher low we bought) for a 1:1 target.
        self.reversal_on = False
        self.rev_stop_buf_pts = 2.0   # stop this far beyond the last lower-high/higher-low
        self._rev_seen = 0            # monotonic bricks folded into the reversal tracker
        self._rev_ema = None          # 8 EMA of brick closes (reversal's own copy)
        self._rev_side = None         # +1 last brick closed ABOVE the ema / -1 below
        self._rev_exc_hi = None       # running high of the current ABOVE-ema excursion
        self._rev_exc_lo = None       # running low of the current BELOW-ema excursion
        self._rev_last_hi = None      # excursion high captured at the previous break-DOWN
        self._rev_last_lo = None      # excursion low captured at the previous break-UP
        self._rev_short_n = 0         # consecutive break-downs off descending highs
        self._rev_long_n = 0          # consecutive break-ups off ascending lows
        self._rev_pending = None      # {'dir','ref'} set on a live qualifying break
        self._rev_warm = False        # True once the historical catch-up is folded;
                                      # only then may a qualifying break arm a pending
        self._h1_dir = None           # 1-hour trend colour, pushed in by the bot
        self._last_rev_ctx = None     # last reversal snapshot, for the bot to log
        # ---- MSS-on-5s (client 29-Jul, Abhi only) ---------------------------
        # Same shift concept as the reversal, but driven by 5-SECOND time candles
        # (ZigZag swings) instead of bricks, and with NO 1h gate. When on, signal()
        # returns the MSS tracker's call and populates _last_rev_ctx so the bot's
        # existing reversal entry/stop plumbing is reused unchanged.
        self.mss_on = False
        self._mss   = None

    def configure_mss(self, on, ema_len=8, stop_buf=2.0, confirm=1, zz_thresh=15.0):
        self.mss_on = bool(on)
        if on:
            self._mss = MSS5s(ema_len=ema_len, stop_buf_pts=stop_buf,
                              confirm=confirm, zz_thresh_pts=zz_thresh)

    def on_5s(self, o, h, l, c):
        """Fold ONE closed 5-second candle into the MSS tracker."""
        if self._mss is not None:
            self._mss.update(o, h, l, c)

    def mss_mark_warm(self):
        if self._mss is not None:
            self._mss.mark_warm()

    def mss_context(self):
        m = self._mss
        if m is None:
            return {}
        return {"ema": (round(m.ema, 2) if m.ema is not None else None),
                "armed": m.armed_dir, "hi": m.last_hi, "lo": m.last_lo,
                "sN": m.short_n, "lN": m.long_n, "warm": m.warm}

    def configure_reversal(self, on, stop_buf=None):
        self.reversal_on = bool(on)
        if stop_buf is not None:
            self.rev_stop_buf_pts = float(stop_buf)

    def mark_reversal_warm(self):
        """Called by the bot once the historical bricks are folded in at startup.
        Folds any remaining history WITHOUT arming, then lets live breaks fire."""
        self._update_reversal()       # absorb all history so nothing stale is pending
        self._rev_pending = None
        self._rev_warm = True

    def set_h1_trend(self, color):
        """The bot pushes the current 1-hour trend colour ('green'/'red'/None) here;
        a reversal only fires when it agrees with the 1h trend."""
        self._h1_dir = color if color in ("green", "red") else None

    def configure_wave(self, on, min_run=None, max_pivots=None):
        self.wave_on = bool(on)
        if min_run is not None:
            self.swing_min_run = int(min_run)
        if max_pivots is not None:
            self.wave_max_pivots = int(max_pivots)

    def configure_levels(self, on, tol=None, brk=None, min_run=None, max_n=None):
        self.levels_on = bool(on)
        if tol is not None:
            self.level_tol_pts = float(tol)
        if brk is not None:
            self.level_break_pts = float(brk)
        if min_run is not None:
            self.swing_min_run = int(min_run)
        if max_n is not None:
            self.level_max = int(max_n)

    def on_bar(self, close, ts):
        """Feed one CLOSED 5-minute bar close. Forms confirmed bricks."""
        self.eng.on_bar_close(close, ts)

    def _ghost_color(self, price):
        """Colour of the brick currently FORMING from the live price, using the
        engine's exact next-brick levels (which respect the 2x reversal). None if
        no new brick is forming yet."""
        nl = self.eng.next_levels()
        if not nl:
            return None
        anchor = self.eng._anchor
        bs = self.eng.brick_size
        # early continuation: fire once price has pushed entry_frac of a brick past
        # the last close, not the full brick. Reversal (flip) still needs the true
        # 2x-brick level from next_levels().
        cont_up = anchor + self.entry_frac * bs
        cont_dn = anchor - self.entry_frac * bs
        if nl["dir"] == "green":
            if price >= cont_up:
                return "green"
            if price <= nl["flip"]:
                return "red"
        else:
            if price <= cont_dn:
                return "red"
            if price >= nl["flip"]:
                return "green"
        return None

    def _seq(self, price):
        seq = [b["direction"] for b in self.eng.bricks]
        g = self._ghost_color(price)
        if g:
            seq = seq + [g]
        return seq

    def _update_ema(self):
        """Fold every brick that has completed since the last call into the EMA of
        brick closes, and detect a CROSS (a brick close moving to the other side of
        the EMA). A cross arms a direction and records the stop = the brick BEFORE
        the crossing brick. Uses the engine's monotonic counter so it stays correct
        even after bricks[] truncates at 1000."""
        if self.ema_len <= 0:
            return
        total = self.eng._total_bricks
        if total <= self._ema_seen:
            return
        new = total - self._ema_seen
        tail = self.eng.bricks[-new:] if 0 < new <= len(self.eng.bricks) else list(self.eng.bricks)
        k = 2.0 / (self.ema_len + 1.0)
        for b in tail:
            c = b["close"]
            self.ema = c if self.ema is None else (c * k + self.ema * (1.0 - k))
            side = 1 if c > self.ema else (-1 if c < self.ema else self._ema_side)
            if self._ema_side is not None and side is not None and side != self._ema_side:
                # THIS brick just crossed the EMA -> arm the breakout direction.
                self._cross_dir = "green" if side > 0 else "red"
                # remember how many 15s candles had closed at arming time, so the BB
                # trigger only fires on a FRESH 15s close AFTER the arm (not a stale one).
                self._arm_bb_count = self._bb_count
                pre = self._prev_brick if self._prev_brick is not None else b
                if self._cross_dir == "green":     # long: stop below the pre-cross brick
                    self._cross_stop = min(pre["open"], pre["close"]) - self.tick_size
                else:                              # short: stop above the pre-cross brick
                    self._cross_stop = max(pre["open"], pre["close"]) + self.tick_size
            self._ema_side = side
            self._prev_brick = b
        self._ema_seen = total

    def _bb_recompute(self):
        n = len(self._bb_closes)
        if n < self.bb_len:
            self._bb_mid = self._bb_up = self._bb_lo = None
            return
        m = sum(self._bb_closes) / n
        var = sum((x - m) ** 2 for x in self._bb_closes) / n   # population std
        sd = var ** 0.5
        self._bb_mid = m
        self._bb_up = m + self.bb_std * sd
        self._bb_lo = m - self.bb_std * sd

    def push_bb_close(self, o, h, l, c):
        """Fold ONE completed 15-second candle into the Bollinger Band window and
        recompute the bands. Used by the live bucketer and the history warmup."""
        self._last_bb = {"open": float(o), "high": float(h),
                         "low": float(l), "close": float(c)}
        self._bb_closes.append(float(c))
        if len(self._bb_closes) > self.bb_len:
            self._bb_closes = self._bb_closes[-self.bb_len:]
        self._bb_count += 1
        self._bb_recompute()

    def feed_bb_tick(self, price, ts):
        """Bucket the live price into 15-second candles; when a bucket rolls over,
        the just-completed candle is folded into the bands via push_bb_close."""
        b = int(ts // self.bb_tf_s)
        if self._bb_bucket is None:
            self._bb_bucket = b
            self._bb_o = self._bb_h = self._bb_l = self._bb_c = price
            return
        if b == self._bb_bucket:
            self._bb_c = price
            if price > self._bb_h:
                self._bb_h = price
            if price < self._bb_l:
                self._bb_l = price
            return
        # bucket rolled over -> the previous 15s candle is complete
        self.push_bb_close(self._bb_o, self._bb_h, self._bb_l, self._bb_c)
        self._bb_bucket = b
        self._bb_o = self._bb_h = self._bb_l = self._bb_c = price

    def _add_level(self, price, side, born):
        """Register a pivot as a level. If one of the SAME side already sits within
        tolerance, treat this as a retest: nudge it toward the new price and bump its
        touch count instead of stacking a duplicate (a level hit twice is stronger)."""
        for lv in self._levels:
            if lv["side"] == side and abs(lv["price"] - price) <= self.level_tol_pts:
                lv["price"] = (lv["price"] + float(price)) / 2.0
                lv["touches"] += 1
                lv["born"] = int(born)
                return
        self._levels.append({"price": float(price), "side": side,
                             "born": int(born), "touches": 1, "flipped": False})
        if len(self._levels) > self.level_max:
            self._levels = self._levels[-self.level_max:]

    def _update_levels(self):
        """Fold every brick completed since the last call: track the current colour
        run and, when it ends, drop a pivot (green run -> resistance, red run ->
        support) if it was at least swing_min_run bricks. Also flip a level R<->S the
        moment a brick closes decisively through it (broken resistance becomes
        support and vice-versa)."""
        if not (self.levels_on or self.wave_on):
            return
        total = self.eng._total_bricks
        if total <= self._lv_seen:
            return
        new = total - self._lv_seen
        tail = self.eng.bricks[-new:] if 0 < new <= len(self.eng.bricks) else list(self.eng.bricks)
        for b in tail:
            d = b["direction"]
            hi = max(b["open"], b["close"])
            lo = min(b["open"], b["close"])
            # break / flip: a brick that CLOSES through a level invalidates it and
            # flips its role (classic broken-resistance-becomes-support).
            for lv in self._levels:
                if lv["side"] == "R" and b["close"] > lv["price"] + self.level_break_pts:
                    lv["side"] = "S"
                    lv["flipped"] = True
                elif lv["side"] == "S" and b["close"] < lv["price"] - self.level_break_pts:
                    lv["side"] = "R"
                    lv["flipped"] = True
            # run tracking
            if d == self._run_dir:
                self._run_len += 1
                self._run_ext = max(self._run_ext, hi) if d == "green" else min(self._run_ext, lo)
            else:
                if self._run_dir is not None and self._run_len >= self.swing_min_run:
                    side = "R" if self._run_dir == "green" else "S"
                    if self.levels_on:
                        self._add_level(self._run_ext, side, total)
                    if self.wave_on:
                        self._add_pivot(self._run_ext,
                                        "H" if self._run_dir == "green" else "L", total)
                self._run_dir = d
                self._run_len = 1
                self._run_ext = hi if d == "green" else lo
        self._lv_seen = total

    def level_context(self, price):
        """Nearest still-valid resistance (near/above price) and support (near/below),
        plus whether price is sitting AT one within tolerance. Cached for the trade log."""
        self._update_levels()
        nearest_R = nearest_S = None
        for lv in self._levels:
            if lv["side"] == "R" and lv["price"] >= price - self.level_tol_pts:
                if nearest_R is None or lv["price"] < nearest_R["price"]:
                    nearest_R = lv
            elif lv["side"] == "S" and lv["price"] <= price + self.level_tol_pts:
                if nearest_S is None or lv["price"] > nearest_S["price"]:
                    nearest_S = lv
        at_R = nearest_R is not None and abs(nearest_R["price"] - price) <= self.level_tol_pts
        at_S = nearest_S is not None and abs(nearest_S["price"] - price) <= self.level_tol_pts
        ctx = {
            "at_support": bool(at_S), "at_resistance": bool(at_R),
            "R_price": (round(nearest_R["price"], 2) if nearest_R else None),
            "S_price": (round(nearest_S["price"], 2) if nearest_S else None),
            "R_touches": (nearest_R["touches"] if nearest_R else 0),
            "S_touches": (nearest_S["touches"] if nearest_S else 0),
            "R_flipped": (bool(nearest_R["flipped"]) if nearest_R else False),
            "S_flipped": (bool(nearest_S["flipped"]) if nearest_S else False),
            "n_levels": len(self._levels),
        }
        self._last_level_ctx = ctx
        return ctx

    def _add_pivot(self, price, kind, idx):
        """Append a swing pivot ('H' swing high / 'L' swing low). Runs alternate
        colour so pivots should already alternate H/L; if the same kind repeats
        (rare), keep the more extreme one so the wave count stays clean."""
        if self._pivots and self._pivots[-1]["kind"] == kind:
            p = self._pivots[-1]
            if (kind == "H" and price > p["price"]) or (kind == "L" and price < p["price"]):
                p["price"] = float(price)
                p["idx"] = int(idx)
            return
        self._pivots.append({"price": float(price), "kind": kind, "idx": int(idx)})
        if len(self._pivots) > self.wave_max_pivots:
            self._pivots = self._pivots[-self.wave_max_pivots:]

    def wave_context(self, price):
        """Classify the last 3 swing pivots as a wave-3 impulse, if any.

        Up-impulse wave 3 (LONG): pivots read Low, High, HIGHER-Low and the live
        price has broken back ABOVE the wave-1 high - higher-high + higher-low, the
        classic 'making a higher high' the client described. Down-impulse (SHORT) is
        the mirror: High, Low, LOWER-High and price breaks below the wave-1 low.
        Returns leg='W3' with dir 'green'/'red' when armed, else leg=None."""
        self._update_levels()   # folds new bricks -> pivots
        ctx = {"leg": None, "dir": None, "w1": None, "w2": None,
               "n_pivots": len(self._pivots)}
        P = self._pivots
        if len(P) >= 3:
            a, b, c = P[-3], P[-2], P[-1]
            if a["kind"] == "L" and b["kind"] == "H" and c["kind"] == "L":
                # up impulse: wave1 up (a<b), higher low (c>a), wave3 breaks the high
                if b["price"] > a["price"] and c["price"] > a["price"] \
                        and price > b["price"] + self.tick_size:
                    ctx.update(leg="W3", dir="green",
                               w1=round(b["price"], 2), w2=round(c["price"], 2))
            elif a["kind"] == "H" and b["kind"] == "L" and c["kind"] == "H":
                # down impulse: wave1 down (a>b), lower high (c<a), wave3 breaks the low
                if b["price"] < a["price"] and c["price"] < a["price"] \
                        and price < b["price"] - self.tick_size:
                    ctx.update(leg="W3", dir="red",
                               w1=round(b["price"], 2), w2=round(c["price"], 2))
        self._last_wave_ctx = ctx
        return ctx

    def _wave_signal(self, price):
        """Wave-3 trend-riding entry. Structure ARMS a direction; the 15s BB then
        times the entry IN that direction (buy the dip into the lower band on an
        up-impulse, sell the rip into the upper band on a down-impulse). One entry
        per wave-3 leg."""
        if self.ema_len > 0:
            self._update_ema()       # keep the EMA current for status/logging only
        wc = self.wave_context(price)
        if wc["leg"] == "W3":
            leg_id = (wc["dir"], wc["w1"], wc["w2"])
            if leg_id != self._wave_leg_id:   # a fresh wave-3 leg -> (re)arm
                self._wave_dir = wc["dir"]
                self._wave_arm_bb = self._bb_count
                self._wave_leg_id = leg_id
        if self._wave_dir is None:
            return None
        # bands warming, or no fresh 15s candle has closed since the arm
        if self._bb_lo is None or self._last_bb is None:
            return None
        if self._bb_count <= self._wave_arm_bb:
            return None
        c = self._last_bb["close"]
        if self._wave_dir == "green":
            if c < self._bb_lo:      # buy the pullback into support of the up-impulse
                self._wave_dir = None
                return "LONG"
        else:
            if c > self._bb_up:      # sell the pullback into resistance of the down-impulse
                self._wave_dir = None
                return "SHORT"
        return None

    def _update_reversal(self):
        """Fold every brick completed since the last call into the 8-EMA break
        tracker. Runs the client's count: a brick close crossing from ABOVE the ema
        to BELOW it is a break-down; consecutive break-downs coming off LOWER highs
        increment the short count (a higher high resets it). A break-up off a HIGHER
        low increments the long count. When a count reaches 2 AFTER WARMUP, a pending
        reversal is armed for the bot. Arming is gated on self._rev_warm (set once the
        historical catch-up is folded) so the startup history never fires a stale entry,
        while any live break - even several bricks folded at once on a feed hiccup -
        still arms correctly."""
        if self.ema_len <= 0:
            return
        total = self.eng._total_bricks
        if total <= self._rev_seen:
            return
        new = total - self._rev_seen
        tail = self.eng.bricks[-new:] if 0 < new <= len(self.eng.bricks) else list(self.eng.bricks)
        k = 2.0 / (self.ema_len + 1.0)
        for i, b in enumerate(tail):
            c = b["close"]
            hi = max(b["open"], b["close"])
            lo = min(b["open"], b["close"])
            self._rev_ema = c if self._rev_ema is None else (c * k + self._rev_ema * (1.0 - k))
            side = 1 if c > self._rev_ema else (-1 if c < self._rev_ema else self._rev_side)
            if side is None:            # price sits exactly on the ema and no side yet
                continue
            # extend the running excursion on the side we are on
            if side > 0:
                self._rev_exc_hi = hi if self._rev_exc_hi is None else max(self._rev_exc_hi, hi)
            elif side < 0:
                self._rev_exc_lo = lo if self._rev_exc_lo is None else min(self._rev_exc_lo, lo)
            if self._rev_side is not None and side is not None and side != self._rev_side:
                if side < 0:
                    # BREAK DOWN through the ema: the just-ended above-ema excursion
                    # made a high of self._rev_exc_hi. Count it if it is a LOWER high.
                    this_hi = self._rev_exc_hi
                    if this_hi is not None:
                        if self._rev_last_hi is not None and this_hi < self._rev_last_hi:
                            self._rev_short_n += 1
                        else:
                            self._rev_short_n = 1     # new (or higher) high -> restart the count
                        self._rev_last_hi = this_hi
                        if self._rev_short_n >= 2 and self._rev_warm:
                            self._rev_pending = {"dir": "red", "ref": this_hi}
                    # NB: do NOT reset the long count here - the rally that follows a
                    # break-down is exactly how the next higher low forms.
                    self._rev_exc_lo = lo            # start the new below-ema excursion
                    self._rev_exc_hi = None
                else:
                    # BREAK UP through the ema: count it if it is a HIGHER low.
                    this_lo = self._rev_exc_lo
                    if this_lo is not None:
                        if self._rev_last_lo is not None and this_lo > self._rev_last_lo:
                            self._rev_long_n += 1
                        else:
                            self._rev_long_n = 1
                        self._rev_last_lo = this_lo
                        if self._rev_long_n >= 2 and self._rev_warm:
                            self._rev_pending = {"dir": "green", "ref": this_lo}
                    # NB: do NOT reset the short count here - the pullback that follows
                    # a break-up is exactly how the next lower high forms.
                    self._rev_exc_hi = hi
                    self._rev_exc_lo = None
            self._rev_side = side
        self._rev_seen = total

    def reversal_context(self, price):
        """Snapshot of the reversal tracker for logging: current ema, which side of
        it price is on, the running break counts and any armed pending reversal."""
        self._update_reversal()
        ctx = {"dir": None, "stop": None, "risk_pts": None,
               "ema": (round(self._rev_ema, 2) if self._rev_ema is not None else None),
               "h1": self._h1_dir,
               "short_n": self._rev_short_n, "long_n": self._rev_long_n,
               "pending": (self._rev_pending["dir"] if self._rev_pending else None)}
        self._last_rev_ctx = ctx
        return ctx

    def _reversal_signal(self, price):
        """Fire a reversal at market on the second 8-EMA break, but only when it
        agrees with the 1-hour trend. Evaluated once per armed break (consumed here
        so it never double-fires)."""
        self._update_reversal()
        pend = self._rev_pending
        self._rev_pending = None            # evaluate exactly once
        # keep the snapshot fresh for the log either way
        base = {"dir": None, "stop": None, "risk_pts": None,
                "ema": (round(self._rev_ema, 2) if self._rev_ema is not None else None),
                "h1": self._h1_dir,
                "short_n": self._rev_short_n, "long_n": self._rev_long_n}
        if not pend or self._rev_ema is None:
            self._last_rev_ctx = base
            return None
        d, ref = pend["dir"], pend["ref"]
        if d == "red":
            # must be below the ema now AND the 1h trend must be down
            if price >= self._rev_ema or self._h1_dir != "red":
                self._last_rev_ctx = {**base, "skip": f"h1={self._h1_dir}"}
                return None
            stop = ref + self.rev_stop_buf_pts
            self._last_rev_ctx = {**base, "dir": "red", "stop": round(stop, 2),
                                  "risk_pts": round(stop - price, 2), "ref_high": round(ref, 2)}
            return "SHORT"
        else:
            if price <= self._rev_ema or self._h1_dir != "green":
                self._last_rev_ctx = {**base, "skip": f"h1={self._h1_dir}"}
                return None
            stop = ref - self.rev_stop_buf_pts
            self._last_rev_ctx = {**base, "dir": "green", "stop": round(stop, 2),
                                  "risk_pts": round(price - stop, 2), "ref_low": round(ref, 2)}
            return "LONG"

    def signal(self, price):
        # ---- MSS-on-5s mode (client 29-Jul, Abhi) ----------------------------
        # Driven by folded 5s candles (on_5s), not bricks. Returns the shift call
        # and fills _last_rev_ctx so the bot's reversal entry path works as-is.
        if self.mss_on and self._mss is not None:
            res = self._mss.poll(price)
            if not res:
                return None
            d, stop = res
            self._last_rev_ctx = {
                "dir": ("red" if d == "SHORT" else "green"),
                "stop": round(stop, 2),
                "risk_pts": round(abs(stop - price), 2),
                "ema": (round(self._mss.ema, 2) if self._mss.ema is not None else None)}
            return d
        # ---- 8-EMA trend-reversal mode (client 28-Jul) -----------------------
        if self.reversal_on:
            return self._reversal_signal(price)
        # ---- Elliott wave-3 trend-riding mode (client 27-Jul) ----------------
        if self.wave_on:
            return self._wave_signal(price)
        # ---- 9-EMA cross ARM + 15s Bollinger-Band reversal entry -------------
        if self.ema_len > 0:
            self._update_ema()
            if self._cross_dir is None or self.ema is None:
                return None
            # bands still warming, or no fresh 15s candle has closed since the arm
            if self._bb_lo is None or self._last_bb is None:
                return None
            if self._bb_count <= self._arm_bb_count:
                return None
            c = self._last_bb["close"]
            if self._cross_dir == "green":
                # armed LONG (cross up) -> fade: enter when a 15s candle CLOSES
                # below the lower band. One entry per armed cross.
                if c < self._bb_lo:
                    # STRICT levels: only buy the dip if it is AT a support level.
                    if self.levels_on and not self.level_context(price)["at_support"]:
                        self._arm_bb_count = self._bb_count   # wait for a fresh 15s close
                        return None
                    self._cross_dir = None
                    return "LONG"
            else:
                # armed SHORT (cross down) -> fade: enter on a 15s close ABOVE upper.
                if c > self._bb_up:
                    # STRICT levels: only sell the rip if it is AT a resistance level.
                    if self.levels_on and not self.level_context(price)["at_resistance"]:
                        self._arm_bb_count = self._bb_count
                        return None
                    self._cross_dir = None
                    return "SHORT"
            return None
        # ---- plain confirm-N same-colour mode --------------------------------
        seq = self._seq(price)
        if len(seq) < self.confirm:
            return None
        tail = seq[-self.confirm:]
        if all(c == "green" for c in tail):
            return "LONG"
        if all(c == "red" for c in tail):
            return "SHORT"
        return None

    def ready(self):
        if self.ema_len > 0:
            return len(self.eng.bricks) >= self.ema_len and self.ema is not None
        return len(self.eng.bricks) >= self.confirm

    def status(self):
        cols = [b["direction"][0] for b in self.eng.bricks[-8:]]
        if self.ema_len > 0:
            self._update_ema()
            ema = f"{self.ema:.2f}" if self.ema is not None else "warming"
            bb = (f"{self._bb_lo:.1f}/{self._bb_up:.1f}"
                  if self._bb_lo is not None else f"warming({len(self._bb_closes)}/{self.bb_len})")
            lv = ""
            if self.levels_on:
                self._update_levels()
                nR = sum(1 for x in self._levels if x["side"] == "R")
                nS = sum(1 for x in self._levels if x["side"] == "S")
                lv = f" levels[{len(self._levels)}: {nR}R/{nS}S]"
            wv = ""
            if self.wave_on:
                self._update_levels()
                nH = sum(1 for x in self._pivots if x["kind"] == "H")
                nL = sum(1 for x in self._pivots if x["kind"] == "L")
                wv = f" wave[armed={self._wave_dir} pivots={len(self._pivots)}: {nH}H/{nL}L]"
            rv = ""
            if self.reversal_on:
                self._update_reversal()
                sd = "above" if (self._rev_side or 0) > 0 else ("below" if (self._rev_side or 0) < 0 else "?")
                re = f"{self._rev_ema:.1f}" if self._rev_ema is not None else "warming"
                rv = (f" rev[ema8={re} price={sd} shortN={self._rev_short_n} "
                      f"longN={self._rev_long_n} h1={self._h1_dir}]")
            return (f"renkoflip brick={self.brick_size}pt ema{self.ema_len}={ema} | "
                    f"bricks={len(self.eng.bricks)} last8={''.join(cols)} "
                    f"armed={self._cross_dir} bb15s={bb}{lv}{wv}{rv}")
        return (f"renkoflip brick={self.brick_size}pt confirm={self.confirm} | "
                f"bricks={len(self.eng.bricks)} last8={''.join(cols)}")


def _brick_to_bar(b):
    o, c = b["open"], b["close"]
    return {"open": o, "close": c, "high": max(o, c), "low": min(o, c)}


class NQSignalRenko:
    def __init__(self, brick_size, group=6, confirm_frac=0.70, trend_on_htf=True,
                 use_time_htf=False, tick_size=0.25, max_bars=600):
        self.brick_size = float(brick_size)
        self.group = int(group)
        self.confirm_frac = float(confirm_frac)   # unconfirmed-brick entry fraction
        self.trend_on_htf = bool(trend_on_htf)    # trend on higher group vs fast bricks
        # use_time_htf: take the trend + filter from REAL 5m/30m bars (set_htf),
        # not from grouped bricks. The brick stream then drives ONLY the breakout
        # and the flip. This matches the client's actual 5min/30min charts.
        self.use_time_htf = bool(use_time_htf)
        self.eng = RenkoVolumeEngine(brick_size=self.brick_size, tick_size=tick_size)
        self.sig = NQSignal()
        self.bricks = []        # individual brick-bars (fast / entry)
        self.htf = []           # aggregated higher-timeframe bars (trend + filter)
        self._grp = []          # bricks accumulating toward the next htf bar
        self._last_close = None  # close of the last CONFIRMED brick
        self._prov_dir = None   # provisional direction already armed off the forming brick
        self._max = max_bars

    # -- expose the same surface the bot's decide()/diagnostic read -----------
    @property
    def up(self):           return self.sig.up
    @property
    def down(self):         return self.sig.down
    @property
    def filt(self):         return self.sig.filt
    @property
    def pending_buy(self):  return self.sig.pending_buy
    @property
    def pending_sell(self): return self.sig.pending_sell
    @property
    def break_high(self):   return self.sig.break_high
    @property
    def break_low(self):    return self.sig.break_low

    def _arm(self, direction, level):
        """Latch a pending breakout, exactly like the time-candle colour flip."""
        if direction == "green":
            self.sig.break_high = level
            self.sig.pending_buy = True
            self.sig.pending_sell = False
        else:
            self.sig.break_low = level
            self.sig.pending_sell = True
            self.sig.pending_buy = False

    def _refresh_trend_filter(self):
        src = self.htf if self.trend_on_htf else self.bricks
        if len(src) >= 6:
            self.sig.up, self.sig.down = calc_trend(src)
        if len(self.htf) >= 26:                     # filter always on the higher TF
            self.sig.filt = calc_filter(self.htf)

    def _ingest(self, new_bricks, price, lp):
        """Process any bricks that just completed, then run the unconfirmed-brick
        arming off the live price. Shared by the live (on_tick) and warmup (on_bar)
        paths so both build the exact same state."""
        for b in new_bricks:
            self.bricks.append(_brick_to_bar(b))
            if len(self.bricks) > self._max:
                self.bricks = self.bricks[-self._max:]
            self._last_close = b["close"]
            self._prov_dir = b["direction"]          # a confirmed brick resets provisional
            # aggregate into the higher timeframe
            self._grp.append(b)
            if len(self._grp) >= self.group:
                o = self._grp[0]["open"]
                c = self._grp[-1]["close"]
                hi = max(max(x["open"], x["close"]) for x in self._grp)
                lo = min(min(x["open"], x["close"]) for x in self._grp)
                self.htf.append({"open": o, "close": c, "high": hi, "low": lo})
                if len(self.htf) > self._max:
                    self.htf = self.htf[-self._max:]
                self._grp = []
                if not self.use_time_htf:
                    self._refresh_trend_filter()
            elif not self.trend_on_htf and not self.use_time_htf:
                # trend lives on the fast bricks -> refresh every brick
                self._refresh_trend_filter()

        # UNCONFIRMED brick: arm the breakout as soon as the forming brick has
        # moved `confirm_frac` of a brick beyond the last close in a NEW direction,
        # before it fully closes. This is the "based off unconfirmed brick" entry.
        if self._last_close is not None:
            move = price - self._last_close
            thr = self.confirm_frac * self.brick_size
            if move >= thr and self._prov_dir != "green":
                self._prov_dir = "green"
                self._arm("green", lp)
            elif move <= -thr and self._prov_dir != "red":
                self._prov_dir = "red"
                self._arm("red", lp)

    def on_tick(self, price, ts, live_price=None):
        """LIVE path: called every loop tick (~0.1s) with the current price and an
        epoch-seconds ts. The engine buckets ticks into 1-second closes internally,
        so bricks form off 1-second closes (not off every raw tick)."""
        lp = live_price if live_price is not None else price
        self._ingest(self.eng.on_tick(price, ts), price, lp)

    def on_bar(self, price, ts, live_price=None):
        """WARMUP path: feed one already-closed 1-second bar. Each call IS a bar
        close, so the engine forms bricks immediately from the historical series."""
        lp = live_price if live_price is not None else price
        self._ingest(self.eng.on_bar_close(price, ts), price, lp)

    def set_htf(self, bars5, bars30):
        """Set the trend + filter from REAL higher-timeframe bars (actual 5-minute
        and 30-minute candles). Used when use_time_htf is on: the brick stream
        drives only the breakout/flip, the trend context comes from the real
        5m/30m charts the client watches. Cheap; safe to call every few seconds."""
        if bars5 and len(bars5) >= 6:
            self.sig.up, self.sig.down = calc_trend(bars5[-400:])
        if bars30 and len(bars30) >= 26:
            self.sig.filt = calc_filter(bars30[-400:])

    def flip_signal(self, position):
        """If we hold `position` (>0 long, <0 short) and the UNCONFIRMED brick
        colour has flipped to the opposite side, signal a reversal to that side.
        'LONG' | 'SHORT' | None."""
        if position > 0 and self._prov_dir == "red":
            return "SHORT"
        if position < 0 and self._prov_dir == "green":
            return "LONG"
        return None

    def decide(self, price, ts_ny):
        return self.sig.decide(price, ts_ny)

    def ready(self):
        return (self.sig.up is not None and self.sig.down is not None
                and self.sig.filt is not None)

    def status(self):
        return (f"renko brick={self.brick_size}pt | bricks={len(self.bricks)} "
                f"htf={len(self.htf)}(grp {self.group}) | ready={self.ready()}")
