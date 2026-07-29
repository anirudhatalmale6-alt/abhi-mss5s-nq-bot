"""Market Structure Shift (MSS) on N-second candles - ZigZag swing version.

v1 (raw 8-EMA cross count) overtraded massively on 5s candles: price crosses a
fast EMA on every micro-wobble, so "EMA cross == swing" fired ~1300x/day. His
chart shows a HANDFUL of clean structure shifts, so real swing pivots are needed.

This version:
  * ZigZag swing detection with a POINT threshold -> real, spaced swing highs/lows
    (sized like the swings a human marks on the chart), not every EMA touch.
  * MSS: after higher highs, a lower swing high (confirm of them) = bearish shift;
    enter SHORT when price then rolls back UNDER the 8-EMA (his "under the EMA"),
    stop just above the shift high. Mirror for LONG.
  * NO 1-hour trend gate (client 29-Jul, Abhi). confirm/threshold are knobs.

Feed CLOSED candles via update(o,h,l,c); then poll(price).
"""


class MSS5s:
    def __init__(self, ema_len=8, stop_buf_pts=2.0, confirm=1, zz_thresh_pts=10.0):
        self.ema_len   = int(ema_len)
        self.stop_buf  = float(stop_buf_pts)
        self.confirm   = int(confirm)          # successive lower-highs / higher-lows to arm
        self.zz_thresh = float(zz_thresh_pts)  # ZigZag reversal size (points)
        self.ema       = None
        self.warm      = False
        # ZigZag state
        self.zz_dir    = 0            # +1 seeking swing-high, -1 seeking swing-low, 0 unset
        self.zz_ext    = None         # running extreme of the current leg
        # last confirmed swings
        self.last_hi   = None
        self.last_lo   = None
        self.short_n   = 0            # successive lower swing-highs
        self.long_n    = 0            # successive higher swing-lows
        self.armed_dir = None         # "red" / "green" once structure has shifted
        self.armed_ref = None         # the shift swing high (short) / low (long) for the stop
        self.pending   = None

    def mark_warm(self):
        self.warm = True

    def _on_swing_high(self, price):
        if self.last_hi is not None and price < self.last_hi:
            self.short_n += 1
        else:
            self.short_n = 1          # new / higher high -> uptrend intact, restart
            self.armed_dir = None     # a higher high cancels a pending bearish shift
        self.last_hi = price
        if self.short_n > self.confirm and self.warm:
            # more lower highs than 'confirm' consecutive tops => bearish shift armed
            self.armed_dir = "red"
            self.armed_ref = price

    def _on_swing_low(self, price):
        if self.last_lo is not None and price > self.last_lo:
            self.long_n += 1
        else:
            self.long_n = 1
            self.armed_dir = None
        self.last_lo = price
        if self.long_n > self.confirm and self.warm:
            self.armed_dir = "green"
            self.armed_ref = price

    def update(self, o, h, l, c):
        k = 2.0 / (self.ema_len + 1.0)
        self.ema = c if self.ema is None else (c * k + self.ema * (1.0 - k))
        # ---- ZigZag ----
        if self.zz_dir == 0:
            self.zz_dir = 1
            self.zz_ext = h
            return
        if self.zz_dir == 1:                       # tracking a swing HIGH
            if h > self.zz_ext:
                self.zz_ext = h
            elif l <= self.zz_ext - self.zz_thresh:
                self._on_swing_high(self.zz_ext)   # confirm the top
                self.zz_dir = -1
                self.zz_ext = l
        else:                                       # tracking a swing LOW
            if l < self.zz_ext:
                self.zz_ext = l
            elif h >= self.zz_ext + self.zz_thresh:
                self._on_swing_low(self.zz_ext)
                self.zz_dir = 1
                self.zz_ext = h

    def poll(self, price):
        """Enter when a shift is armed AND price has rolled to the signal side of the
        8-EMA (short: below; long: above). Fires once, then disarms."""
        if not self.warm or self.armed_dir is None or self.ema is None:
            return None
        if self.armed_dir == "red" and price < self.ema:
            ref = self.armed_ref
            self.armed_dir = None
            return ("SHORT", ref + self.stop_buf)
        if self.armed_dir == "green" and price > self.ema:
            ref = self.armed_ref
            self.armed_dir = None
            return ("LONG", ref - self.stop_buf)
        return None
