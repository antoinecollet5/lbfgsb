# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Pure-Python Moré-Thuente line search scalar engine.

Faithful port of ``scipy.optimize._dcsrch`` (itself ported from MINPACK-2
Fortran by Averick, Carter and Moré, 1993).  All inputs and outputs are plain
Python ``float`` / ``int`` / ``str``.  No array library is imported anywhere
in this module, making it fully backend-agnostic.

References
----------
[1] Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with
    guaranteed sufficient decrease. ACM TOMS, 20(3), 286-307.
[2] Original MINPACK-2 Fortran: ``dcsrch.f`` / ``dcstep.f``.
[3] SciPy Python port: ``scipy/optimize/_dcsrch.py`` (2023).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from typing_extensions import Literal

Task = Literal["FG", "CONV", "WARN", "ERROR"]

# ---------------------------------------------------------------------------
# Mutable state (replaces isave / dsave Fortran arrays)
# ---------------------------------------------------------------------------


@dataclass
class DcsrchState:
    """Carries state between successive :func:`dcsrch` calls.

    Pass the *same* instance on every call within one line search.
    Reset by creating a new ``DcsrchState()`` for each new line search.
    """

    stage: int = 1
    ginit: float = 0.0
    gtest: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    finit: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    stx: float = 0.0
    sty: float = 0.0
    stmin: float = 0.0
    stmax: float = 0.0
    width: float = 0.0
    width1: float = 0.0
    brackt: bool = False
    initialized: bool = False  # True after first call


# ---------------------------------------------------------------------------
# _dcstep — cubic / quadratic safeguarded interpolation
# ---------------------------------------------------------------------------


def _dcstep(
    stx: float,
    fx: float,
    dx: float,
    sty: float,
    fy: float,
    dy: float,
    stp: float,
    fp: float,
    dp: float,
    brackt: bool,
    stpmin: float,
    stpmax: float,
) -> tuple[float, float, float, float, float, float, float, bool]:
    """One safeguarded interpolation step (port of Fortran ``dcstep``).

    Returns
    -------
    (stx, fx, dx, sty, fy, dy, stp, brackt)
    """
    sgnd = dp * (dx / abs(dx))

    # Case 1: higher function value — bracket and interpolate.
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * math.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = (gamma - dx) + gamma + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + (dx / ((fx - fp) / (stp - stx) + dx)) / 2.0 * (stp - stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True

    # Case 2: lower function, opposite-sign derivative — bracket.
    elif sgnd < 0.0:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * math.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma - dp) + gamma + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        stpf = stpc if abs(stpc - stp) > abs(stpq - stp) else stpq
        brackt = True

    # Case 3: lower function, same sign, |dp| < |dx| — extrapolate.
    elif abs(dp) < abs(dx):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma_sq = (theta / s) ** 2 - (dx / s) * (dp / s)
        gamma = s * math.sqrt(max(0.0, gamma_sq)) if gamma_sq > 0.0 else 0.0
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q if q != 0.0 else 0.0
        if r < 0.0 and gamma != 0.0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if brackt:
            stpf = stpc if abs(stpc - stp) < abs(stpq - stp) else stpq
        else:
            stpf = stpc if abs(stpc - stp) > abs(stpq - stp) else stpq

    # Case 4: lower function, same sign, |dp| >= |dx|.
    else:
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s * math.sqrt(max(0.0, (theta / s) ** 2 - (dy / s) * (dp / s)))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = (gamma - dp) + gamma + dy
            r = p / q
            stpf = stp + r * (sty - stp)
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # --- Update the interval of uncertainty (stx, sty) ---
    if fp > fx:
        sty, fy, dy = stp, fp, dp
    else:
        if sgnd < 0.0:
            sty, fy, dy = stx, fx, dx
        stx, fx, dx = stp, fp, dp

    # Clip and return proposed step.
    stp = min(stpmax, max(stpmin, stpf))
    return stx, fx, dx, sty, fy, dy, stp, brackt


# ---------------------------------------------------------------------------
# dcsrch — main entry point
# ---------------------------------------------------------------------------


def dcsrch(
    stp: float,
    f: float,
    g: float,
    ftol: float,
    gtol: float,
    xtol: float,
    stpmin: float,
    stpmax: float,
    state: DcsrchState,
) -> tuple[float, float, float, Task]:
    """One iteration of the Moré-Thuente line search.

    Faithful port of ``scipy.optimize._dcsrch.DCSRCH._iterate``.

    Parameters
    ----------
    stp : float
        Proposed step length.
    f : float
        Objective value ``phi(stp)``.
    g : float
        Directional derivative ``phi'(stp)``.
    ftol, gtol, xtol : float
        Wolfe / step-width tolerances.
    stpmin, stpmax : float
        Step bounds.
    state : DcsrchState
        Mutable carry-over state (same object for every call in one search).

    Returns
    -------
    (stp, f, g, task)
        ``task`` is one of:
        ``"FG"``    — evaluate ``f`` and ``g`` at the returned ``stp`` then call again.
        ``"CONV"``  — strong Wolfe conditions satisfied.
        ``"WARN"``  — step at bound / interval too narrow; stp is still usable.
        ``"ERROR"`` — invalid inputs.
    """
    XTRAPL, XTRAPU = 1.1, 4.0
    P5, P66 = 0.5, 0.66

    if not state.initialized:
        # --- Validate inputs ---
        if g >= 0.0:
            return stp, f, g, "ERROR"
        if ftol < 0.0 or gtol < 0.0 or xtol < 0.0:
            return stp, f, g, "ERROR"
        if stpmin < 0.0 or stpmax < stpmin:
            return stp, f, g, "ERROR"
        if stp < stpmin or stp > stpmax:
            return stp, f, g, "ERROR"

        state.finit = f
        state.ginit = g
        state.gtest = ftol * g  # negative because g < 0
        state.width = stpmax - stpmin
        state.width1 = state.width / P5

        state.stx = 0.0
        state.fx = f
        state.gx = g
        state.sty = 0.0
        state.fy = f
        state.gy = g
        state.stmin = 0.0
        state.stmax = stp + XTRAPU * stp
        state.initialized = True
        # stage stays 1 — return FG immediately (caller evaluates at stp)
        return stp, f, g, "FG"

    # ---------------------------------------------------------------- update
    ftest = state.finit + stp * state.gtest  # gtest < 0, so ftest < finit for small stp

    # Transition to stage 2 once f <= ftest and g >= 0
    if state.stage == 1 and f <= ftest and g >= 0.0:
        state.stage = 2

    # --- Warnings ---
    task: Task = "FG"
    if state.brackt and (stp <= state.stmin or stp >= state.stmax):
        task = "WARN"
    if state.brackt and state.stmax - state.stmin <= xtol * state.stmax:
        task = "WARN"
    if stp == stpmax and f <= ftest and g <= state.gtest:
        task = "WARN"
    if stp == stpmin and (f > ftest or g >= state.gtest):
        task = "WARN"

    # --- Convergence (strong Wolfe) ---
    if f <= ftest and abs(g) <= gtol * (-state.ginit):
        task = "CONV"

    if task in ("WARN", "CONV"):
        return stp, f, g, task

    # --- Stage-1: modified-function update ---
    if state.stage == 1 and f <= state.fx and f > ftest:
        fm = f - stp * state.gtest
        fxm = state.fx - state.stx * state.gtest
        fym = state.fy - state.sty * state.gtest
        gm = g - state.gtest
        gxm = state.gx - state.gtest
        gym = state.gy - state.gtest

        (state.stx, fxm, gxm, state.sty, fym, gym, stp, state.brackt) = _dcstep(
            state.stx,
            fxm,
            gxm,
            state.sty,
            fym,
            gym,
            stp,
            fm,
            gm,
            state.brackt,
            state.stmin,
            state.stmax,
        )
        state.fx = fxm + state.stx * state.gtest
        state.fy = fym + state.sty * state.gtest
        state.gx = gxm + state.gtest
        state.gy = gym + state.gtest

    else:
        # --- Standard dcstep ---
        (
            state.stx,
            state.fx,
            state.gx,
            state.sty,
            state.fy,
            state.gy,
            stp,
            state.brackt,
        ) = _dcstep(
            state.stx,
            state.fx,
            state.gx,
            state.sty,
            state.fy,
            state.gy,
            stp,
            f,
            g,
            state.brackt,
            state.stmin,
            state.stmax,
        )

    # --- Bisection safeguard ---
    if state.brackt:
        if abs(state.sty - state.stx) >= P66 * state.width1:
            stp = state.stx + P5 * (state.sty - state.stx)
        state.width1 = state.width
        state.width = abs(state.sty - state.stx)

    # --- Update step bounds ---
    if state.brackt:
        state.stmin = min(state.stx, state.sty)
        state.stmax = max(state.stx, state.sty)
    else:
        state.stmin = stp + XTRAPL * (stp - state.stx)
        state.stmax = stp + XTRAPU * (stp - state.stx)

    # Clip to user bounds.
    stp = max(stpmin, min(stpmax, stp))

    return stp, f, g, "FG"
