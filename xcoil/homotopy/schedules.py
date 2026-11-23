"""Homotopy schedules: cap profile and fading regularizer weights.

All weights are in f units (the objective is F = c_f [f_B + R], R = sum of regularizer residual
squares). Residual amplitudes are returned directly (sqrt of the weight) and written so their
t-derivative is finite everywhere on [0, 1], including t = 1 where the fading weights vanish:
sqrt(c (1 - t)^2) is implemented as sqrt(c) (1 - t), never through sqrt at 0.
"""
from __future__ import annotations

import jax.numpy as jnp


def h(t, t_h):
    """Cap profile: 0 until t_h, then linear to 1 at t = 1."""
    return jnp.clip((t - t_h) / (1.0 - t_h), 0.0, 1.0)


def cap(t, eps0, eps_target, t_h):
    """epsilon_i(t) = eps(0) + (eps_i - eps(0)) h(t)."""
    return eps0 + (eps_target - eps0) * h(t, t_h)


def amp_arc(cfg):
    return jnp.sqrt(cfg.w_arc * cfg.f_ref)


def amp_L(t, cfg):
    """sqrt(w_L(t)), w_L = w_L0 f_ref (1 - t)^2."""
    return jnp.sqrt(cfg.w_L0 * cfg.f_ref) * (1.0 - t)


def amp_msc(t, cfg):
    """sqrt(w_msc(t)), w_msc = w_msc_final rho(t) + w_msc0 f_ref (1 - t)^2.

    rho = 1 (msc_final_ramp = "none") or rho = t^2 (msc_final_ramp = "t2"): a large final MSC
    weight makes the top Fourier modes ~1e9 stiffer than the soft shape modes and the t = 0 settle
    cannot converge; ramping it in from zero keeps the t = 0 problem well-conditioned, has zero
    slope at t = 0 (no tangent kink) and equals the full weight at t = 1. Written without sqrt(0) so t-derivatives stay finite."""
    fade = cfg.w_msc0 * cfg.f_ref
    wf = cfg.w_msc_final
    if wf == 0.0:
        return jnp.sqrt(fade) * (1.0 - t)
    if cfg.msc_final_ramp == "t2":
        if fade == 0.0:
            return jnp.sqrt(wf) * t
        return jnp.sqrt(wf * t ** 2 + fade * (1.0 - t) ** 2)
    return jnp.sqrt(wf + fade * (1.0 - t) ** 2)
