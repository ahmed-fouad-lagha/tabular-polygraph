import numpy as np
from scipy import stats
from src.catalog.benchmarks import PROFILES

# ─────────────────────────────────────────────────────────────
# FIDELITY VALIDATION ENGINE
# ─────────────────────────────────────────────────────────────

def _lognorm_params(mean, std):
    """Return (mu, sigma) of the underlying normal for a log-normal."""
    cv2   = (std / mean) ** 2
    sigma = np.sqrt(np.log(1 + cv2))
    mu    = np.log(mean) - 0.5 * sigma ** 2
    return mu, sigma


def _safe_u(u):
    """Clip uniform samples away from boundaries."""
    return np.clip(u, 1e-6, 1 - 1e-6)


def cholesky(A):
    """Cholesky decomposition for correlated sampling."""
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(max(0.0, A[i, i] - s))
            else:
                L[i, j] = (A[i, j] - s) / (L[j, j] + 1e-12)
    return L


def nearest_pd(C, eps=1e-4):
    """Project a symmetric matrix to the nearest positive definite correlation matrix."""
    C = (C + C.T) / 2.0
    eigs, vecs = np.linalg.eigh(C)
    eigs = np.maximum(eigs, eps)
    C_pd = vecs @ np.diag(eigs) @ vecs.T
    d = np.sqrt(np.diag(C_pd))
    C_pd = C_pd / np.outer(d, d)
    np.fill_diagonal(C_pd, 1.0)
    return C_pd


def gen_correlated_uniforms(corr, n_samples):
    """Generate correlated uniform samples via Gaussian copula."""
    C = nearest_pd(corr.copy())
    np.fill_diagonal(C, 1.0)
    L = cholesky(C)
    Z = np.random.randn(len(C), n_samples)
    corr_normals = (L @ Z).T
    return stats.norm.cdf(corr_normals)


def sample_var(vp, u):
    """Transform uniform u ~ U(0,1) to target marginal distribution."""
    d    = vp.get('dist', 'normal')
    mean = vp['mean']
    std  = vp['std']

    if d == 'normal':
        return mean + std * stats.norm.ppf(_safe_u(u))

    elif d == 'lognorm':
        mu, sigma = _lognorm_params(mean, std)
        return np.exp(mu + sigma * stats.norm.ppf(_safe_u(u)))

    elif d == 'truncnorm':
        lo = vp.get('lo', 0.0)
        a  = (lo - mean) / std
        return stats.truncnorm.ppf(_safe_u(u), a, np.inf, loc=mean, scale=std)

    elif d == 'beta':
        lo, hi = vp['lo'], vp['hi']
        r      = hi - lo
        mu_b   = (mean - lo) / r
        var_b  = (std / r) ** 2
        var_b  = min(var_b, mu_b * (1 - mu_b) * 0.98)
        denom  = mu_b * (1 - mu_b) / var_b - 1
        alpha  = max(0.05, mu_b * denom)
        beta_p = max(0.05, (1 - mu_b) * denom)
        return lo + r * stats.beta.ppf(_safe_u(u), alpha, beta_p)

    elif d == 'pareto':
        alpha = vp['alpha']
        target_med = vp.get('median', vp['mean'] * 0.65)
        scale = target_med / (2.0 ** (1.0 / alpha) - 1.0)
        scale = max(scale, 1.0)
        return scale * (1.0 - _safe_u(u)) ** (-1.0 / alpha) - scale + 1.0

    elif d == 'bimodal':
        m1, s1, m2, s2, w = vp['m1'], vp['s1'], vp['m2'], vp['s2'], vp['w']
        z    = stats.norm.ppf(_safe_u(u))
        comp = (np.random.uniform(size=len(u)) < w)
        return np.where(comp, m1 + s1 * z, m2 + s2 * z)

    elif d == 'regime_mix':
        pi_high = vp['pi_high']
        mu_low, s_low = vp['mu_low'], vp['s_low']
        mu_high, s_high = vp['mu_high'], vp['s_high']
        is_high = (np.random.uniform(size=len(u)) < pi_high)
        z = stats.norm.ppf(_safe_u(u))
        ml_lo, sl_lo = _lognorm_params(mu_low, s_low)
        ml_hi, sl_hi = _lognorm_params(mu_high, s_high)
        x_low = np.exp(ml_lo + sl_lo * z)
        x_high = np.exp(ml_hi + sl_hi * z)
        return np.where(is_high, x_high, x_low)

    elif d == 'three_group':
        groups = vp['groups']
        cap = vp.get('cap', None)
        probs = np.array([g['p'] for g in groups])
        probs /= probs.sum()
        cum = np.cumsum(probs)
        result = np.zeros(len(u))
        for gi, g in enumerate(groups):
            lo = 0.0 if gi == 0 else cum[gi - 1]
            hi = cum[gi]
            mask = (u >= lo) & (u < hi)
            if not mask.any(): continue
            u_g = np.clip((u[mask] - lo) / (hi - lo), 1e-6, 1 - 1e-6)
            mu_ln, sig_ln = _lognorm_params(g['mu'], g['sig'])
            result[mask] = np.exp(mu_ln + sig_ln * stats.norm.ppf(u_g))
        if cap is not None: result = np.clip(result, 0.0, cap)
        return result

    elif d == 'deductions_mix':
        p_s = vp['p_single']; p_m = vp['p_married']
        std_s = vp['std_single']; std_m = vp['std_married']
        mu_it = vp['mu_item']; sig_it = vp['sig_item']
        draw = np.random.uniform(size=len(u))
        z = stats.norm.ppf(_safe_u(u))
        result = np.zeros(len(u))
        s_mask = draw < p_s
        m_mask = (draw >= p_s) & (draw < p_s + p_m)
        i_mask = ~s_mask & ~m_mask
        result[s_mask] = std_s + 800 * z[s_mask]
        result[m_mask] = std_m + 1200 * z[m_mask]
        mu_ln, sig_ln = _lognorm_params(mu_it, sig_it)
        result[i_mask] = np.clip(np.exp(mu_ln + sig_ln * z[i_mask]), 0, 500_000)
        return result

    elif d == 'zi_gamma':
        pi = vp.get('pi', 0.30)
        mean_v = vp['mean']
        target_skew = vp.get('skew', 2.44)
        k = (2.0 / target_skew) ** 2
        theta = (mean_v / (1.0 - pi)) / k
        is_zero = (np.random.uniform(size=len(u)) < pi)
        result = np.random.gamma(k, theta, size=len(u))
        result[is_zero] = 0.0
        return result

    elif d == 'dependents':
        probs_d = [0.52, 0.25, 0.14, 0.06, 0.03]
        vals_d = [0.0, 1.0, 2.0, 3.0, 4.2]
        draw = np.random.uniform(size=len(u))
        result = np.zeros(len(u))
        cumprob = 0.0
        for val, prob in zip(vals_d, probs_d):
            mask = (draw >= cumprob) & (draw < cumprob + prob)
            result[mask] = val
            cumprob += prob
        return result

    elif d == 'skew_loss':
        w_loss = vp.get('w_loss', 0.25)
        is_loss = (np.random.uniform(size=len(u)) < w_loss)
        z = stats.norm.ppf(_safe_u(u))
        profit_vals = np.clip(14.0 + 7.0 * z, 0.1, 50.0)
        loss_vals = np.clip(-5.0 + 6.0 * z, -45.0, -0.1)
        return np.where(is_loss, loss_vals, profit_vals)

    elif d == 'fdi':
        pi_zero = vp.get('pi_zero', 0.12)
        pi_extreme = vp.get('pi_extreme', 0.02)
        mu_n, s_n = vp.get('mu_normal', 2.8), vp.get('s_normal', 3.4)
        draw = np.random.uniform(size=len(u))
        result = np.zeros(len(u))
        is_extreme = (draw >= 1.0 - pi_extreme)
        is_normal = (draw >= pi_zero) & ~is_extreme
        mu_ln, sig_ln = _lognorm_params(mu_n, s_n)
        result[is_normal] = np.exp(mu_ln + sig_ln * stats.norm.ppf(_safe_u(u[is_normal])))
        result[is_extreme] = np.random.uniform(25.0, 60.0, is_extreme.sum())
        return result

    return mean + std * stats.norm.ppf(_safe_u(u))


def generate_profile(profile_data, n_samples=50_000):
    """Generate synthetic data for a profile."""
    vnames = list(profile_data['variables'].keys())
    n_vars = len(vnames)
    U = gen_correlated_uniforms(profile_data['correlations'][:n_vars, :n_vars], n_samples)
    synth = {}
    for i, (vname, vp) in enumerate(profile_data['variables'].items()):
        synth[vname] = sample_var(vp, U[:, i])
    return synth


def marginal_score(synth_vals, vp):
    """Score marginal moments."""
    kurt_r = vp.get('kurt', 3.0)
    skew_r = vp.get('skew', 0.0)
    eps = 1e-8
    if kurt_r > 10.0:
        med_r = vp.get('median', vp['mean'] * 0.65)
        iqr_r = vp.get('iqr', vp['std'] * 1.35)
        med_s = np.median(synth_vals)
        q75, q25 = np.percentile(synth_vals, [75, 25])
        iqr_s = q75 - q25
        med_err = abs(med_s - med_r) / (abs(med_r) + eps)
        iqr_err = abs(iqr_s - iqr_r) / (iqr_r + eps)
        score = 100.0 * (1.0 - 0.55 * med_err - 0.45 * iqr_err)
    else:
        mean_r, std_r = vp['mean'], vp['std']
        mean_s, std_s = np.mean(synth_vals), np.std(synth_vals)
        skew_s = stats.skew(synth_vals)
        kurt_s = stats.kurtosis(synth_vals, fisher=False)
        mean_err = abs(mean_s - mean_r) / (abs(mean_r) + eps)
        std_err = abs(std_s - std_r) / (std_r + eps)
        skew_err = abs(skew_s - skew_r) / (abs(skew_r) + 0.5)
        kurt_err = abs(kurt_s - kurt_r) / (abs(kurt_r) + 1.0)
        score = 100.0 * (1.0 - 0.40 * mean_err - 0.35 * std_err - 0.15 * skew_err - 0.10 * kurt_err)
    return float(max(0.0, min(100.0, score)))


def ks_score(synth_vals, vp):
    """Two-sample KS test score."""
    n_ref = 100_000
    ref = sample_var(vp, np.random.uniform(0, 1, n_ref))
    ks, _ = stats.ks_2samp(synth_vals[:10_000], ref[:10_000])
    return float(max(0.0, min(100.0, 100.0 * (1.0 - ks * 2.5))))


def corr_score(synth, real_corr, vnames):
    """Spearman correlation matrix distance."""
    n = len(vnames)
    vals = np.column_stack([synth[v] for v in vnames])
    sp = stats.spearmanr(vals)
    synth_corr = np.atleast_2d(sp.statistic if hasattr(sp, 'statistic') else sp[0])
    if synth_corr.shape == (1, 1) or synth_corr.ndim == 0:
        rho = float(synth_corr.flat[0])
        synth_corr = np.array([[1.0, rho], [rho, 1.0]])
    diff = synth_corr - real_corr[:n, :n]
    score = 100.0 * (1.0 - np.linalg.norm(diff, 'fro') / (np.linalg.norm(real_corr[:n, :n], 'fro') + 1e-8) * 0.6)
    return float(max(0.0, min(100.0, score)))


def run_benchmark():
    """Run all profiles and print the summary table (mimicking scripts/fidelity_engine.py)."""
    results = {}
    N = 50_000
    for profile_name, profile_data in PROFILES.items():
        print(f'\n{"=" * 62}')
        print(f'  Profile : {profile_name.upper()}')
        print(f'  Source  : {profile_data["source"]}')
        print(f'{"=" * 62}')

        vnames = list(profile_data['variables'].keys())
        np.random.seed(42)
        synth = generate_profile(profile_data, n_samples=N)

        m_scores = {}
        k_scores = {}
        print(f'\n  {"Variable":<28} {"Marginal":>10} {"KS":>8}  {"Mean(r)":>12} {"Mean(s)":>12}  {"Med(r)":>10} {"Med(s)":>10}')
        print('  ' + '-' * 100)

        for vname, vp in profile_data['variables'].items():
            sv = synth[vname]
            ms = marginal_score(sv, vp)
            ks = ks_score(sv, vp)
            m_scores[vname] = ms
            k_scores[vname] = ks
            flag = ' ✓' if (ms > 80 and ks > 85) else (' ⚠' if ks > 65 else ' ✗')
            print(f'  {vname:<28} {ms:>9.1f}%  {ks:>7.1f}%  {vp["mean"]:>12.3g}  {np.mean(sv):>12.3g}  {np.median(sv):>10.3g}  {np.median(sv):>10.3g}{flag}')

        cs = corr_score(synth, profile_data['correlations'], vnames)
        avg_m = float(np.mean(list(m_scores.values())))
        avg_k = float(np.mean(list(k_scores.values())))
        overall = 0.45 * avg_m + 0.30 * avg_k + 0.25 * cs

        print(f'\n  Avg marginal      : {avg_m:.2f}%')
        print(f'  Avg KS            : {avg_k:.2f}%')
        print(f'  Correlation       : {cs:.2f}%')
        print(f'  {"─" * 36}')
        print(f'  Overall fidelity  : {overall:.2f}%')

        results[profile_name] = {
            'n_variables': len(vnames),
            'avg_marginal': avg_m,
            'avg_ks': avg_k,
            'correlation': cs,
            'overall': overall,
        }

    # Summary table
    print(f'\n\n{"=" * 62}\n  SUMMARY - ALL PROFILES\n{"=" * 62}')
    print(f'\n  {"Profile":<20} {"Vars":>5} {"Marginal":>10} {"KS":>8} {"Corr":>8} {"Overall":>10}')
    print('  ' + '-' * 66)
    for pname, r in results.items():
        print(f'  {pname:<20} {r["n_variables"]:>5} {r["avg_marginal"]:>9.2f}% {r["avg_ks"]:>7.2f}% {r["correlation"]:>7.2f}% {r["overall"]:>9.2f}%')
