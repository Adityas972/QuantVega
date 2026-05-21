import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


def _bs_d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S, K, T, r, sigma):
    S, K, T, sigma = map(np.asarray, [S, K, T, sigma])
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    return bs_call_price(S, K, T, r, sigma) - S + K * np.exp(-r * T)


def bs_greeks(S, K, T, r, sigma):
    S, K, T, sigma = map(float, [S, K, T, sigma])
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega  = S * pdf_d1 * sqrt_T / 100
    theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def _heston_char_fn(phi, S, K, T, r, v0, kappa, theta, sigma_v, rho, j):
    i = complex(0, 1)
    x = np.log(S / K)

    if j == 1:
        u = 0.5
        b = kappa - rho * sigma_v
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    d = np.sqrt((rho * sigma_v * i * phi - b)**2 - sigma_v**2 * (2 * u * i * phi - phi**2))
    g = (b - rho * sigma_v * i * phi + d) / (b - rho * sigma_v * i * phi - d)

    C = (r * i * phi * T
         + (a / sigma_v**2)
         * ((b - rho * sigma_v * i * phi + d) * T
            - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
    D = ((b - rho * sigma_v * i * phi + d) / sigma_v**2
         * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

    return np.exp(C + D * v0 + i * phi * x)


def _heston_integrand(phi, S, K, T, r, v0, kappa, theta, sigma_v, rho, j):
    i = complex(0, 1)
    cf = _heston_char_fn(phi, S, K, T, r, v0, kappa, theta, sigma_v, rho, j)
    return np.real(cf / (i * phi))


def heston_call_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, limit=200, tol=1e-6):
    args = (S, K, T, r, v0, kappa, theta, sigma_v, rho)

    P1, _ = quad(_heston_integrand, 0, limit, args=(*args, 1), limit=200, epsabs=tol, epsrel=tol)
    P2, _ = quad(_heston_integrand, 0, limit, args=(*args, 2), limit=200, epsabs=tol, epsrel=tol)

    P1 = 0.5 + P1 / np.pi
    P2 = 0.5 + P2 / np.pi

    return S * P1 - K * np.exp(-r * T) * P2


def heston_call_price_batch(params_array, **kwargs):
    prices = np.zeros(len(params_array))
    for i, row in enumerate(params_array):
        S, K, T, r, v0, kappa, theta, sigma_v, rho = row
        try:
            prices[i] = heston_call_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, **kwargs)
        except Exception:
            prices[i] = np.nan
    return prices


if __name__ == "__main__":
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma = 0.20

    bs_price = bs_call_price(S, K, T, r, sigma)
    greeks = bs_greeks(S, K, T, r, sigma)
    print(f"BS Call Price : {bs_price:.4f}")
    print(f"BS Greeks     : {greeks}")

    h_price = heston_call_price(S, K, T, r, v0=sigma**2, kappa=2.0, theta=sigma**2, sigma_v=0.30, rho=-0.70)
    print(f"\nHeston Call   : {h_price:.4f}")
    print(f"Diff vs BS    : {abs(h_price - bs_price):.4f}")
