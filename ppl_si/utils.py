import numpy as np
from numpy.linalg import pinv
from mpmath import mp

mp.dps = 500


def construct_X_tilde(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def construct_active_set(coef_hat, X):
    p = X.shape[1]
    coef_active, sign_active, active_set, inactive_set = [], [], [], []
    for i, val in enumerate(coef_hat):
        if val == 0.0:
            inactive_set.append(i)
        else:
            active_set.append(i)
            coef_active.append(val)
            sign_active.append(np.sign(val))

    X_active = X[:, active_set] if active_set else np.zeros((X.shape[0], 0))
    X_inactive = X[:, inactive_set] if inactive_set else np.zeros((X.shape[0], 0))

    coef_active = np.array(coef_active).reshape(-1, 1)
    sign_active = np.array(sign_active).reshape(-1, 1)

    return {
        "coef_active": coef_active, 
        "sign_active": sign_active, 
        "active_set": active_set, 
        "X_active": X_active, 
        "inactive_set": inactive_set, 
        "X_inactive": X_inactive,
    }


def construct_Q(n, nK): 
    nK_ = n - nK
    Q = np.zeros((nK, n))
    Q[:, nK_: ] = np.eye(nK)
    return Q


def construct_P(K, p, n, n_list):
    blocks = []
    for k in range(K-1):
        scale = n_list[k] / n
        blocks.append(scale * np.eye(p))

    blocks.append(np.eye(p))
    return np.hstack(blocks)


def construct_XY_tilde(beta_tilde_list, n_list, XK, YK, K, p):
    Y_tilde = np.concatenate(
        [np.sqrt(n_list[k]) * beta_tilde_list[k] for k in range(K-1)] + [YK]
    )

    X_blocks = []
    for k in range(K - 1):
        row_k = np.hstack(
            [np.sqrt(n_list[k]) * np.eye(p) if i == k else np.zeros((p, p)) for i in range(K - 1)] + [np.sqrt(n_list[k]) * np.eye(p)]
        )
        X_blocks.append(row_k)

    row_K = np.hstack([np.zeros((XK.shape[0], (K - 1) * p)), XK])
    X_blocks.append(row_K)
    X_tilde = np.vstack(X_blocks)

    return X_tilde, Y_tilde


def construct_test_statistic_pretrained(j, XK_M, YK, M):
    idx = M.index(j)
    ej = np.zeros((len(M), 1))
    ej[idx, 0] = 1.0

    inv = pinv(XK_M.T @ XK_M)
    etaj = XK_M @ inv @ ej

    etajTY = (etaj.T @ YK.reshape(-1, 1))[0, 0]

    return etaj, etajTY


def calculate_a_b_pretrained(etaj, YK, Sigma, nK):
    e1 = etaj.T @ Sigma @ etaj
    b = (Sigma @ etaj)/e1

    e2 = np.eye(nK) - b @ etaj.T
    a = e2 @ YK
    return a.reshape(-1, 1), b.reshape(-1, 1)


def calculate_a_b(etaj, Y, Sigma, n):
    e1 = etaj.T @ Sigma @ etaj
    b = (Sigma @ etaj)/e1

    e2 = np.eye(n) - b @ etaj.T
    a = e2 @ Y

    return a.reshape(-1, 1), b.reshape(-1, 1)


def merge_intervals(intervals, tol=1e-4):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or interval[0] - merged[-1][1] > tol:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return merged


def pivot(intervals, etajTy, etaj, Sigma, tn_mu=0):
    if len(intervals) == 0: 
        return None 

    etaj = etaj.ravel()
    stdev = np.sqrt(etaj @ (Sigma @ etaj))

    numerator = mp.mpf('0')
    denominator = mp.mpf('0')

    for (left, right) in intervals:
        cdf_left = mp.ncdf((left - tn_mu)/ stdev)
        cdf_right = mp.ncdf((right - tn_mu)/ stdev)
        piece = cdf_right - cdf_left
        denominator += piece

        if etajTy >= right:
            numerator += piece
        elif left <= etajTy < right:
            numerator += mp.ncdf((etajTy - tn_mu)/ stdev) - cdf_left

    if denominator == 0:
        return None
    return float(numerator/ denominator)


def calculate_TN_p_value(intervals, etaj, etajTY, Sigma, tn_mu=0.0):
    cdf = pivot(intervals, etajTY, etaj, Sigma, tn_mu)
    if cdf is not None:
        return 2.0 * min(cdf, 1.0 - cdf)
    else: 
        return None


def construct_test_statistic(j, XK_tilde_Mstar, Y, Mstar, n, nK):
    idx = Mstar.index(j)
    ej = np.zeros((len(Mstar), 1))
    ej[idx, 0] = 1.0

    inv = pinv(XK_tilde_Mstar.T @ XK_tilde_Mstar)
    etaj_tail = XK_tilde_Mstar @ inv @ ej

    etaj = np.zeros((n, 1))
    etaj[-nK:, 0] = etaj_tail.ravel()

    etajTY = (etaj.T @ Y.reshape(-1, 1))[0, 0]

    return etaj, etajTY
