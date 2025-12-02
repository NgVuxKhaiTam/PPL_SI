import numpy as np
import os
from joblib import Parallel, delayed
from scipy.linalg import block_diag

from .algorithms import PretrainedLasso, DTransFusion, source_estimator
from .utils import (
    construct_X_tilde, construct_active_set, construct_Q, construct_P,
    construct_XY_tilde, construct_test_statistic_pretrained,
    calculate_a_b_pretrained, merge_intervals, calculate_TN_p_value,
    construct_test_statistic, calculate_a_b
)
from .sub_prob import (
    compute_Zu, compute_Zv, compute_Zt, calculate_phi_iota_xi_zeta,
    compute_Zu_dtf, compute_Zv_dtf, compute_Zt_dtf,
    calculate_phi_iota_xi_zeta_dtf, calculate_c_d
)


def segment_worker(X, XK, X_tilde, XK_tilde, a, b, Mobs, Oobs, Lobs, n, nK, p, Q, lambda_sh, lambda_K, rho, w_tilde, w_inactive, z_start, z_end):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    intervals, oc_intervals = [], []

    z = z_start
    while z < z_end:
        Yz = a + b * z
        Yz = Yz.ravel()
        YKz = Q @ Yz

        beta_sh, beta_indiv, betaK = PretrainedLasso(X, Yz, XK, YKz, lambda_sh, lambda_K, rho, n, nK)
        beta_sh_info = construct_active_set(beta_sh, X_tilde)
        beta_indiv_info = construct_active_set(beta_indiv, XK_tilde)
        betaK_info = construct_active_set(betaK, XK)

        Ou = beta_sh_info["active_set"]
        a_tilde = np.full((p + 1, 1), w_inactive)

        if len(Ou) > 0:
            a_tilde[Ou, 0] = lambda_K
        a_tilde[0, 0] = 0.0

        phi_u, iota_u, xi_uv, zeta_uv = calculate_phi_iota_xi_zeta(beta_sh_info, beta_indiv_info, XK_tilde, p, Q, rho, w_tilde, a_tilde, n, nK)
    
        lu, ru = compute_Zu(beta_sh_info, a, b, w_tilde, n, Yz)
        lv, rv = compute_Zv(beta_indiv_info, a, b, phi_u, iota_u, nK, a_tilde, Yz)
        lt, rt = compute_Zt(betaK_info, xi_uv, zeta_uv, a, b)
       
        right = min(ru, rv, rt)
        left = max(lu, lv, lt)
        if right < left or right < z: 
            print('Error')
            return ([], [])

        Mt = betaK_info["active_set"]
        Lv = beta_indiv_info["active_set"]
        if np.array_equal(Mobs, Mt):
            intervals.append((left, right))

        oc_match = np.array_equal(Mobs, Mt) and np.array_equal(Oobs, Ou) and np.array_equal(Lobs, Lv)
        if oc_match:
            oc_intervals.append((left, right))

        z = right + 1e-5

    return intervals, oc_intervals


def divide_and_conquer(X, XK, a, b, Mobs, Oobs, Lobs, n, nK, p, lambda_sh, lambda_K, rho, z_min, z_max, num_segments=24):
    seg_w = (z_max - z_min) / num_segments
    segments = [(z_min + i * seg_w, z_min + (i + 1) * seg_w) for i in range(num_segments)]
    n_jobs = min(num_segments, os.cpu_count())

    w_tilde = np.concatenate(([0.0], np.full(p, lambda_sh))).reshape(-1, 1)
    
    w_inactive = lambda_K / rho if rho != 0.0 else 1e15
    Q = construct_Q(n, nK)

    X_tilde = construct_X_tilde(X)
    XK_tilde = construct_X_tilde(XK)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        (delayed(segment_worker)(X, XK, X_tilde, XK_tilde, a, b, Mobs, Oobs, Lobs, n, nK, p, Q, lambda_sh, lambda_K, rho, w_tilde, w_inactive, seg[0], seg[1]) for seg in segments)
    )

    intervals, oc_intervals = [], []

    for seg_intervals, seg_oc_intervals in results:
        intervals.extend(seg_intervals)
        oc_intervals.extend(seg_oc_intervals)
    
    intervals = merge_intervals(intervals, tol=1e-4)
    oc_intervals = merge_intervals(oc_intervals, tol=1e-4)

    return intervals, oc_intervals


def PPL_SI(X_list, Y_list, lambda_sh, lambda_K, rho, Sigma_list, z_min=-20, z_max=20):
    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    XK = X_list[-1]
    YK = Y_list[-1]
    n_list = [Xk.shape[0] for Xk in X_list]
    n = sum(n_list)
    nK = n_list[-1]
    p = XK.shape[1]

    beta_sh_hat, beta_indiv_hat, betaK_hat = PretrainedLasso(X, Y, XK, YK, lambda_sh, lambda_K, rho, n, nK)

    Mobs = [i for i in range(p) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None
    
    Mstar = [0] + [i + 1 for i in Mobs]
    XK_tilde = construct_X_tilde(XK)
    XK_tilde_Mstar = XK_tilde[:, Mstar]
    Sigma = block_diag(*Sigma_list)

    Oobs = [i for i in range(p + 1) if beta_sh_hat[i] != 0.0]
    Lobs = [i for i in range(p + 1) if beta_indiv_hat[i] != 0.0]
    
    p_sel_list = []
    
    for j in Mobs:
        etaj, etajTY = construct_test_statistic(j + 1, XK_tilde_Mstar, Y, Mstar, n, nK)
        a, b = calculate_a_b(etaj, Y, Sigma, n)
        intervals, oc_intervals = divide_and_conquer(X, XK, a, b, Mobs, Oobs, Lobs, n, nK, p, lambda_sh, lambda_K, rho, z_min, z_max)
        p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
        p_sel_list.append((j, p_value))
    
    return p_sel_list


def PPL_SI_randj(X_list, Y_list, lambda_sh, lambda_K, rho, Sigma_list, z_min=-20, z_max=20):
    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    XK = X_list[-1]
    YK = Y_list[-1]
    n_list = [Xk.shape[0] for Xk in X_list]
    n = sum(n_list)
    nK = n_list[-1]
    p = XK.shape[1]

    beta_sh_hat, beta_indiv_hat, betaK_hat = PretrainedLasso(X, Y, XK, YK, lambda_sh, lambda_K, rho, n, nK)

    Mobs = [i for i in range(p) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None
    
    Mstar = [0] + [i + 1 for i in Mobs]
    XK_tilde = construct_X_tilde(XK)
    XK_tilde_Mstar = XK_tilde[:, Mstar]
    Sigma = block_diag(*Sigma_list)

    Oobs = [i for i in range(p + 1) if beta_sh_hat[i] != 0.0]
    Lobs = [i for i in range(p + 1) if beta_indiv_hat[i] != 0.0]
    
    j = np.random.choice(Mobs)

    etaj, etajTY = construct_test_statistic(j + 1, XK_tilde_Mstar, Y, Mstar, n, nK)
    a, b = calculate_a_b(etaj, Y, Sigma, n)
    intervals, oc_intervals = divide_and_conquer(X, XK, a, b, Mobs, Oobs, Lobs, n, nK, p, lambda_sh, lambda_K, rho, z_min, z_max)
    p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

    return p_value


def segment_worker_dtf(X_tilde, XK, a, b, c, d, Mobs, Oobs, Lobs, SMobs, SOobs, SLobs, q_tilde, lambda_tilde, P, n, nK, K, z_start, z_end):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    intervals, oc_intervals = [], []

    z = z_start
    while z < z_end:
        Yz = (a + b * z).ravel()
        Y_tilde_z = (c + d * z).ravel()

        theta, delta, betaK = DTransFusion(X_tilde, Y_tilde_z, XK, Yz, q_tilde, lambda_tilde, P)
        theta_info = construct_active_set(theta, X_tilde)
        delta_info = construct_active_set(delta, XK)
        betaK_info = construct_active_set(betaK, XK)

        phi_u, iota_u, xi_uv, zeta_uv = calculate_phi_iota_xi_zeta_dtf(theta_info, delta_info, XK, a, b, c, d, P, q_tilde, lambda_tilde, n, nK, K)
    
        lu, ru = compute_Zu_dtf(theta_info, c, d, q_tilde, n, Y_tilde_z)
        lv, rv = compute_Zv_dtf(delta_info, phi_u, iota_u, lambda_tilde, nK, z)
        lt, rt = compute_Zt_dtf(betaK_info, xi_uv, zeta_uv, z)
       
        right = min(ru, rv, rt)
        left = max(lu, lv, lt)
        if right < left or right < z: 
            print('Error')
            return ([], [])

        Mt = betaK_info["active_set"]
        SMt = betaK_info["sign_active"]
        Ou = theta_info["active_set"]
        SOu = theta_info["sign_active"]
        Lv = delta_info["active_set"]
        SLv = delta_info["sign_active"]
        
        if np.array_equal(Mobs, Mt):
            intervals.append((left, right))

        oc_match = np.array_equal(Mobs, Mt) and np.array_equal(Oobs, Ou) and np.array_equal(Lobs, Lv) and np.array_equal(SMobs, SMt) and np.array_equal(SOobs, SOu) and np.array_equal(SLobs, SLv)
        if oc_match:
            oc_intervals.append((left, right))

        z = right + 1e-5

    return intervals, oc_intervals


def divide_and_conquer_dtf(X_tilde, XK, a, b, c, d, Mobs, Oobs, Lobs, SMobs, SOobs, SLobs, q_tilde, lambda_tilde, P, n, nK, K, z_min, z_max, num_segments=24):
    seg_w = (z_max - z_min) / num_segments
    segments = [(z_min + i * seg_w, z_min + (i + 1) * seg_w) for i in range(num_segments)]
    n_jobs = min(num_segments, os.cpu_count())

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        (delayed(segment_worker_dtf)(X_tilde, XK, a, b, c, d, Mobs, Oobs, Lobs, SMobs, SOobs, SLobs, q_tilde, lambda_tilde, P, n, nK, K, seg[0], seg[1]) for seg in segments)
    )

    intervals, oc_intervals = [], []

    for seg_intervals, seg_oc_intervals in results:
        intervals.extend(seg_intervals)
        oc_intervals.extend(seg_oc_intervals)
    
    intervals = merge_intervals(intervals, tol=1e-4)
    oc_intervals = merge_intervals(oc_intervals, tol=1e-4)

    return intervals, oc_intervals


def DTF_SI(X_list, Y_list, lambda_k_list, lambda_0, lambda_tilde, qk_weights, Sigma_list, z_min=-20, z_max=20):
    K = len(X_list)
    n_list = [Xk.shape[0] for Xk in X_list]
    XK = X_list[-1]
    YK = Y_list[-1]
    nK = n_list[-1]
    p = XK.shape[1]
    n = p * (K - 1) + nK

    beta_tilde_list = []
    for k in range(K - 1):
        beta_tilde_k = source_estimator(X_list[k], Y_list[k], lambda_k_list[k])
        beta_tilde_list.append(beta_tilde_k)

    P = construct_P(K, p, n, n_list)
    X_tilde, Y_tilde = construct_XY_tilde(beta_tilde_list, n_list, XK, YK, K, p)

    q_tilde = np.concatenate(
        [lambda_0 * qk_weights[k] * np.ones(p) for k in range(K - 1)] +
        [lambda_0 * np.ones(p)]
    ).reshape(-1, 1)
    
    theta_hat, delta_hat, betaK_hat = DTransFusion(X_tilde, Y_tilde, XK, YK, q_tilde, lambda_tilde, P)

    Mobs = [i for i in range(p) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None
    
    XK_M = XK[:, Mobs]
    Sigma = Sigma_list[-1]
    SMobs = np.where(betaK_hat[Mobs] > 0, 1, -1).reshape(-1, 1)
    Oobs = [i for i in range(p * K) if theta_hat[i] != 0.0]
    SOobs = np.where(theta_hat[Oobs] > 0, 1, -1).reshape(-1, 1)
    Lobs = [i for i in range(p) if delta_hat[i] != 0.0]
    SLobs = np.where(delta_hat[Lobs] > 0, 1, -1).reshape(-1, 1)

    p_sel_list = []

    for j in Mobs:
        etaj, etajTY = construct_test_statistic_pretrained(j, XK_M, YK, Mobs)
        a, b = calculate_a_b_pretrained(etaj, YK, Sigma, nK)
        c, d = calculate_c_d(a, b, beta_tilde_list, n_list, K, p)
        intervals, oc_intervals = divide_and_conquer_dtf(
            X_tilde, XK, a, b, c, d, Mobs, Oobs, Lobs,
            SMobs, SOobs, SLobs, q_tilde, lambda_tilde, P,
            n, nK, K, z_min, z_max
        )
        p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
        p_sel_list.append((j, p_value))

    return p_sel_list


def DTF_SI_randj(X_list, Y_list, lambda_k_list, lambda_0, lambda_tilde, qk_weights, Sigma_list, z_min=-20, z_max=20):
    K = len(X_list)
    n_list = [Xk.shape[0] for Xk in X_list]
    XK = X_list[-1]
    YK = Y_list[-1]
    nK = n_list[-1]
    p = XK.shape[1]
    n = p * (K - 1) + nK

    beta_tilde_list = []
    for k in range(K - 1):
        beta_tilde_k = source_estimator(X_list[k], Y_list[k], lambda_k_list[k])
        beta_tilde_list.append(beta_tilde_k)

    P = construct_P(K, p, n, n_list)
    X_tilde, Y_tilde = construct_XY_tilde(beta_tilde_list, n_list, XK, YK, K, p)

    q_tilde = np.concatenate(
        [lambda_0 * qk_weights[k] * np.ones(p) for k in range(K - 1)] +
        [lambda_0 * np.ones(p)]
    ).reshape(-1, 1)
    
    theta_hat, delta_hat, betaK_hat = DTransFusion(X_tilde, Y_tilde, XK, YK, q_tilde, lambda_tilde, P)

    Mobs = [i for i in range(p) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None
    
    XK_M = XK[:, Mobs]
    Sigma = Sigma_list[-1]
    SMobs = np.where(betaK_hat[Mobs] > 0, 1, -1).reshape(-1, 1)
    Oobs = [i for i in range(p * K) if theta_hat[i] != 0.0]
    SOobs = np.where(theta_hat[Oobs] > 0, 1, -1).reshape(-1, 1)
    Lobs = [i for i in range(p) if delta_hat[i] != 0.0]
    SLobs = np.where(delta_hat[Lobs] > 0, 1, -1).reshape(-1, 1)

    j = np.random.choice(Mobs)

    etaj, etajTY = construct_test_statistic_pretrained(j, XK_M, YK, Mobs)
    a, b = calculate_a_b_pretrained(etaj, YK, Sigma, nK)
    c, d = calculate_c_d(a, b, beta_tilde_list, n_list, K, p)
    intervals, oc_intervals = divide_and_conquer_dtf(
        X_tilde, XK, a, b, c, d, Mobs, Oobs, Lobs,
        SMobs, SOobs, SLobs, q_tilde, lambda_tilde, P,
        n, nK, K, z_min, z_max
    )
    p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

    return p_value
