import numpy as np
from numpy.linalg import pinv


def compute_interval_from_inequalities(A, B, Z=None):
    l, r = -np.inf, np.inf

    for i in range(len(A)):
        if A[i] == 0:
            if B[i] < 0: 
                return np.inf, -np.inf
       
        elif A[i] > 0: 
            r = min(r, B[i] / A[i])
        
        else: 
            l = max(l, B[i] / A[i])

    if l > r:
        print(f"Lỗi l > r ở {Z}")
    
    return l, r


def calculate_c_d(a, b, beta_tilde_list, n_list, K, p):
    c = np.vstack(
        [np.sqrt(n_list[k]) * beta_tilde_list[k].reshape(-1, 1) for k in range(K-1)] + [a]
    )

    d = np.vstack([np.zeros((p * (K - 1), 1)), b])

    return c, d


def compute_Zu(beta_sh_info, a, b, w_tilde, n, Yz):
    psi0 = gamma0 = psi1 = gamma1 = np.empty(0)
    O, Oc, X_tildeO, X_tildeOc, SO = beta_sh_info["active_set"], beta_sh_info["inactive_set"], beta_sh_info["X_active"], beta_sh_info["X_inactive"], beta_sh_info["sign_active"]
    w_tilde_O = w_tilde[O]
    w_tilde_Oc = w_tilde[Oc]

    if len(O) > 0:
        inv = pinv(X_tildeO.T @ X_tildeO)
        X_tildeO_plus = inv @ X_tildeO.T

        psi0 = (-SO * (X_tildeO_plus @ b)).ravel()
        gamma0 = (SO * ((X_tildeO_plus @ a) - (inv @ (w_tilde_O * SO)))).ravel()

    if len(Oc) > 0:
        if len(O) == 0:
            proj = np.eye(n)
            temp2 = 0

        else:
            proj = np.eye(n) - X_tildeO @ X_tildeO_plus
            X_tildeO_plus_T = X_tildeO @ inv
            temp2 = (X_tildeOc.T @ X_tildeO_plus_T) @ (w_tilde_O * SO)

        temp1 = X_tildeOc.T @ proj

        term_b = (temp1 @ b).ravel()
        psi1 = np.concatenate([term_b, -term_b])

        term_a = temp1 @ a
        gamma1 = np.concatenate([(w_tilde_Oc - temp2 - term_a).ravel(), (w_tilde_Oc + temp2 + term_a).ravel()])
    
    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))

    return compute_interval_from_inequalities(psi, gamma, "Zu")


def compute_Zv(beta_indiv_info, a, b, phi_u, iota_u, nK, a_tilde_u, Yz):
    nu0 = kappa0 = nu1 = kappa1 = np.empty(0)
    L, Lc, XK_tildeL, XK_tildeLc, SL = beta_indiv_info["active_set"], beta_indiv_info["inactive_set"], beta_indiv_info["X_active"], beta_indiv_info["X_inactive"], beta_indiv_info["sign_active"]

    a_tilde_L = a_tilde_u[L]
    a_tilde_Lc = a_tilde_u[Lc]
    phi_a_iota = (phi_u @ a) + iota_u
    phi_b = phi_u @ b

    if len(L) > 0:
        inv = pinv(XK_tildeL.T @ XK_tildeL)
        XK_tildeL_plus = inv @ XK_tildeL.T

        nu0 = (-SL * (XK_tildeL_plus @ phi_b)).ravel()
        kappa0 = (SL * ((XK_tildeL_plus @ phi_a_iota) - (inv @ (a_tilde_L * SL)))).ravel()

    if len(Lc) > 0:
        if len(L) == 0:
            proj = np.eye(nK)
            temp2 = 0

        else:
            proj = np.eye(nK) - XK_tildeL @ XK_tildeL_plus
            XK_tildeL_plus_T = XK_tildeL @ inv
            temp2 = (XK_tildeLc.T @ XK_tildeL_plus_T) @ (a_tilde_L * SL)

        temp1 = XK_tildeLc.T @ proj

        term_b = (temp1 @ phi_b).ravel()
        nu1 = np.concatenate([term_b, -term_b])

        term_a = temp1 @ phi_a_iota
        kappa1 = np.concatenate([(a_tilde_Lc - temp2 - term_a).ravel(), (a_tilde_Lc + temp2 + term_a).ravel()])
                
    nu = np.concatenate((nu0, nu1))
    kappa = np.concatenate((kappa0, kappa1))
    
    return compute_interval_from_inequalities(nu, kappa, "Zv")


def compute_Zt(betaK_info, xi_uv, zeta_uv, a, b):
    omega0 = chi0 = omega1 = chi1 = np.empty(0)
    M, Mc, SM = betaK_info["active_set"], betaK_info["inactive_set"], betaK_info["sign_active"]

    xi_a_zeta = (xi_uv @ a) + zeta_uv
    xi_b = xi_uv @ b

    if len(M) > 0:
        Dt_xi_a_zeta = xi_a_zeta[M]
        Dt_xi_b = xi_b[M]

        omega0 = (-SM * Dt_xi_b).ravel()
        chi0 = (SM * Dt_xi_a_zeta).ravel()

    if len(Mc) > 0:
        Dtc_xi_a_zeta = xi_a_zeta[Mc].ravel()
        Dtc_xi_b = xi_b[Mc].ravel()

        omega1 = np.concatenate([Dtc_xi_b, -Dtc_xi_b])
        chi1 = np.concatenate([-Dtc_xi_a_zeta, Dtc_xi_a_zeta])

    omega = np.concatenate((omega0, omega1))
    chi = np.concatenate((chi0, chi1))

    return compute_interval_from_inequalities(omega, chi, "Zt")


def calculate_phi_iota_xi_zeta(beta_sh_info, beta_indiv_info, XK_tilde, p, Q, rho, w_tilde, a_tilde, n, nK):
    phi_u = Q.copy()
    iota_u = np.zeros((nK, 1))
    xi_uv = np.zeros((p + 1, n))
    zeta_uv = np.zeros((p + 1, 1))
    
    O, X_tildeO, SO = beta_sh_info["active_set"], beta_sh_info["X_active"], beta_sh_info["sign_active"]
    L, XK_tildeL, SL = beta_indiv_info["active_set"], beta_indiv_info["X_active"], beta_indiv_info["sign_active"]

    if len(O) > 0:
        w_tilde_O = w_tilde[O]
        Eu = np.eye(p + 1)[:, O] 
        inv_X_tildeO = pinv(X_tildeO.T @ X_tildeO)
        Eu_inv = Eu @ inv_X_tildeO
        XKtilde_Eu_inv = XK_tilde @ Eu_inv

        phi_u -= (1 - rho) * (XKtilde_Eu_inv @ X_tildeO.T)
        iota_u = (1 - rho) * (XKtilde_Eu_inv @ (w_tilde_O * SO))

        xi_uv += (1 - rho) * (Eu_inv @ X_tildeO.T)
        zeta_uv += -(1 - rho) * (Eu_inv @ (w_tilde_O * SO))

    if len(L) > 0:
        a_tilde_L = a_tilde[L]
        Fv = np.eye(p + 1)[:, L]
        inv_XK_tildeL = pinv(XK_tildeL.T @ XK_tildeL)
        Fv_inv = Fv @ inv_XK_tildeL

        xi_uv += Fv_inv @ XK_tildeL.T @ phi_u
        zeta_uv += Fv_inv @ (XK_tildeL.T @ iota_u - a_tilde_L * SL)

    xi_uv = xi_uv[1:, :]
    zeta_uv = zeta_uv[1:, :]

    return phi_u, iota_u, xi_uv, zeta_uv


def compute_Zu_dtf(theta_info, c, d, q_tilde, n, Y_tilde_z):
    psi0 = gamma0 = psi1 = gamma1 = np.empty(0)
    O, Oc, X_tildeO, X_tildeOc, SO = theta_info["active_set"], theta_info["inactive_set"], theta_info["X_active"], theta_info["X_inactive"], theta_info["sign_active"]

    q_tilde_O = q_tilde[O]
    q_tilde_Oc = q_tilde[Oc]

    if len(O) > 0:
        inv = pinv(X_tildeO.T @ X_tildeO)
        X_tildeO_plus = inv @ X_tildeO.T

        psi0 = (-SO * (X_tildeO_plus @ d)).ravel()
        gamma0 = (SO * ((X_tildeO_plus @ c) - n * (inv @ (q_tilde_O * SO)))).ravel()

    proj = np.eye(X_tildeO.shape[0]) 
    
    if len(Oc) > 0:
        if len(O) == 0:
            temp2 = 0

        else:
            proj -= X_tildeO @ X_tildeO_plus
            X_tildeO_plus_T = X_tildeO @ inv
            temp2 = (X_tildeOc.T @ X_tildeO_plus_T) @ (q_tilde_O * SO)

        temp1 = (X_tildeOc.T @ proj) / n

        term_d = (temp1 @ d).ravel()
        psi1 = np.concatenate([term_d, -term_d])
        
        term_c = temp1 @ c
        gamma1 = np.concatenate([(q_tilde_Oc - temp2 - term_c).ravel(), (q_tilde_Oc + temp2 + term_c).ravel()])
    
    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))

    return compute_interval_from_inequalities(psi, gamma, "Zu")


def compute_Zv_dtf(delta_info, phi_u, iota_u, lambda_tilde, nK, z):
    nu0 = kappa0 = nu1 = kappa1 = np.empty(0)
    L, Lc, XK_L, XK_Lc, SL = delta_info["active_set"], delta_info["inactive_set"], delta_info["X_active"], delta_info["X_inactive"], delta_info["sign_active"]

    if len(L) > 0:
        inv = pinv(XK_L.T @ XK_L)
        XK_L_plus = inv @ XK_L.T

        nu0 = (-SL * (XK_L_plus @ phi_u)).ravel()
        kappa0 = (SL * ((XK_L_plus @ iota_u) - ((lambda_tilde * nK) * (inv @ SL)))).ravel()
    
    proj = np.eye(nK)
    if len(Lc) > 0:
        if len(L) == 0:  
            temp2 = 0

        else:
            proj -= XK_L @ XK_L_plus
            XK_L_plus_T = XK_L @ inv
            temp2 = (XK_Lc.T @ XK_L_plus_T) @ SL

        temp1 = (XK_Lc.T @ proj) / (nK * lambda_tilde)

        term_phi = (temp1 @ phi_u).ravel()
        nu1 = np.concatenate([term_phi, -term_phi])

        term_iota = temp1 @ iota_u
        ones = np.ones_like(term_iota)
        kappa1 = np.concatenate([(ones - temp2 - term_iota).ravel(), (ones + temp2 + term_iota).ravel()])
                
    nu = np.concatenate((nu0, nu1))
    kappa = np.concatenate((kappa0, kappa1))
    
    return compute_interval_from_inequalities(nu, kappa, "Zv")


def compute_Zt_dtf(betaK_info, xi_uv, zeta_uv, z):
    omega0 = chi0 = omega1 = chi1 = np.empty(0)
    M, Mc, SM = betaK_info["active_set"], betaK_info["inactive_set"], betaK_info["sign_active"]

    if len(M) > 0:
        Dt_xi = xi_uv[M]
        Dt_zeta = zeta_uv[M]

        omega0 = (-SM * Dt_xi).ravel()
        chi0 = (SM * Dt_zeta).ravel()

    if len(Mc) > 0:
        Dtc_xi = xi_uv[Mc].ravel()
        Dtc_zeta = zeta_uv[Mc].ravel()

        omega1 = np.concatenate([Dtc_xi, -Dtc_xi])
        chi1 = np.concatenate([-Dtc_zeta, Dtc_zeta])

    omega = np.concatenate((omega0, omega1))
    chi = np.concatenate((chi0, chi1))

    return compute_interval_from_inequalities(omega, chi, "Zt")


def calculate_phi_iota_xi_zeta_dtf(theta_info, delta_info, XK, a, b, c, d, P, q_tilde, lambda_tilde, n, nK, K):
    p = XK.shape[1]
    phi_u = b.copy()
    iota_u = a.copy()
    xi_uv = np.zeros((p, 1))
    zeta_uv = np.zeros((p, 1))

    O, X_tildeO, SO = theta_info["active_set"], theta_info["X_active"], theta_info["sign_active"]
    L, XK_L, SL = delta_info["active_set"], delta_info["X_active"], delta_info["sign_active"]

    if len(O) > 0:
        q_tilde_O = q_tilde[O]
        Eu = np.eye(p * K)[:, O] 
        inv_X_tildeO = pinv(X_tildeO.T @ X_tildeO)
        P_Eu_inv = P @ Eu @ inv_X_tildeO
        term_d = P_Eu_inv @ X_tildeO.T @ d
        term_c = P_Eu_inv @ (X_tildeO.T @ c - n * q_tilde_O * SO)

        phi_u -= XK @ term_d
        iota_u -= XK @ term_c
        xi_uv += term_d
        zeta_uv += term_c

    if len(L) > 0:
        Fv = np.eye(p)[:, L]
        inv_XK_L = pinv(XK_L.T @ XK_L)
        Fv_inv = Fv @ inv_XK_L

        xi_uv += Fv_inv @ XK_L.T @ phi_u
        zeta_uv += Fv_inv @ (XK_L.T @ iota_u - nK * lambda_tilde * SL)

    return phi_u, iota_u, xi_uv, zeta_uv
