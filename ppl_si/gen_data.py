import numpy as np


def generate_synthetic_data(p, num_sh, num_inv, K, n_list, true_beta_sh=0.5, Gamma=0.01, itc=0.5, rho_=None):
    beta_sh = np.zeros(p)
    beta_sh[:num_sh] = true_beta_sh  
    
    available_idx = np.arange(num_sh, p)
    
    beta_list = []
    X_list, Y_list = [], []
    cov = np.eye(p)
    u, v = 1, 1
    if rho_ is not None:
        u = rho_
        v = 1 - rho_

    for k in range(K):
        indiv_idx = np.random.choice(available_idx, size=num_inv, replace=False)
        
        beta_indiv = np.zeros(p)
        beta_indiv[indiv_idx] = np.random.uniform(-Gamma, Gamma, size=num_inv)

        beta_k = u * beta_sh + v * beta_indiv

        beta_list.append(beta_k)

        Xk = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n_list[k])
        true_Yk = Xk @ beta_k 
        noise = np.random.normal(0, 1, n_list[k])
        Yk = true_Yk + noise 
        X_list.append(Xk)
        Y_list.append(Yk)

    Sigma_list = [np.eye(nk) for nk in n_list]
    betaK = beta_list[-1] 

    return X_list, Y_list, betaK, Sigma_list
