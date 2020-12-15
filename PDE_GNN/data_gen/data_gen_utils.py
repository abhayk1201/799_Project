import os
import pickle
import numpy as np
import numpy.fft as fft


def pickle_data(data_dict, path="./"):
    if not os.path.isdir(path):
        os.mkdir(path)

    for name, data in data_dict.items():
        with open(path+name+".pkl", "wb") as f:
            pickle.dump(data, f, protocol=4)


def read_pickle(keys, path="./"):
    data_dict = {}
    for key in keys:
        with open(path+key+".pkl", "rb") as f:
            data_dict[key] = pickle.load(f)
    return data_dict


# def generate_IC(x, a=4, b=4):
#     # Heat
#     lm = np.random.randn(4*a*b)
#     gm = np.random.randn(4*a*b)
#     c = np.random.uniform(0, 1)
#     u0 = np.zeros(len(x))

#     for k in range(-a, a+1):
#         for l in range(-b, b+1):
#             u0 += lm[k*2*a+l] * np.cos(1.0 * (k * x[:, 0] + l * x[:, 1]))
#             u0 += gm[k*2*a+l] * np.sin(1.0 * (k * x[:, 0] + l * x[:, 1]))

#     u0 = u0**2
    
#     return u0 / u0.max() + c * 0


def generate_node_periodic_ic(x, N=4):  # gives node periodic u0
    n_k = N
    n_l = N

    u0 = np.zeros(len(x))
    
    b = 1.0

    for k in range(-n_k, n_k+1):
        for l in range(-n_l, n_l+1):
            lm = np.random.randn()
            gm = np.random.randn()

            u0 += lm * np.cos(b * (k * x[:, 0] + l * x[:, 1]))
            u0 += gm * np.sin(b * (k * x[:, 0] + l * x[:, 1]))
    
    u0_max = u0.max()
    u0_min = u0.min()

    return (u0 - u0_min) / (u0_max - u0_min)


def _generate_IC_burgers(x, a=4, b=4):
    lm = np.random.randn(4*a*b)
    gm = np.random.randn(4*a*b)
    c = np.random.uniform(0, 1)
    u0 = np.zeros(len(x))

    for k in range(-a, a+1):
        for l in range(-b, b+1):
            u0 += lm[k*2*a+l] * np.cos(3.14 * (k * x[:, 0] + l * x[:, 1]))
            u0 += gm[k*2*a+l] * np.sin(3.14 * (k * x[:, 0] + l * x[:, 1]))
    
    return 2 * u0 / np.abs(u0).max() + c * 0


def generate_IC_burgers(x, vtdm, a, b):
    eps = 1.0e-14

    vtdm = vtdm.reshape(-1, 2)
    n_nodes = len(x)
    u = np.zeros(2*n_nodes)
    u_x = _generate_IC_burgers(x, a, b)
    u_y = _generate_IC_burgers(x, a, b)
    
    for i, x_i in enumerate(x):
        if x_i[0] < eps or abs(1.0-x_i[0]) < eps:
            u[vtdm[i, 0]] = 0.0
            u[vtdm[i, 1]] = 0.0
        elif x_i[1] < eps or abs(1.0-x_i[1]) < eps:
            u[vtdm[i, 0]] = 0.0
            u[vtdm[i, 1]] = 0.0
        else:
            u[vtdm[i, 0]] = u_x[i]
            u[vtdm[i, 1]] = u_y[i]

    return u
    

def initgen_node_periodic(mesh_size, freq=3):
    dim = len(mesh_size)
    x = np.random.randn(*mesh_size)
    coe = fft.ifftn(x)
    freqs = np.random.randint(freq, 2*freq, size=[dim, ])
    for i in range(dim):
        perm = np.arange(dim, dtype=np.int32)
        perm[i] = 0
        perm[0] = i
        coe = coe.transpose(*perm)
        coe[freqs[i]+1:-freqs[i]] = 0
        coe = coe.transpose(*perm)
    x = fft.fftn(coe)
    assert np.linalg.norm(x.imag) < 1e-8
    x = x.real

    n = x.shape[0]
    x_p = np.empty((n+1, n+1))
    
    x_p[0:n, 0:n] = x
    x_p[0:n, -1] = x[:, 0]
    x_p[-1, 0:n] = x[0, :]
    x_p[-1, -1] = x[0, 0]

    return x_p


def assign_u_burgers(x, u, vtdm):
    u_new = []
    for i in range(x.shape[0]):
        u_vertex = [u[vtdm[i, 0]], u[vtdm[i, 1]]]
        u_new.extend(u_vertex)
    return np.array(u_new)


def generate_time_grid(T, dt, sig):
    """Generates time grid [0, T] with step dt and noise~N(0, sig^2).

    Args:
        T (float): Terminal time.
        dt (float): Time step.
        sig (float): Std of the Gaussian noise.

    Returns:
        ndarray: 1D array with time points.
    """

    t = np.arange(0, T+dt, dt)
    if t[-1] > T and t[-2] != T:
        t[-1] = T
    elif t[-1] > T and t[-2] == T:
        t = t[:-1]
      
    t[1:-1] += np.random.randn(len(t)-2) * sig
    print(t)
    assert all(np.diff(t) > 0)
    return t


def subsample_inds(n, step):
    inds = np.arange(n).astype(np.int8)
    if (n - 1) % step == 0:
        inds_new = inds[::step]
    else:
        inds_new = np.concatenate([inds[::step], [n-1]])
    return inds_new


def undersample_time(t, dt, sig):
    """ Approximates a noisy time grid with step dt and 
    noise~N(0, sig^2) by elements of t.
    
    Args:
        t (ndarray): 1D array with time points.
        dt (float): Time step.
        sig (flaot): Std of the Gaussian noise.
        
    Returns:
        t_new (ndarray): 'Approximation' of the noisy time grid.
        inds (List[int]): Indices of corresponding elements of t_new in t.
    """
    
    assert all(np.diff(t) > 0)

    t_noisy = generate_time_grid(t[-1], dt, sig)
    t_new = np.zeros_like(t_noisy)
    t_new[0] = t_noisy[0]
    t_new[-1] = t_noisy[-1]
    
    ptr = 1
    min_dist = np.inf
    cur_dist = 0.0
    
    inds = np.zeros(len(t_noisy), dtype=np.int64)
    inds[-1] = len(t) - 1

    for i in range(1, len(t_new)-1):
        min_dist = np.inf
        while(1):
            cur_dist = abs(t_noisy[i] - t[ptr])
            if cur_dist <= min_dist:
                min_dist = cur_dist
                ptr += 1
            else:
                t_new[i] = t[ptr-1]
                inds[i] = ptr - 1
                ptr += 1
                break

    return t_new, inds


def subsample_data(dt, time_sig, node_perc, sim_inds, path, save_to):
    """Undersamples time grid and nodes. t is the same for all simulations.
    
    Args:
        dt (float): Required time step in undersampled time grid.
        time_sig (float): Required noise on the undersampled time grid.
        node_perc (float): Value in range (0, 1], defines the perecentage of 
            nodes in the undersampled grid sampled from the full grid.
        sim_inds (List[int]): indices of the simulations to undersample.
        path (str): Path to the full dataset.
        save_to (str): Where to save the udnersampled dataset.
        """

    print("Reading from", path)
    data_dict = read_pickle(['t', 'x', 'u'], path=path)
    print("Data shape")
    for k, v in data_dict.items():
        print(k, v.shape)

    t = data_dict["t"][sim_inds]  # (n_sim, n_times)
    x = data_dict["x"][sim_inds]  # (n_sim, n_nodes, n_space_dim)
    u = data_dict["u"][sim_inds]  # (n_sim, n_times, n_nodes, n_field_dim)

    assert dt >= t[0, 1] - t[0, 0]

    # new_time_inds = subsample_inds(t.shape[1], time_step)
    # t_new = t[:, new_time_inds]

    _, new_time_inds = undersample_time(t[0], dt, time_sig)
    t_new = t[:, new_time_inds]  # use the same time grid for all simulations

    num_nodes = x.shape[1]
    num_new_nodes = int(node_perc*num_nodes)
    x_new = np.empty((x.shape[0], num_new_nodes, x.shape[2]))
    u_new = np.empty((u.shape[0], t_new.shape[1], num_new_nodes, u.shape[3]))

    for i in range(x.shape[0]):
        new_node_inds = np.random.choice(num_nodes, int(node_perc*num_nodes), replace=False)
        x_new[i] = x[i, new_node_inds, :]
        u_new_tmp = u[i, new_time_inds, :, :]
        u_new[i] = u_new_tmp[:, new_node_inds, :]

    data_dict_new = {"t": t_new, "x": x_new, "u": u_new}
    pickle_data(data_dict_new, path=save_to)

    print("New data shape")
    for k, v in data_dict_new.items():
        print(k, v.shape)
    print("Saved to", save_to)
