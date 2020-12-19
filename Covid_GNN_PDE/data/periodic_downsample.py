import os
import pickle
import numpy as np


def pickle_data(data_dict, path="./"):
    if not os.path.isdir(path):
        os.mkdir(path)

    for name, data in data_dict.items():
        with open(path+name+".pkl", "wb") as f:
            pickle.dump(data, f)


def read_pickle(keys, path="./"):
    data_dict = {}
    for key in keys:
        with open(path+key+".pkl", "rb") as f:
            data_dict[key] = pickle.load(f)
    return data_dict
    

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
    if t[-1] > T:
        t[-1] = T
    t[1:-1] += np.random.randn(len(t)-2) * sig
    assert all(np.diff(t) > 0)
    return t


def linenterp_states(u, t, t_c):
    n_sim, _, n_nodes, state_dim = u.shape
    u_new = np.empty((n_sim, len(t_c), n_nodes, state_dim))

    ind = 0
    
    for i, ti in enumerate(t):
        if ti > t_c[ind]:
            dt = t[i] - t[i-1]
            wb = 1.0 / dt * (t_c[ind] - t[i-1])
            wa = 1.0 - wb
            u_new[:, [ind], :, :] = wa * u[:, [i-1], :, :] + wb * u[:, [i], :, :]
            ind += 1
        elif abs(ti - t_c[ind]) < 1.0e-16:
            u_new[:, [ind], :, :] = u[:, [i], :, :]
            ind += 1
        else:
            continue 

    return u_new


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

    # Assume time points in all simualtions are the same
    t_c = generate_time_grid(t[0][-1], dt, time_sig)
    u_sub_t = linenterp_states(u, t[0], t_c)
    
    t_sub = np.array([t_c]*x.shape[0])
    u_sub = []
    x_sub = []

    print(f"u_sub_t shape {u_sub_t.shape}")

    n_sim = x.shape[0]

    eps = 1.0e-14
    b = 2 * np.pi  # domain size

    for i in range(n_sim):
        bounds = []
        bound_couples = []
        inside = []

        coords = x[i]
        num_nodes = len(coords)

        for node_i in range(num_nodes):
            on_bound = False
            has_neighbr = False

            if abs(coords[node_i, 0]) < eps and abs(coords[node_i, 1]) >= eps and abs(coords[node_i, 1]-b) >= eps:  # left boundary
                on_bound = True
                for node_j in range(num_nodes):
                    if abs(coords[node_j, 0] - b) < eps and abs(coords[node_j, 1] - coords[node_i, 1]) < eps:
                        bound_couples.append([node_i, node_j])
                        has_neighbr = True
            elif abs(coords[node_i, 1]) < eps and abs(coords[node_i, 0]) >= eps and abs(coords[node_i, 0]-b) >= eps:  # bottom boundary
                on_bound = True
                for node_j in range(num_nodes):
                    if abs(coords[node_j, 1] - b) < eps and abs(coords[node_j, 0] - coords[node_i, 0]) < eps:
                        bound_couples.append([node_i, node_j])
                        has_neighbr = True
            elif (abs(coords[node_i, 0])>=eps and abs(coords[node_i, 0]-b)>=eps) and (abs(coords[node_i, 1])>=eps and abs(coords[node_i, 1]-b)>=eps):
                inside.append(node_i)
            else:
                pass
            
            if on_bound and not has_neighbr:
                print(f"Node at {coords[node_i]} has no pair!")
        
        bound_couples = np.array(bound_couples)
        inside = np.array(inside)

        nn_couples = int(len(bound_couples) * (1.0 - node_perc))
        nn_inside = int(len(inside) * (1.0 - node_perc))

        bound_couples = bound_couples[np.random.choice(len(bound_couples), nn_couples, replace=False)]
        bounds = bound_couples.ravel()
        inside = inside[np.random.choice(len(inside), nn_inside, replace=False)]

        nodes_to_remove = np.concatenate((bounds, inside))

        new_node_inds = list(set(range(num_nodes)) - set(nodes_to_remove))
        new_node_inds = np.array(new_node_inds)
        
        x_sub.append(x[i, new_node_inds, :])
        u_sub.append(u_sub_t[i][:, new_node_inds, :])

    data_dict_new = {"t": t_sub, "x": np.array(x_sub), "u": np.array(u_sub)}
    print(t_sub)
    pickle_data(data_dict_new, path=save_to)

    print("New data shape")
    for k, v in data_dict_new.items():
        print(k, v.shape)
    print("Saved to", save_to)


dt = 0.02  # 0.01 convdiff, 0.04 burgers'
time_sig = 0.02 / 6  # 3*std=dt/2 

node_perc = 0.730038948  # convdiff: use 0.75 as full grid, burgers/heat: use 1.0 as full grid, 0.95 for BURGERS 
# 3000 -> 0.730038948
# 1500 -> 0.364167478
# 750 -> 0.181353457

sim_inds = list(range(0, 24))
path = "./full/convdiff_2pi_train/"
save_to = "./subs_paper/convdiff_2pi_n3000_t11_irregular_train/"  # convdiff_2pi_train

subsample_data(dt, time_sig, node_perc, sim_inds, path, save_to)
