import os
from dolfin import *
from mshr import *
import numpy as np
from data_gen_utils import generate_node_periodic_ic, generate_time_grid, pickle_data

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def seed_boundary(p0, p1, direction, n):
    """Adds n poins on the line from p0 to p1, first at point p0,
    last at point p1."""
    p0 = np.array(p0)
    p1 = np.array(p1)
    direction = np.array(direction)
    vertices = []
    L = np.linalg.norm(p1 - p0)
    dL = L / (n - 1)
    for i in range(n):
        p = p0 + i * direction * dL
        vertices.append(Point(p[0], p[1]))
    return vertices


def generate_tri_mesh(a=0.0, b=1.0, n=3):
    vertices = []
    vertices.extend(seed_boundary([a, a], [b, a], [1, 0], n))
    vertices.extend(seed_boundary([b, a], [b, b], [0, 1], n))
    vertices.extend(seed_boundary([b, b], [a, b], [-1, 0], n))
    vertices.extend(seed_boundary([a, b], [a, a], [0, -1], n))
    geometry = Polygon(vertices)
    mesh = generate_mesh(geometry, int(0.75*n))
    return mesh


class PeriodicBoundary(SubDomain):
    def __init__(self, L=1.0):
        super().__init__()
        self.L = L

    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and # noqa              
                (not ((near(x[0], 0) and near(x[1], self.L)) or # noqa
                        (near(x[0], self.L) and near(x[1], 0)))) and on_boundary) # noqa

    def map(self, x, y):
        if near(x[0], self.L) and near(x[1], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1] - self.L
        elif near(x[0], self.L):
            y[0] = x[0] - self.L
            y[1] = x[1]
        elif near(x[1], self.L):
            y[0] = x[0]
            y[1] = x[1] - self.L


# Create meshes
n_f = 101  # number of points along x/y direction for fine mesh
n_c = 60  # number of points along x/y direction for coarse mesh
unif_mesh = RectangleMesh(Point(0.0, 0.0), Point(2*np.pi, 2*np.pi), n_f-1, n_f-1)
tri_mesh = generate_tri_mesh(0.0, 2*np.pi, n_c)

# BCs
pbc = PeriodicBoundary(L=2*np.pi)
bcs = []

# Set func. spaces
V_unif = FunctionSpace(unif_mesh, "CG", 1, constrained_domain=pbc)
V_unif_nopbc = FunctionSpace(unif_mesh, "CG", 1)
V_tri = FunctionSpace(tri_mesh, "CG", 1)

# ...
n_sim = 1
T = 0.1  # terminal time 0.6
t = generate_time_grid(T, dt=0.0002, sig=0.0)  # time grid  # dt=0.0002
dt = Constant(t[1]-t[0])  # time step (changes at each time point)
data_dicts = []
ic_freq = 4
save_every = 1

for i in range(n_sim):
    print("Simulation", i)
    # Generate and set ICs
    u_0_np = generate_node_periodic_ic(unif_mesh.coordinates(), ic_freq)

    u_0_unif_nopbc = Function(V_unif_nopbc)
    u_0_unif_nopbc.vector()[:] = u_0_np.ravel()[dof_to_vertex_map(V_unif_nopbc)]
   
    u_0_unif = Function(V_unif)
    u_0_unif.interpolate(u_0_unif_nopbc)

    # ...
    u = Function(V_unif)
    v = TestFunction(V_unif)
    kappa = Expression("0.25", degree=1)
    vel = Constant((5.0, 2.0))

    u_n = Function(V_unif)
    u_n.assign(u_0_unif)

    # Forms
    n = FacetNormal(unif_mesh)
    F = (u-u_n)*v*dx + dt*dot(vel, grad(u_n))*v*dx + dt*kappa*dot(grad(u_n), grad(v))*dx  # - kappa*dot(v*grad(u), n)*ds
    # a, L = lhs(F), rhs(F)

    # ...
    x = tri_mesh.coordinates()
    u_history = np.zeros((t.shape[0], x.shape[0], 1), dtype=np.float32)

    # u = Function(V_unif)
    u_tri = Function(V_tri)

    for j, tj in enumerate(t):
        solve(F == 0, u, bcs)

        if j % save_every == 0 or j == len(t) - 1:
            print("t: %.4f" % tj)
            u_tri.interpolate(u_n)
            u_history[j] = u_tri.compute_vertex_values().reshape(-1, 1)

            plt.figure(1)
            # plot(tri_mesh)
            plot(u_tri, cmap="coolwarm")
            # plot(u_tri, norm=mpl.colors.Normalize(vmin=-2., vmax=2.), cmap="coolwarm")
            
            # coords = tri_mesh.coordinates()
            # triang = mtri.Triangulation(coords[:, 0], coords[:, 1])
            # values = u_tri.compute_vertex_values()
            # plt.tricontourf(triang, values, norm=mpl.colors.Normalize(vmin=0., vmax=1.))
            # plt.colorbar()

            plt.savefig('./data_gen/tmp_out/u{:d}.png'.format(j))
            plt.clf()

        # for x, vv in zip(tri_mesh.coordinates(), u_tri.compute_vertex_values()):
        #     if x[0] == 0 or abs(x[0] - 2*np.pi) < 1.0e-16:
        #         print(x, vv)

        u_n.assign(u)

        if j != len(t)-1:
            dt.assign(t[j+1]-t[j])

    data_dict = {'t': t, 'x': x, 'u': u_history}
    data_dicts.append(data_dict)
    
    # ###################

t = np.zeros((n_sim, *data_dicts[0]["t"].shape))
x = np.zeros((n_sim, *data_dicts[0]["x"].shape))
u = np.zeros((n_sim, *data_dicts[0]["u"].shape))

for i in range(n_sim):
    t[i] = data_dicts[i]["t"]
    x[i] = data_dicts[i]["x"]
    u[i] = data_dicts[i]["u"]

full_data_dict = {"t": t, "x": x, "u": u}

for k, v in full_data_dict.items():
    print(k, v.shape)

path = os.path.join(
    os.environ["HOME"],
    "Desktop/work_files/aalto/paper_exp_comps/data/full/convdiff_2pi_test/")
pickle_data(full_data_dict, path=path)
