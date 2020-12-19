import os
#from dolfin import *
from mshr import *
import numpy as np
from data_gen_utils import generate_node_periodic_ic, generate_time_grid, pickle_data

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_grid(coords, path):
    x = coords[:, 0]
    y = coords[:, 1]
    triang = mtri.Triangulation(x, y)
    plt.triplot(triang, 'ko-', linewidth=0.1, ms=0.5)
    plt.savefig(path)


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


def generate_tri_mesh_cutout(a=0.0, b=1.0, n=3):
    vertices = []

    vertices.extend(seed_boundary([0., 0.], [0.25, 0.], [1, 0], n//4))  # 1
    vertices.extend(seed_boundary([0.25, 0.], [0.25, 0.5], [0, 1], n//2))  # 2
    vertices.extend(seed_boundary([0.25, 0.5], [0.75, 0.5], [1, 0], n//2))  # 3
    vertices.extend(seed_boundary([0.75, 0.5], [0.75, 0.], [0, -1], n//2))  # 4
    vertices.extend(seed_boundary([0.75, 0.], [1., 0.], [1, 0], n//5))  # 5
    vertices.extend(seed_boundary([1., 0.], [1., 1.], [0, 1], n))  # 6
    vertices.extend(seed_boundary([1., 1.], [0., 1.], [-1, 0], n))  # 7
    vertices.extend(seed_boundary([0., 1.], [0., 0.], [0, -1], n))  # 8

    geometry = Polygon(vertices)
    mesh = generate_mesh(geometry, int(0.75*n))
    return mesh


def boundary(x, on_boundary):
    on_lr = near(x[0], 0.0, DOLFIN_EPS) or near(x[0], 1.0, DOLFIN_EPS)
    on_bt = near(x[1], 0.0, DOLFIN_EPS) or near(x[1], 1.0, DOLFIN_EPS)
    return (on_lr or on_bt) and on_boundary
    # return on_boundary  # for cutout


# Create meshes
n_c = 30  # number of points along x/y direction for mesh
mesh = generate_tri_mesh(0.0, 1.0, n_c)
# mesh = generate_tri_mesh_cutout(0.0, 1.0, n_c)
print("mesh size", len(mesh.coordinates()))
# plot(mesh)
# plt.show()
# Set func. spaces
V = FunctionSpace(mesh, "Lagrange", 1)

# ...
n_sim = 50
T = 0.3  # terminal time
t = generate_time_grid(T, dt=0.0001, sig=0.0)  # time grid 0.0001
dt = Constant(t[1]-t[0])  # time step (changes at each time point)
data_dicts = []
ic_freq = 10
save_every = 1

for i in range(n_sim):
    print("Simulation", i)
    # Generate and set ICs
    u_0_np = generate_node_periodic_ic(mesh.coordinates(), N=ic_freq)
    u_0 = Function(V)
    u_0.vector()[:] = u_0_np[dof_to_vertex_map(V)]

    # Set BCs
    bcs = [DirichletBC(V, u_0, boundary)]

    # ...
    u = TrialFunction(V)
    v = TestFunction(V)
    kappa = Expression("0.2", degree=3)

    u_n = Function(V)
    u_n = interpolate(u_0, V)

    # Forms
    n = FacetNormal(mesh)
    F = u*v*dx + dt*kappa*dot(grad(u), grad(v))*dx - u_n*v*dx - dt*dot(v*grad(u), n)*ds
    a, L = lhs(F), rhs(F)

    # ...
    x = mesh.coordinates()
    u_history = np.zeros((t.shape[0], x.shape[0], 1), dtype=np.float32)

    u = Function(V)

    for j, tj in enumerate(t):
        solve(a == L, u, bcs)

        if j % save_every == 0 or j == len(t) - 1:
            print("t: %.4f" % tj)
            u_history[j] = u_n.compute_vertex_values().reshape(-1, 1)

            # plt.figure(1)
            # plot(mesh)
            # plot(u, norm=mpl.colors.Normalize(vmin=0., vmax=1.), cmap="jet")

            # coords = mesh.coordinates()
            # triang = mtri.Triangulation(coords[:, 0], coords[:, 1])
            # values = u_n.compute_vertex_values()
            # plt.tricontourf(triang, values, norm=mpl.colors.Normalize(vmin=0., vmax=1.))
            # plt.colorbar()

            # plot(u_n)  # CHANGE

            # plot_grid(x, './data_gen/grid.png')
            #plt.tight_layout()
            # plt.savefig('./data_gen/tmp_out/u{:d}.png'.format(j))
            # plt.clf()

            # print(len(x))
            # print(qwe)

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
   "GoogleDrive/714/code/data/full/heat_unit_square_test/")  
pickle_data(full_data_dict, path=path)
