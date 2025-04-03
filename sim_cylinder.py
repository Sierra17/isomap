from implement import *
from plot import *

set_seed(2201)

# Generate data for a cylinder
n_points = 1000
theta = np.random.uniform(0, 2 * np.pi, n_points)

z = np.random.uniform(-1, 1, n_points)
x = np.cos(theta) + 0.1 * np.random.normal(size=n_points)
y = np.sin(theta) + 0.1 * np.random.normal(size=n_points)

cylinder = np.column_stack((x, y, z))

# Perform ISOMAP
cylinder_embedding = isomap(cylinder, type='k', hp=10, d=2)[2]

# Plot
fig_cyl = scatterplot(cylinder, title='Original Cylinder in 3D')

fig_cyl_embed = scatterplot(cylinder_embedding, title='Embedded Cylinder in 2D')
fig_cyl_embed.update_xaxes(range=(-2, 2), constrain='domain')
fig_cyl_embed.update_yaxes(scaleanchor='x', scaleratio=1)

fig_cyl.show()
fig_cyl_embed.show()

