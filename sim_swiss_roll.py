from implement import *
from plot import *

from sklearn.datasets import make_swiss_roll

# Generate Swiss Roll data
swiss_roll, t = make_swiss_roll(n_samples=1000, noise=0.1, random_state=2201)

# Perform ISOMAP
swiss_roll_embedding = isomap(swiss_roll, type='k', hp=10, d=2)[2]

# Plot
fig_swiss_roll = scatterplot(swiss_roll, title='Original Swiss Roll')
fig_swiss_roll.update_traces(marker=dict(color=t, colorscale='Phase'))

fig_swiss_roll_embed = scatterplot(swiss_roll_embedding, title='Un-Rolled Swiss Roll')
fig_swiss_roll_embed.update_traces(marker=dict(color=t, colorscale='Phase'))

fig_swiss_roll.show()
fig_swiss_roll_embed.show()

