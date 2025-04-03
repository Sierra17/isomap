from implement import *
from plot import *

from random import sample
import scipy.io

# Load face dataset
faces = scipy.io.loadmat('isomap/data/faces/face_data.mat')['images'].T

# Perform ISOMAP
face_embed = isomap(faces, type='k', hp=6, d=2)[2]

# Select images to annotate
set_seed(2201)
image_idx = sample(range(faces.shape[0]), 40)
image_paths = []

for i in image_idx:
    image_array = faces[i].reshape(64, 64).T
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    image_path = f"isomap/data/faces/face_{i}.png"
    image.save(image_path)
    image_paths.append(image_path)

# Plot
fig_faces_embed = scatterplot(face_embed, title='Faces Embedded')
image_annotations(fig=fig_faces_embed, X=face_embed, image_paths=image_paths, image_idx=image_idx)
fig_faces_embed.show()

