from implement import *
from plot import *

import time
from torchvision import datasets, transforms

DIGITS = [8]

# Load MNIST via PyTorch
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Format & filter data
mnist = np.array([image.flatten() for image, _ in mnist_train])
labels = np.array([label for _, label in mnist_train])

mask = np.isin(labels, DIGITS)
subset_idx = np.where(mask)[0]

mnist_sub = mnist[mask]
labels_sub = labels[mask]

# # Perform ISOMAP
# start_time = time.time()
# mnist_embed = isomap(mnist_sub, type='k', hp=6, d=2)[2]
# end_time = time.time()
# print(f'ISOMAP Execution time: {end_time-start_time:.4f} seconds')

# # Save for good measure
# np.save('isomap/data/mnist/embedding_digit_8_.npy', mnist_embed)
mnist_embed = np.load('isomap/data/mnist/embedding_digit_8_.npy')

# Select images to annotate
set_seed(2201)
image_idx = np.sort(np.random.choice(subset_idx, size=100, replace=False))
image_paths = []

for i in image_idx:
    image_array = mnist[i].reshape(28, 28)
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    image_path = f'isomap/data/mnist/image_{i}.png'
    image.save(image_path)
    image_paths.append(image_path)

# Plot
fig_mnist_embed = scatterplot(mnist_embed, title='MNIST Embedded')
image_annotations(
    fig=fig_mnist_embed, 
    X=mnist_embed, 
    image_paths=image_paths, 
    image_idx=image_idx, 
    subset=True, 
    subset_idx=subset_idx,
    sizex=2, sizey=2)
fig_mnist_embed.show()



