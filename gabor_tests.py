# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.color import lab2rgb
# %%
def gabor(size, sigma, theta, Lambda, psi, gamma):
    """Draw a gabor patch."""
    sigma_x = sigma
    sigma_y = sigma / gamma
    Lambda = torch.tensor(Lambda)
    psi = torch.tensor(psi)
    theta = torch.tensor(theta)

    # Bounding box
    s = torch.linspace(-1, 1, size)

    (x, y) = torch.meshgrid(s, s)

    # Rotation
    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

    gb = torch.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * torch.cos(2 * torch.pi / Lambda * x_theta + psi)
    return gb

# %%
g = gabor(232, .25, torch.pi/4, .25, 0, 1)
g += 1
g /= 2
g *= 75

plt.imshow(g, cmap="gray")

# %%
theta = np.random.rand() * 2 * np.pi

# a and b dimensions range from -128 to 127
a = np.cos(theta) * 128
b = np.sin(theta) * 128
g_lab = g.unsqueeze(-1).repeat(1,1,3)
g_lab[:,:,1] = a
g_lab[:,:,2] = b
g_rgb = lab2rgb(g_lab)

plt.imshow(g_rgb)
# %%
p = 0.25
r = .5

assert 1 >= r >= max(-p/(1-p),-(1-p)/p), "not a valid value for r"

p1 = p**2 + r * p * (1-p)
p2 = p * (1-p)  * (1 - r)
p3 = p * (1-p) * (1 - r)
p4 = (1-p)**2 + r * p * (1-p)

# %%
batch = torch.stack([g_lab,g_lab])

# %%
