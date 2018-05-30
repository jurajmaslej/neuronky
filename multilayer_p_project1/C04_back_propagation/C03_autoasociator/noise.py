import numpy as np


img_shape = (64, 60, -1)


def cutout_rows(inputs, r0, r1):
    result = inputs.copy()

    view = result.reshape(img_shape)

    view[:, r0:r1] = 0

    return result


def cutout_cols(inputs, c0, c1):
    result = inputs.copy()

    view = result.reshape(img_shape)

    view[c0:c1] = 0

    return result


def cutout_rect(inputs, r0, r1, c0, c1):
    result = inputs.copy()

    view = result.reshape(img_shape)

    view[c0:c1, r0:r1] = 0

    return result


def noise_gaussian(inputs, amount):
    return inputs + 0 # FIXME


def noise_masking(inputs, amount):
    result = inputs.copy()

    mask = np.random.rand(*inputs.shape) < amount

    result[mask] = 0

    return result


def noise_salt_and_pepper(inputs, amount):
    result = inputs.copy()

    mask = np.random.rand(*inputs.shape) < amount
    value = np.array(np.random.rand(*inputs.shape) > 0.5, dtype=float)

    result[mask] = value[mask]

    return result
