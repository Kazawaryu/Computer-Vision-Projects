import numpy as np
from matplotlib import pyplot as plt

from CV.Assign1.code.utils import load_image


def my_imfilter(image, filter):
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    wei_h, wei_w = filter.shape
    im = np.array(image)

    r = np.pad(im[:, :, 0], pad_width=int(wei_h / 2), mode='constant', constant_values=0)
    g = np.pad(im[:, :, 1], pad_width=int(wei_h / 2), mode='constant', constant_values=0)
    b = np.pad(im[:, :, 2], pad_width=int(wei_h / 2), mode='constant', constant_values=0)
    R = conv(r, filter)
    G = conv(g, filter)
    B = conv(b, filter)
    filtered_image = np.stack((R, G, B), axis=2)
    filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image))

    return filtered_image


def conv(img, weight):
    wei_h, wei_w = weight.shape
    img_h, img_w = img.shape
    conv_h = img_h - wei_h + 1
    conv_w = img_w - wei_w + 1
    convimage = np.zeros((conv_h, conv_w))
    for y in range(conv_w):
        for x in range(conv_h):
            sub = img[x:x + wei_h, y:y + wei_w]
            count = sub * weight
            c = sum(sum(count))
            convimage[x, y] = c
    np.clip(convimage, 0.0, 255.0)

    return convimage


def create_hybrid_image(image1, image2, filter):

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    low_frequencies = my_imfilter(image1, filter)

    high_frequencies = image2 - my_imfilter(image2, filter)
    np.clip(high_frequencies,0.0,255.0)
    high_frequencies = (high_frequencies - np.min(high_frequencies)) / (np.max(high_frequencies) - np.min(high_frequencies))
    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = (hybrid_image - np.min(hybrid_image)) / (np.max(hybrid_image) - np.min(hybrid_image))

    return low_frequencies, high_frequencies, hybrid_image


if __name__ == "__main__":
    identity_filter = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    test_image = load_image('../data/cat.bmp')
    identity_image = my_imfilter(test_image, identity_filter)
    # plt.imshow(identity_image*255)
    # plt.show()
    plt.figure();
    plt.imshow((identity_image * 255).astype(np.uint8));
