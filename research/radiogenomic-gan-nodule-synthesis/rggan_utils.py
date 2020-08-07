from torch import nn


def weights_init(m):  # G used for G init and D init
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def largest_object(image):  # used in G loss, save eval, and save snapshot imgs
    image = image.astype("uint8")
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = 0
    for i in range(1, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 1
    return img2
