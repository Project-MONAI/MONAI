from GeometricTransforms import LoadImageAndAnnotations, ResampleImageAndAnnotations, FlipImageAndAnnotations, \
    RotateImageAndAnnotations
from monai.transforms import Compose

from GeometricTransforms import sanity_check

def demo_flywheel():
    sample_data = {'images': './sample_data/Mammo_194_LMLO_N_080839.jpg',
        'labels': './sample_data/Mammo_194_LMLO_N_080839.json'}

    t = LoadImageAndAnnotations(labels=['pent', 'line', 'box'])(sample_data)
    img = sanity_check(t['images'], t['labels'])
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()
    print("Shown original image and annotations")

    transform = Compose([LoadImageAndAnnotations(labels=['pent', 'line', 'box']),
                         RotateImageAndAnnotations(angle=10.0),
                         ResampleImageAndAnnotations(output_size=(1000, 1512)),
                         FlipImageAndAnnotations(flip_axes=0),
                         FlipImageAndAnnotations(flip_axes=1), FlipImageAndAnnotations(flip_axes=0),
                         ])

    img = transform(sample_data)
    image = sanity_check(img['images'], img['labels'])
    plt.imshow(image)
    plt.show()



def main():
    demo_flywheel()

if __name__ == '__main__':
    demo_flywheel()
    print('Done')