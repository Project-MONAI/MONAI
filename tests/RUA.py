# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from skimage.measure import find_contours



def _coords_to_pixel(current, previous):
    """For contour coordinate generation.
    Given the previous and current border positions,
    returns the int pixel that marks the extremity
    of the segmented pixels
    """

    p_delta = (current[0] - previous[0], current[1] - previous[1])

    if p_delta == (0.0, 1.0):
        row = int(current[0] + 0.5)
        col = int(current[1])
    elif p_delta == (0.0, -1.0):
        row = int(current[0])
        col = int(current[1])
    elif p_delta == (0.5, 0.5):
        row = int(current[0] + 0.5)
        col = int(current[1])
    elif p_delta == (0.5, -0.5):
        row = int(current[0])
        col = int(current[1])
    elif p_delta == (1.0, 0.0):
        row = int(current[0] + 0.5)
        col = int(current[1])
    elif p_delta == (-1, 0.0):
        row = int(current[0])
        col = int(current[1] + 0.5)
    elif p_delta == (-0.5, 0.5):
        row = int(current[0] + 0.5)
        col = int(current[1] + 0.5)
    elif p_delta == (-0.5, -0.5):
        row = int(current[0])
        col = int(current[1] + 0.5)

    return row, col


def _dist_from_topleft(sequence, h, w):
    """For contour coordinate generation.
    Each sequence of cordinates describes a boundary between
    foreground and background starting and ending at two sides
    of the bounding box. To order the sequences correctly,
    we compute the distannce from the topleft of the bounding box
    around the perimeter in a clock-wise direction.
     Args:
         sequence: list of border points
         h: height of the bounding box
         w: width of the bounding box
     Returns:
         distance: the distance round the perimeter of the bounding
             box from the top-left origin
    """

    first = sequence[0]
    if first[0] == 0:
        distance = first[1]
    elif first[1] == w - 1:
        distance = w + first[0]
    elif first[0] == h - 1:
        distance = 2 * w + h - first[1]
    else:
        distance = 2 * (w + h) - first[0]

    return distance


def _sp_contours_to_cv(contours, h, w):
    """Converts Scipy-style contours to a more succinct version
       which only includes the pixels to which lines need to
       be drawn (i.e. not the intervening pixels along each line).
    Args:
        contours: scipy-style clockwise line segments, with line separating foreground/background
        h: Height of bounding box - used to detect direction of line segment
        w: Width of bounding box - used to detect direction of line segment
    Returns:
        pixels: the pixels that need to be joined by straight lines to
                describe the outmost pixels of the foreground similar to
                OpenCV's cv.CHAIN_APPROX_SIMPLE (anti-clockwise)
    """
    pixels = None
    sequences = []
    corners = [False, False, False, False]

    for group in contours:
        sequence = []
        last_added = None
        prev = None
        corner = -1

        for i, coord in enumerate(group):
            if i == 0:
                if coord[0] == 0.0:
                    # originating from the top, so must be heading south east
                    corner = 1
                    pixel = (0, int(coord[1] - 0.5))
                    if pixel[1] == w - 1:
                        corners[1] = True
                    elif pixel[1] == 0.0:
                        corners[0] = True
                elif coord[1] == 0.0:
                    corner = 0
                    # originating from the left, so must be heading north east
                    pixel = (int(coord[0] + 0.5), 0)
                elif coord[0] == h - 1:
                    corner = 3
                    # originating from the bottom, so must be heading north west
                    pixel = (int(coord[0]), int(coord[1] + 0.5))
                    if pixel[1] == w - 1:
                        corners[2] = True
                elif coord[1] == w - 1:
                    corner = 2
                    # originating from the right, so must be heading south west
                    pixel = (int(coord[0] - 0.5), int(coord[1]))

                sequence.append(pixel)
                last_added = pixel
            elif i == len(group) - 1:
                # add this point
                pixel = _coords_to_pixel(coord, prev)
                if pixel != last_added:
                    sequence.append(pixel)
                    last_added = pixel
            elif np.any(coord - prev != group[i + 1] - coord):
                pixel = _coords_to_pixel(coord, prev)
                if pixel != last_added:
                    sequence.append(pixel)
                    last_added = pixel

            # flag whether each corner has been crossed
            if i == len(group) - 1:
                if corner == 0:
                    if coord[0] == 0:
                        corners[corner] = True
                elif corner == 1:
                    if coord[1] == w - 1:
                        corners[corner] = True
                elif corner == 2:
                    if coord[0] == h - 1:
                        corners[corner] = True
                elif corner == 3:
                    if coord[1] == 0.0:
                        corners[corner] = True

            prev = coord

        dist = _dist_from_topleft(sequence, h, w)

        sequences.append({"distance": dist, "sequence": sequence})

    # check whether we need to insert any missing corners
    if corners[0] is False:
        sequences.append({"distance": 0, "sequence": [(0, 0)]})
    if corners[1] is False:
        sequences.append({"distance": w, "sequence": [(0, w - 1)]})
    if corners[2] is False:
        sequences.append({"distance": w + h, "sequence": [(h - 1, w - 1)]})
    if corners[3] is False:
        sequences.append({"distance": 2 * w + h, "sequence": [(h - 1, 0)]})

    # now, join the sequences into a single contour
    # starting at top left and rotating clockwise
    sequences.sort(key=lambda x: x.get("distance"))

    last = (-1, -1)
    for sequence in sequences:
        if sequence["sequence"][0] == last:
            pixels.pop()

        if pixels:
            pixels = [*pixels, *sequence["sequence"]]
        else:
            pixels = sequence["sequence"]

        last = pixels[-1]

    if pixels[0] == last:
        pixels.pop(0)

    if pixels[0] == (0, 0):
        pixels.append(pixels.pop(0))

    pixels = np.array(pixels).astype("int32")
    pixels = np.flip(pixels)

    return pixels


a = np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]])
print(a)
b = find_contours(a, 0.5)
print(b)

pixels = _sp_contours_to_cv(b, 4, 4)
print(pixels)
