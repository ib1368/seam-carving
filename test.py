import sys
import cv2
import numpy as np

from content_aware_image_resizing import SeamCarving

def energy_map(input_image):
    blurred_image = cv2.GaussianBlur(input_image, (3, 3), 0, 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    image_dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    image_dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    return cv2.add(np.absolute(image_dx), np.absolute(image_dy))

def usage(program_name):
    print '''Usage: python {} image [--debug] [new_width new_height]

    --debug              Shows the seam to be removed in each step.
    new_width            The width of resized image.
    new_height           The height of resized image.
    '''.format(program_name)

if __name__ == '__main__':
    seam_carving = SeamCarving()
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        usage(sys.argv[0])
        sys.exit(1)
    elif len(sys.argv) == 5:
        input_image = cv2.imread(sys.argv[1])

        if sys.argv[2] != "--debug":
            usage(sys.argv[0])
            sys.exit(1)

        for resized_image, candidate_seam, energy_map in seam_carving.next_seam(input_image, width=int(sys.argv[3]), height=int(sys.argv[4]), energy_function=energy_map):
            cv2.imshow('Candidate Seam To Be Removed', candidate_seam)
            # cv2.imshow('Image Energy Map', energy_map)
            cv2.waitKey(1)

    elif len(sys.argv) == 4:
        input_image = cv2.imread(sys.argv[1])
        resized_image = seam_carving.resize(input_image, width=int(sys.argv[2]), height=int(sys.argv[3]), energy_function=energy_map)

        cv2.imshow('Resized Image', resized_image)
        cv2.waitKey(0)
