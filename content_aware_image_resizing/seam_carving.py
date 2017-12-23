import cv2
import numpy as np


class SeamCarving(object):

    def calculate_energy_map(self, input_image):
        """Calculates energy map for the input image.

        Calculates energy map of the input image by sobel edge detection algorithm.
        Before applying sobel filter, the image is blurred to decrease effect of
        small intensity variations in calculating sobel edges.

        Args:
            input_image: The (multi-channel or gray-scale) image on which, energy
            map should get calculated.

        Returns:
            A matrix showing energy map of the input image using sobel filter.
        """
        blurred_image = cv2.GaussianBlur(input_image, (3, 3), 0, 0)
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        image_dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3,
                             scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        image_dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3,
                             scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        return cv2.add(np.absolute(image_dx), np.absolute(image_dy))

    def calculate_vertical_cumulative_energy_map(self, energy_map):
        """Calculates vertical cumulative energy map based on input energy map.

        Calculates vertical cumulative energy map of the input energy map. It
        is needed by the used dynamic programming algorithm to find the
        vertical seam path with minimum energy. Cumulative energy for each cell
        is calculated by considering one of the three neighbors in the last row
        with minimum cumulative energy.

        Args:
            energy_map: The energy map calculated with user defined energy
            function.

        Returns:
            A matrix which represents cumulative energy map for finding the
            vertical seam with minimum energy.
        """
        height, width = energy_map.shape[:2]
        cumulative_energy_map = np.zeros((height, width))

        for i in xrange(1, height):
            for j in xrange(width):
                left_cumulative_energy = cumulative_energy_map[i - 1, j - 1] \
                                         if j - 1 >= 0 else 1e6
                middle_cumulative_energy = cumulative_energy_map[i - 1, j]
                right_cumulative_energy = cumulative_energy_map[i - 1, j + 1] \
                                          if j + 1 < width else 1e6

                cumulative_energy_map[i, j] = energy_map[i, j] \
                + min(left_cumulative_energy, middle_cumulative_energy, \
                right_cumulative_energy)

        return cumulative_energy_map

    def calculate_horizontal_cumulative_energy_map(self, energy_map):
        """Calculates horizontal cumulative energy map based on input energy map.

        Calculates horizontal cumulative energy map of the input energy map. It
        is needed by the used dynamic programming algorithm to find the
        horizontal seam path with minimum energy. Cumulative energy for each
        cell is calculated by considering one of the three neighbors in the
        last column with minimum cumulative energy.

        Args:
            energy_map: The energy map calculated with user defined energy
            function.

        Returns:
            A matrix which represents cumulative energy map for finding the
            horizontal seam with minimum energy.
        """
        height, width = energy_map.shape[:2]
        cumulative_energy_map = np.zeros((height, width))

        for j in xrange(1, width):
            for i in xrange(height):
                top_cumulative_energy = cumulative_energy_map[i - 1, j - 1] \
                                        if i - 1 >= 0 else 1e6
                middle_cumulative_energy = cumulative_energy_map[i, j - 1]
                bottom_cumulative_energy = cumulative_energy_map[i + 1, j - 1] \
                                           if i + 1 < height else 1e6

                cumulative_energy_map[i, j] = energy_map[i, j] \
                + min(top_cumulative_energy, middle_cumulative_energy, \
                bottom_cumulative_energy)

        return cumulative_energy_map

    def find_horizontal_seam(self, horizontal_cumulative_energy_map):
        """Finds horizontal seam based on horizontal cumulative energy map.

        Finds the horizontal path starting from the last column choosing the
        minimum cumulative energy and continuing the path using one of current
        position's three neighbors in the previous column containing the
        minimum cumulative energy.

        Args:
            horizontal_cumulative_energy_map: The horizontal cumulative energy
            map.

        Returns:
            A list contaning the horizontal seam with minimum energy. Each
            element of the list is pair of [column, row].
        """
        height, width = horizontal_cumulative_energy_map.shape[:2]

        # Row ID with minimum cumulative energy in the last column.
        previous_row = 0
        seam = []

        for i in xrange(width - 1, -1, -1):
            current_column = horizontal_cumulative_energy_map[:, i]

            # If this is the last column, the minimum value index is used as the
            # start of seam.
            if i == width - 1:
                previous_row = np.argmin(current_column)

            # If this is not the last column, the previous row index of the
            # candidate seam path is used as the base to choose from its
            # three neighbors in the current column.
            else:
                top_row_id = current_column[previous_row - 1] \
                             if previous_row - 1 >= 0 else 1e6
                middle_row_id = current_column[previous_row]
                bottom_row_id = current_column[previous_row + 1] \
                                if previous_row + 1 < height else 1e6

                previous_row = previous_row + np.argmin([top_row_id, \
                               middle_row_id, bottom_row_id]) - 1

            # Saves current seam path cooridinates in seam list as sublist of column and row.
            seam.append([i, previous_row])

        return seam

    def find_vertical_seam(self, vertical_cumulative_energy_map):
        """Finds vertical seam based on vertical cumulative energy map.

        Finds the vertical path starting from the last row choosing the
        minimum cumulative energy and continuing the path using one of current
        position's three neighbors in the previous row containing the
        minimum cumulative energy.

        Args:
            vertical_cumulative_energy_map: The vertical cumulative energy map.

        Returns:
            A list contaning the vertical seam with minimum energy. Each
            element of the list is pair of [column, row].
        """
        height, width = vertical_cumulative_energy_map.shape[:2]
        previous_column = 0
        seam = []

        for i in xrange(height - 1, -1, -1):
            current_row = vertical_cumulative_energy_map[i, :]

            # If this is the last row, the minimum value index is used as the
            # start of seam.
            if i == height - 1:
                previous_column = np.argmin(current_row)

            # If this is not the last row, the previous column index of the
            # candidate seam path is used as the base to choose from its
            # three neighbors in the current row.
            else:
                left_column_id = current_row[previous_column - 1] \
                                 if previous_column - 1 >= 0 else 1e6
                middle_column_id = current_row[previous_column]
                right_column_id = current_row[previous_column + 1] \
                                  if previous_column + 1 < width else 1e6

                previous_column = previous_column + np.argmin([left_column_id, \
                                  middle_column_id, right_column_id]) - 1

            # Saves current seam path cooridinates in seam list as sublist of column and row.
            seam.append([previous_column, i])

        return seam

    def draw_seam(self, input_image, seam):
        """Draws found seam on the image

        Draws the input seam on the input image.

        Args:
            input_image: The input image the seam should be drawn on.
            seam: The seam to be drawn.

        Returns:
            An image that the seam is drawn on.
        """
        cv2.polylines(input_image, np.int32([np.asarray(seam)]), False, (0, 255, 0))

        return input_image

    def remove_horizontal_seam(self, input_image, seam):
        """Removes input seam from the image

        Removes the input seam from the input image. The input seam should
        contain horizontal seam's coordinates.

        Args:
            input_image: The input image the seam should be removed from.
            seam: The horizontal seam to be removed.

        Returns:
            An image containing the input image without the seam provided.
        """
        height, width, bands = input_image.shape
        removed = np.zeros((height - 1, width, bands), np.uint8)

        for x, y in reversed(seam):
            removed[0:y, x] = input_image[0:y, x]
            removed[y:height - 1, x] = input_image[y + 1:height, x]

        return removed

    def remove_vertical_seam(self, input_image, seam):
        """Removes input seam from the image

        Removes the input seam from the input image. The input seam should
        contain vertical seam's coordinates.

        Args:
            input_image: The input image the seam should be removed from.
            seam: The vertical seam to be removed.

        Returns:
            An image containing the input image without the seam provided.
        """
        height, width, bands = input_image.shape
        removed = np.zeros((height, width - 1, bands), np.uint8)

        for x, y in reversed(seam):
            removed[y, 0:x] = input_image[y, 0:x]
            removed[y, x:width - 1] = input_image[y, x + 1:width]

        return removed

    def next_seam(self, input_image, width=None, height=None, energy_function=None):
        """Removes one seam per execution

        Removes one seam with minimum energy per execution until the width and
        height of the image equals the provided width and height in the input
        of function. This is a good generator for debugging purpose.

        Args:
            input_image: The input image to be resized
            width: The width that input_image should be resized to
            height: The height that input_image should be resized to
            energy_function: This is the energy function that calculates the
            energy map of the input_image. There is a default energy function
            if it is not provided as input.

        Yields:
            Image after a seam with minimum energy is removed.
            Image with the seam with minimum energy is marked.
            Energy map matrix of the input image.
        """
        result = input_image

        input_image_height, input_image_width = input_image.shape[:2]

        if width is None:
            width = input_image_width

        if height is None:
            height = input_image_height

        change_in_height = input_image_height - height if input_image_height - height > 0 else 0
        change_in_width = input_image_width - width if input_image_width - width > 0 else 0

        for i in xrange(change_in_height):
            if energy_function is None:
                energy_map = self.calculate_energy_map(result)
            else:
                energy_map = energy_function(result)
            cumulative_energy_map = self.calculate_horizontal_cumulative_energy_map(energy_map)
            seam = self.find_horizontal_seam(cumulative_energy_map)
            image_with_candidate_seam = self.draw_seam(result, seam)
            result = self.remove_horizontal_seam(result, seam)
            yield result, image_with_candidate_seam, energy_map

        for i in xrange(change_in_width):
            if energy_function is None:
                energy_map = self.calculate_energy_map(result)
            else:
                energy_map = energy_function(result)
            cumulative_energy_map = self.calculate_vertical_cumulative_energy_map(energy_map)
            seam = self.find_vertical_seam(cumulative_energy_map)
            image_with_candidate_seam = self.draw_seam(result, seam)
            result = self.remove_vertical_seam(result, seam)
            yield result, image_with_candidate_seam, energy_map

    def resize(self, input_image, width=None, height=None, energy_function=None):
        """Resizes the image to the provided width and height.

        Resizes the image to the provided width and height with seam carving
        algorithm.

        Args:
            input_image: The input image to be resized
            width: The width that input_image should be resized to
            height: The height that input_image should be resized to
            energy_function: This is the energy function that calculates the
            energy map of the input_image. There is a default energy function
            if it is not provided as input.

        Returns:
            Image resized to the provided width and height.
        """
        result = input_image

        input_image_height, input_image_width = input_image.shape[:2]

        if width is None:
            width = input_image_width

        if height is None:
            height = input_image_height

        change_in_height = input_image_height - height if input_image_height - height > 0 else 0
        change_in_width = input_image_width - width if input_image_width - width > 0 else 0

        for i in xrange(change_in_height):
            if energy_function is None:
                energy_map = self.calculate_energy_map(result)
            else:
                energy_map = energy_function(result)
            cumulative_energy_map = self.calculate_horizontal_cumulative_energy_map(energy_map)
            seam = self.find_horizontal_seam(cumulative_energy_map)
            result = self.remove_horizontal_seam(result, seam)

        for i in xrange(change_in_width):
            if energy_function is None:
                energy_map = self.calculate_energy_map(result)
            else:
                energy_map = energy_function(result)
            cumulative_energy_map = self.calculate_vertical_cumulative_energy_map(energy_map)
            seam = self.find_vertical_seam(cumulative_energy_map)
            result = self.remove_vertical_seam(result, seam)

        return result

