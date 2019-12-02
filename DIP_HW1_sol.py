import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import misc
from scipy.signal import convolve2d

import scipy.misc
import math


class Trajectories:
    """
    This class represents a trajectory class read from a .mat file with trajectories
    of the photographer hand
    """

    DEFAULT_KERNEL_SIZE = 11
    IMAGE_SIZE = 256
    IMAGE_CENTER = (IMAGE_SIZE-1) // 2

    def __init__(self, traj_file_path: str):
        """
        Constructor
        :param traj_file_path: path to the trajectories.mat file
        """
        mat = scipy.io.loadmat(traj_file_path)
        self.X = mat['X']
        self.Y = mat['Y']

    def get_traj_for_frame(self, frame_num: int)->(np.ndarray, np.ndarray):
        """
        Get the trajectories X and Y movements for a given frame
        :param frame_num:
        """
        return self.X[frame_num], self.Y[frame_num]

    def plot_traj(self, frame_num: int):
        """
        Plots the trajectory for a given frame
        :param frame_num: frame number to be plotted
        """
        X , Y = self.get_traj_for_frame(frame_num)
        plt.plot(X, Y)
        plt.xlabel('X axis movement')
        plt.ylabel('X axis movement')
        plt.title("Camera trajectories for frame = {}".format(frame_num))
        plt.show()

    def plot_all_traj(self):
        """
        Plots all trajectories
        """
        for frame_num in range(len(self.X)):
            self.plot_traj(frame_num)

    def generate_kernel_for_frame_num(self, frame_num: int, plot: bool=False, kernel_size: int = DEFAULT_KERNEL_SIZE,)-> np.ndarray:
        """
        Generates a kernel, using the trajectories of the given frame number
        :param frame_num: frame number whose trajectories will be used to
                          generate the kernel
        :param plot: Whether to plot the PSF(Point Spread Function)
        :param kernel_size: The wanted kernel size to be used
        """
        kernel = np.zeros((kernel_size, kernel_size), int)
        x, y = self.get_traj_for_frame(frame_num)
        kernel_center = (kernel_size-1) // 2

        def apply_motion(ker, x_motion, y_motion, image_size):
            # Math way of describing a motion filtter
            x_loc = int(round(kernel_center + ((x_motion * (image_size/kernel_size)) / kernel_size) ))
            y_loc = int(round(kernel_center + ((y_motion * (image_size/kernel_size)) / kernel_size) ))

            def bound(low, high, value):
                return max(low, min(high, value))

            x_loc = bound(0, kernel_size - 1, x_loc)
            y_loc = bound(0, kernel_size - 1, y_loc)
            y_loc = y_loc if y_loc < kernel_size - 1 else kernel_size - 1
            ker[x_loc][y_loc] += 1

        for x_motion, y_motion in zip(x, y):
            apply_motion(kernel, x_motion, y_motion, self.IMAGE_SIZE)
        # For some reason the kernel is calculated with a 90 degree mistake
        # This fix this issue
        kernel = np.rot90(kernel)

        if plot:
            plt.imshow(kernel, cmap='gray')
            plt.show()
        return kernel


class Image:

    WHITE_LEVEL = 255

    def __init__(self, path_to_im_file):
        im = cv2.imread(path_to_im_file, cv2.IMREAD_GRAYSCALE).astype(float)
        im /= self.WHITE_LEVEL
        im = cv2.resize(im, (256, 256,))
        self.im = im
        self.__filter = None

    @property
    def filter(self):
        return self.__filter

    @filter.setter
    def filter(self, kernel):
        self.__filter = kernel

    def show_orig(self):
        plt.imshow(self.im, cmap='gray')
        plt.show()

    def show_filtered(self):
        filtered_im = convolve2d(self.im, self.__filter, boundary='symm', mode='same')
        plt.imshow(filtered_im, cmap='gray')
        plt.show()


def main():
    TRAJ_NUM = 5
    trajectories = Trajectories('100_motion_paths.mat')
    trajectories.plot_traj(TRAJ_NUM)
    kernel = trajectories.generate_kernel_for_frame_num(TRAJ_NUM, True)
    image = Image("DIPSourceHW1.jpg")
    image.filter = kernel
    image.show_filtered()


if __name__ == '__main__':
    main()
