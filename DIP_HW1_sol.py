import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy.signal import convolve2d
import scipy.misc
import imageio
from skimage.measure import compare_psnr

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
        self.num_traj = len(self.X)

    def get_traj_for_frame(self, frame_num: int)->(np.ndarray, np.ndarray):
        """
        Get the trajectories X and Y movements for a given frame
        :param frame_num:
        """
        return self.X[frame_num], self.Y[frame_num]

    def plot_traj(self, frame_num: int, show=True, save=False):
        """
        Plots the trajectory for a given frame
        :param frame_num: frame number to be plotted
        """
        if show:
            X , Y = self.get_traj_for_frame(frame_num)
            plt.plot(X, Y)
            plt.xlabel('X axis movement')
            plt.ylabel('X axis movement')
            plt.title("Camera trajectories for frame = {}".format(frame_num))
            plt.show()
        if save:
            X , Y = self.get_traj_for_frame(frame_num)
            plt.plot(X, Y)
            plt.xlabel('X axis movement')
            plt.ylabel('X axis movement')
            plt.title("Camera trajectories for frame = {}".format(frame_num))
            plt.savefig("Camera_trajectories_{}".format(frame_num))
            plt.close()

    def plot_all_traj(self, show=True, save=False):
        """
        Plots all trajectories
        """
        if show:
            for frame_num in range(len(self.X)):
                self.plot_traj(frame_num, show=True, save=False)
        if save:
            for frame_num in range(len(self.X)):
                self.plot_traj(frame_num, show=False, save=True)

    def generate_kernel_for_frame_num(self, frame_num: int, show: bool=False, kernel_size: int = DEFAULT_KERNEL_SIZE,save=False)-> np.ndarray:
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

        if show:
            plt.imshow(kernel, cmap='gray')
            plt.show()
        if save:
            imageio.imwrite("PSF_{}.jpg".format(frame_num),
                            scipy.misc.bytescale(kernel))
        return kernel

    def show_save_psf(self, show=True, save=False):
        kernel_list = []
        for i in range(self.num_traj):
            kernel_list.append(self.generate_kernel_for_frame_num(i))
            if show:
                plt.imshow(kernel_list[i], cmap='gray')
                plt.show()
            if save:
                imageio.imwrite("PSF_{}.jpg".format(i), scipy.misc.bytescale(kernel_list[i]))


class BlurredImage:
    """
    This class represents a blurred image
    """

    WHITE_LEVEL = 255

    def __init__(self, path_to_im_file, kernel):
        """
        :param path_to_im_file: the image file path
        :param kernel: the kernel to filter with
        """
        im = cv2.imread(path_to_im_file, cv2.IMREAD_GRAYSCALE).astype(float)
        im /= self.WHITE_LEVEL
        im = cv2.resize(im, (256, 256,))
        self.__original_image = im
        self.__blurred_image = convolve2d(self.__original_image, kernel, boundary='symm',
                                 mode='same')

    @property
    def original_image(self):
        return self.__original_image

    @property
    def blurred_image(self):
        return self.__blurred_image


class BlurredImageList:

    def __init__(self, path_to_im_file, path_to_traj_file):
        self.traj = Trajectories(path_to_traj_file)
        self.num_images = self.traj.num_traj
        self.__blurred_images_list = []
        for i in range(self.num_images):
            im = BlurredImage(path_to_im_file,
                              self.traj.generate_kernel_for_frame_num(i))
            self.__blurred_images_list.append(im.blurred_image)

    @property
    def blurred_image_list(self):
        return self.__blurred_images_list

    def show_save_blurred_image(self, index, show=True, save=False):
        if show:
            plt.imshow(self.__blurred_images_list[index], cmap='gray')
            plt.show()
        if save:
            scipy.misc.imsave("blurred_image_{}.jpg".format(index), self.blurred_image_list[index])

    def save_all_blurred_images(self):
        for i in range(self.num_images):
            self.show_save_blurred_image(i, show=False, save=True)


class ImageRestorer:
    """
    This class will restore the image from the blurred images
    """
    def __init__(self, blurred_image_list, p=8):
        """

        :param path_to_im_file: path to image file
        :param path_to_traj_file:  path to trajectories .m file
        :param p: int, the power to apply to the algorithm
        """
        self.blurred_image_list = blurred_image_list
        self.p = p
        self.num_images = len(blurred_image_list)
        self.filters_list = []
        self.__restored_image_list = []
        for i in range(1, self.num_images + 1):
            fft_sum = 0
            w_list = []
            fourier_transform_list = []
            filters_list = []
            weighted_image = 0
            for j in range(i):
                im_fft = np.fft.fftshift(np.fft.fft2(blurred_image_list[j]))
                fourier_transform_list.append(im_fft)
                filter = np.abs(im_fft)**self.p
                filters_list.append(filter)
                fft_sum += filter
            for j in range(i):
                w_list.append(filters_list[j] / fft_sum)
            for j in range(i):
                weighted_image += w_list[j]*fourier_transform_list[j]
            restored_im = np.abs(np.fft.ifft2(np.fft.ifftshift(weighted_image)))
            self.__restored_image_list.append(restored_im)

    @property
    def restored_image_list(self):
        return self.__restored_image_list

    def show_save_restored_image(self, index, show=True, save=False):
        if show:
            plt.imshow(self.__restored_image_list[index], cmap='gray')
            plt.title("Restored_{}".format(index))
            plt.show()
        if save:
            imageio.imwrite("deblurred_image_{}.jpg".format(index), self.__restored_image_list[index])

    def save_all_restored(self):
        for i in range(self.num_images):
            self.show_save_restored_image(i, show=False, save=True)

    def show_all_restored(self):
        for i in range(self.num_images):
            self.show_save_restored_image(i, show=True, save=False)

class PSNR_results:

    def __init__(self,original_image, deblurred_image_list):
        self.__psnr_values = []
        for image in deblurred_image_list:
            psnr = compare_psnr(original_image, image)
            psnr = psnr if psnr > 0 else -1 * psnr
            self.__psnr_values.append(psnr)

    def save_show_psnr(self, show=True, save=False):
        plt.plot((np.linspace(1, len(self.__psnr_values) , len(self.__psnr_values))), self.__psnr_values)
        plt.xlabel('Number of images used for computation')
        plt.ylabel('PSNR Values in dB')
        plt.title("PSNR vs Number of images used for computation")
        if show:
            plt.show()
        if save:
            plt.savefig("PSNR_graph")
            plt.close()


def main():
    traj = Trajectories('100_motion_paths.mat')
    # traj.plot_all_traj(show=False, save=True)
    traj.show_save_psf(show=False, save=True)
    blurred_image_list = BlurredImageList("DIPSourceHW1.jpg", '100_motion_paths.mat')
    blurred_image_list.save_all_blurred_images()
    image_restorer = ImageRestorer(blurred_image_list.blurred_image_list, p=10)
    image_restorer.save_all_restored()
    def load_image(path_to_file):
        im = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE).astype(float)
        im /= 255
        im = cv2.resize(im, (256, 256,))
        return im
    im = load_image("DIPSourceHW1.jpg")
    psnr = PSNR_results(im, image_restorer.restored_image_list)
    psnr.save_show_psnr(show=False, save=True)


if __name__ == '__main__':
    main()
