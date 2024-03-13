import cv2
import math
import numpy as np
import sklearn
import tensorflow as tf


class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    """

    def __init__(
        self, probability=0.3, sl=0.02, sh=0.4, r1=0.2, mean=[0.4914, 0.4822, 0.4465]
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if np.random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = np.random.randint(0, img.shape[0] - h)
                y1 = np.random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    # img[x1 : x1 + h, y1 : y1 + w, 0] = np.random.uniform(0, 1)
                    # img[x1 : x1 + h, y1 : y1 + w, 1] = np.random.uniform(0, 1)
                    # img[x1 : x1 + h, y1 : y1 + w, 2] = np.random.uniform(0, 1)
                    img[x1 : x1 + h, y1 : y1 + w, 0] = self.mean[0]
                    img[x1 : x1 + h, y1 : y1 + w, 1] = self.mean[1]
                    img[x1 : x1 + h, y1 : y1 + w, 2] = self.mean[2]
                else:
                    img[x1 : x1 + h, y1 : y1 + w, 0] = abs(np.random.randn())
                    # img[x1 : x1 + h, y1 : y1 + w, 0] = self.mean[0]
                return img

        return img


class randomEraserGenerator:
    def __init__(self):
        pass

    def run(self, x, y=None, batch_size=32):
        x, y = sklearn.utils.shuffle(x, y)

        gen = tf.data.Dataset.from_generator(
            self.__dataGenerator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                (None, x.shape[1], x.shape[2], x.shape[3]),
                (None, y.shape[1]),
            ),
            args=[x, y, batch_size],
        )
        return gen

    def __dataGenerator(self, x, y=None, batch_size=32):
        eraser = RandomErasing()
        size = x.shape[0]

        step = 0
        while step < size:
            datas = x[step : step + batch_size].copy()
            labels = y[step : step + batch_size].copy()

            step_batch = 0
            while step_batch < batch_size:
                datas[step_batch] = eraser(datas[step_batch])
                step_batch += 1

            yield (datas, labels)
            step += batch_size


if __name__ == "__main__":
    import read_data

    train_images, _, _, _ = read_data.fromGZ()

    img = train_images[0]
    cv2.imwrite("before.png", img * 255)

    eraser = RandomErasing()
    img = eraser(img)
    cv2.imwrite("after.png", img * 255)
