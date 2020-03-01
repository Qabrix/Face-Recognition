from cv2 import warpAffine
from math import radians, cos, sin
from numpy import random, array, fliplr, copy


class ImageAugmentator:
    def __init__(
        self,
        flip=False,
        rotation=0,
        shear=0,
        translate=0,
        scale_min=1,
        scale_max=1,
        target_size=(480, 640),
    ):
        self.target_size = target_size
        self.flip = flip
        self.rad_rotation = radians(rotation)
        self.translate = translate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shear = shear
        self.transformation_matrix = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def augment_image(self, image, point):
        h, w = self.target_size
        image, face_cords = self.random_flip(image, point)

        image = warpAffine(image, self.transformation_matrix, (w, h))

        face_coordinates: array = array([face_cords[0], face_cords[1], 1])
        face_coordinates = self.transformation_matrix @ face_coordinates

        return copy(image), face_coordinates

    def preset_transformation_matrix(self):
        h, w = self.target_size

        self.transformation_matrix = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.translate_matrix(-w / 2, -h / 2)
        self.random_shear_matrix()
        self.random_rotate_matrix()
        self.random_scale_matrix()
        self.random_translate_matrix()
        self.translate_matrix(w / 2, h / 2)
        self.transformation_matrix = array(
            self.transformation_matrix.flatten()[:6].reshape(2, 3)
        )

    def random_flip(self, image, face_coordinates):
        if self.flip == False:
            return image, face_coordinates
        if random.random() > 0.5:
            return fliplr(image), self.flip_point(face_coordinates)
        else:
            return image, face_coordinates

    def translate_matrix(self, move_x=0, move_y=0):
        self.transformation_matrix = (
            array([[1, 0, move_x], [0, 1, move_y], [0, 0, 1]])
            @ self.transformation_matrix
        )

    def random_translate_matrix(self):
        move_x = (
            random.uniform(-self.translate, self.translate) * self.target_size[0]
        )
        move_y = (
            random.uniform(-self.translate, self.translate) * self.target_size[1]
        )

        self.transformation_matrix = (
            array([[1, 0, move_x], [0, 1, move_y], [0, 0, 1]])
            @ self.transformation_matrix
        )

    def random_scale_matrix(self):
        scale_x = random.uniform(self.scale_min, self.scale_max)
        scale_y = random.uniform(self.scale_min, self.scale_max)

        self.transformation_matrix = (
            array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
            @ self.transformation_matrix
        )

    def random_shear_matrix(self):
        shear = random.uniform(-self.shear, self.shear)

        self.transformation_matrix = (
            array([[1, shear, 0], [0, 1, 0], [0, 0, 1]]) @ self.transformation_matrix
        )

    def flip_point(self, point):
        w = self.target_size[1]
        y = point[1]
        x = 2 * (w / 2 - point[0]) + point[0]
        return (x, y)

    def random_rotate_matrix(self):
        rotation = random.uniform(-self.rad_rotation, self.rad_rotation)
        cosinus = cos(rotation)
        sinus = sin(rotation)

        self.transformation_matrix = (
            array([[cosinus, sinus, 0], [-sinus, cosinus, 0], [0, 0, 1]])
            @ self.transformation_matrix
        )
