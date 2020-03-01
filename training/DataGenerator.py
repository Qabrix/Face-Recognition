from os import walk, path
from matplotlib.pyplot import imread
from tensorflow.keras.utils import Sequence

from cv2 import resize
from json import load as jsonLoad
from pandas import io, MultiIndex, Index
from numpy import floor, random, empty, arange, empty

from ImageAugmentation import ImageAugmentator


class DataGenerator(Sequence):
    def __init__(
        self,
        directory,
        target_json,
        classes_to_consider,
        batch_size=32,
        target_size=(480, 640),
        shuffle=True,
        flip=False,
        rotation=0,
        translate=0,
        scale_min=1,
        scale_max=1,
        shear=0,
    ):
        self.target_size = target_size
        self.batch_size = batch_size
        self.target_json = target_json
        self.directory = directory
        self.img_augmentator = ImageAugmentator(
            flip, rotation, shear, translate, scale_min, scale_max, target_size
        )
        self.shuffle = shuffle

        def json_to_df(json_path, directory):
            with open(json_path, "r") as file:
                json = jsonLoad(file)

            data_frame = io.json.json_normalize(json)
            indexes = MultiIndex.from_tuples([col.split(".") for col in data_frame.columns])
            data_frame.columns = indexes
            data_frame = data_frame.stack(level=[0, 1])
            data_frame = data_frame.set_index(data_frame.index.droplevel(0))
            
            data_frame = data_frame.set_index(
                Index([path.sep.join([directory] + list(c)) for c in data_frame.index.values])
            )
            
            return data_frame

        def preset_data_paths():
            img_paths = []
            img_paths_wo_ext = []
            for root, dirs, files in walk(directory):
                if not(root[-2:] in classes_to_consider):
                    continue

                for file in files:
                    if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
                        img_paths.append(path.join(root, file))
                        img_paths_wo_ext.append(path.splitext(path.join(root, file))[0])
            return img_paths, img_paths_wo_ext

        self.img_paths, self.img_paths_wo_ext = preset_data_paths()
        self.targets = json_to_df(self.target_json, self.directory)
        preset_data_paths()

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = arange(len(self.img_paths))
        if self.shuffle == True:
            random.shuffle(self.indexes)

    def data_generation(self, list_paths, list_paths_wo_ext):
        # Generates data containing batch_size samples
        # Initialization
        h, w = self.target_size
        X = empty((self.batch_size, *self.target_size, 3), dtype=int)
        y = self.targets.loc[list_paths_wo_ext].values
        # Generate data
        for i, ID in enumerate(list_paths):
            image = resize(imread(ID), (w,h))
            self.img_augmentator.preset_transformation_matrix()
            point = (y[i][0] * self.target_size[1], y[i][1] * self.target_size[0])
            X[i,], y[i] = self.img_augmentator.augment_image(image, point)

        return X, y

    def __len__(self):
        return int(floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        list_paths = [self.img_paths[k] for k in indexes]
        list_paths_wo_ext = [self.img_paths_wo_ext[k] for k in indexes]
        return self.data_generation(list_paths, list_paths_wo_ext)
