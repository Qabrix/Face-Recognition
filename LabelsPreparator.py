from os import walk, path, makedirs
from json import dump

from FaceCenterCalculator import get_faceCenter_cords
from ConstValues import ORG_IMG_SIZE

def prepare_json_data(data_set_dir, classes_to_consider):
    json_data = {}
    for (root, dirs, files) in walk(data_set_dir):
        if root == data_set_dir:
            for dir in dirs:
                # creating key in dictionary for each folder from the data
                json_data[dir] = {}
            continue

        if not (root[-2:] in classes_to_consider):
            continue

        for file in files:
            if file.endswith(".jpg"):
                pose_txt_path = path.join(root, file[0:-7] + "pose.txt")
                rgb_cal_path = path.join(root, "rgb.cal")

                with open(pose_txt_path) as poseTxtFile:
                    with open(rgb_cal_path) as rgbCalFile:
                        pose_file_lines = poseTxtFile.readlines()
                        rgb_cal_file_lines = rgbCalFile.readlines()

                        label_x, label_y = get_faceCenter_cords(
                            pose_file_lines, rgb_cal_file_lines
                        )
                        dir_name, file_name = root[-2:], file

                        # creating key in sub-dictionary for each image
                        json_data[dir_name][file_name] = {}
                        json_data[dir_name][file_name]["x"] = label_x / ORG_IMG_SIZE[1]
                        json_data[dir_name][file_name]["y"] = label_y / ORG_IMG_SIZE[0]

    return json_data


def save_labels_in_json(
    project_dir,
    data_set_dir_name,
    labels_dir_name,
    labels_file_name,
    classes_to_consider,
):
    # ensuring that labels folder exist
    labels_path = path.join(project_dir, labels_dir_name, labels_file_name)

    makedirs(path.dirname(labels_path), exist_ok=True)

    json_data = prepare_json_data(
        path.join(project_dir, data_set_dir_name), classes_to_consider
    )
    with open(labels_path, "w") as labelsFile:
        # dumping json data to the file
        dump(json_data, labelsFile, indent=4)
