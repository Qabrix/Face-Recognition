import matplotlib.pyplot as plt
from os import path, listdir
from math import floor

from json import load as jsonLoad
from random import choice as randomChoice


def plot_random_faces(dataSetPath, jsonLabelsPath, numberToDisplay):
    fig = plt.figure(figsize=(20, 20))
    rows = floor(numberToDisplay / 5) + 1
    colums = 4

    with open(jsonLabelsPath) as jsonLabelFile:
        jsonData = jsonLoad(jsonLabelFile)

    dirs = [
        dir for dir in listdir(dataSetPath) if path.isdir(path.join(dataSetPath, dir))
    ]
    for i in range(numberToDisplay):
        # getting random dir for dictionary
        randomDir = randomChoice(dirs)
        dirPath = path.join(dataSetPath, randomDir)

        # getting files from randomed dir, then getting random file
        files = [
            file
            for file in listdir(dirPath)
            if path.isfile(path.join(dirPath, file)) and file.endswith(".jpg")
        ]
        randomFile = randomChoice(files)

        img = plt.imread(path.join(dirPath, randomFile))
        fig.add_subplot(rows, colums, i + 1).axis("Off")
        plt.imshow(img)

        # getting correct label for randomed file
        fx = jsonData[randomDir][randomFile]["x"]
        fy = jsonData[randomDir][randomFile]["y"]

        plt.scatter(x=[fx], y=[fy], c="r", s=5)

    plt.show()
