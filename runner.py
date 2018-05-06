
import constants as c
from neuralMesh import NeuralMesh
from benchmarkModel import FeedForward

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import struct, math, random, sys

import glob
import sys
import os


#Neural net: input -> grid of neurons connected to adjacent neurons -> output. 2D input, mesh, and output implementation.
class Runner:
    def __init__(self):
        # data #
        self.trainImagesLabelsTup = None
        self.testImagesLabelsTup = None
        self.readInData()

        #model
        self.model = None #dont make model yet since benchmark may want lots of models

    # reads in image and label data from specified files. returns list of tuples of (img, label), ()...
    def readInDataFromImgLabelFiles(self, imageFile, labelFile, numImages):
        images = []
        labels = []
        with open(imageFile, "rb") as imageFile:
            with open(labelFile, "rb") as labelFile:
                imageFile.read(16) #get rid of header
                labelFile.read(8) #get rid of header
                for _ in range(numImages):
                    #image data
                    imgBytes = struct.unpack("784B", imageFile.read(c.IMG_VEC_SZ))
                    imgInput = np.array(imgBytes, dtype=float)/255.0
                    #label data
                    labelBytes = struct.unpack("1B", labelFile.read(1))
                    label = int(labelBytes[0])
                    #store both in memory
                    images.append(imgInput)
                    labels.append(label)
        return (np.array(images), np.array(labels))

    def readInData(self):
        print("\nREADING IN DATA...")
        self.trainImagesLabelsTup = self.readInDataFromImgLabelFiles(c.TRAIN_DATA_FILE_PATH, c.TRAIN_LABEL_FILE_PATH, c.NUM_TRAIN_IMGS)
        self.testImagesLabelsTup = self.readInDataFromImgLabelFiles(c.TEST_DATA_FILE_PATH, c.TEST_LABEL_FILE_PATH, c.NUM_TEST_IMGS)
        # to verify we read in corectly...
        # plt.imshow(np.array(self.trainImagesLabelsTup[0][764]).reshape(28,28))
        # plt.show()
        # exit()
        # if you have internet access...
        # mnist = input_data.read_data_sets("MNIST_data/")
        # self.trainImagesLabelsTup = (mnist.train.images[:c.NUM_TRAIN_IMGS], mnist.train.labels[:c.NUM_TRAIN_IMGS])
        # self.testImagesLabelsTup = (mnist.test.images[:c.NUM_TEST_IMGS], mnist.test.labels[:c.NUM_TEST_IMGS])

    def train(self):
        if not self.model:
            self.model = NeuralMesh()
        #train
        print("\nTRAINING...")
        self.model.train(self.trainImagesLabelsTup, self.testImagesLabelsTup)

    def test(self, display_energy_num=0):
        if not self.model:
            self.model = NeuralMesh()
        #test
        print("\nTESTING...")
        self.model.test(self.testImagesLabelsTup, display_energy_num)

    def benchmark_plot(self):
        #lengths = [2, 5, 10, 20, 35]
        lengths = [2, 5, 10, 20, 25]
        #lengths = [90, 95, 105]
        num_neurons = [l**2 for l in lengths]

        mesh_accuracies = []
        ff_accuracies = []

        for length in lengths:
            mesh = NeuralMesh(mesh_rows=length, mesh_cols=length, saveable=False)
            mesh.train(self.trainImagesLabelsTup)
            mesh_accuracies.append(mesh.test(self.testImagesLabelsTup))

            tf.reset_default_graph()

            ff = FeedForward(length**2)
            ff.train(self.trainImagesLabelsTup)
            ff_accuracies.append(ff.test(self.testImagesLabelsTup))

            tf.reset_default_graph()

        plt.plot(num_neurons, mesh_accuracies, '-b', label="Neural Mesh")
        plt.plot(num_neurons, ff_accuracies, '-r', label="Feed Forward")
        plt.title("Model Accuracy by Neuron")
        plt.xlabel("number of neurons")
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    def benchmark_FF_SGD_plot(self):
        #lengths = [2, 5, 10, 20, 35]
        lengths = [2, 5, 10, 20, 25]
        #lengths = [90, 95, 105]
        num_neurons = [l**2 for l in lengths]

        sgd_accuracies = []
        gd_accuracies = []

        for length in lengths:
            ff_SGD = FeedForward(length**2, stochastic=True)
            ff_SGD.train(self.trainImagesLabelsTup)
            sgd_accuracies.append(ff_SGD.test(self.testImagesLabelsTup))

            tf.reset_default_graph()

            ff_GD = FeedForward(length**2, stochastic=False)
            ff_GD.train(self.trainImagesLabelsTup)
            gd_accuracies.append(ff_GD.test(self.testImagesLabelsTup))

            tf.reset_default_graph()

        plt.plot(num_neurons, sgd_accuracies, '-b', label="FF SGD")
        plt.plot(num_neurons, gd_accuracies, '-r', label="FF GD")
        plt.title("Model Accuracy by Neuron")
        plt.xlabel("number of neurons")
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    def window_plot(self):
        szs = [1, 5, 10, 25, 100]
        mesh_accuracies = []
        for sz in szs:
            #hard coded to 25x25 so it fits in mem
            mesh = NeuralMesh(mesh_rows=25, mesh_cols=25, window_sz=sz, saveable=False)
            mesh.train(self.trainImagesLabelsTup)
            mesh_accuracies.append(mesh.test(self.testImagesLabelsTup))
            tf.reset_default_graph()
        plt.plot(szs, mesh_accuracies, '-b')
        plt.title("Mesh Accuracy by Window Size")
        plt.xlabel("window size")
        plt.show()






# helper functions #

def deletePrevSummaries():
    summaryFiles = glob.glob(c.ALL_SUMMARY_FILES)
    for f in summaryFiles:
        os.remove(f)

def deleteModelFiles():
    modelFiles = glob.glob(c.ALL_MODEL_FILES)
    for f in modelFiles:
        os.remove(f)

def parseArgs():
    args = sys.argv[1:]
    argSet = set(args)
    validArgSet = set(["-n", "-N", "--NEW", "--TEST", "--TRAIN", "--BENCH", "--ENERGY"])

    #check to make sure all args valid
    if any(map(lambda arg: arg not in validArgSet, args)):
        print("<USAGE: only --TRAIN --TEST -n, -N, --NEW are supported>")
        exit(0)


    #new model option
    new = False
    if any(map(lambda arg: arg in argSet, ["-n", "-N", "--NEW"])):
        new = True

    #test option
    test = False
    if "--TEST" in argSet:
        test = True

    #train option
    train = False
    if "--TRAIN" in argSet:
        train = True

    #benchmark option
    bench = False
    if "--BENCH" in argSet:
        bench = True

    #show energies option
    energy = False
    if "--ENERGY" in argSet:
        energy = True


    return train, test, new, bench, energy



# main #

def main():
    train, test, new, bench, energy = parseArgs()

    #make dirs for saving 
    if not os.path.exists(c.MODEL_DIR):
        os.mkdir(c.MODEL_DIR)
    if not os.path.exists(c.SUMMARY_DIR):
        os.mkdir(c.SUMMARY_DIR)

    if new:
        #remove all model files
        deleteModelFiles()
        #delete all summafries
        deletePrevSummaries()

    if any([train, test, bench, energy]):
       runner = Runner()

    if bench:
        print("comparing SGD to GD for FF")
        runner.benchmark_FF_SGD_plot()
        print("making benchmark plot...")
        runner.benchmark_plot()
        print("making windows plot")
        runner.window_plot()
    if train:
        runner.train()
    if test or energy:
        display_energy_num = 3 if energy else 0
        runner.test(display_energy_num=display_energy_num)

    


if __name__ == "__main__":
    main()
