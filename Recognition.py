import os
from sklearn.preprocessing import normalize
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt



class FaceRecongnition:
    def __init__(self, dirToInput: str) -> None:
        self.__FilesList = 0  # list of sample's files names
        self.__DirToInput = dirToInput  # dir to the sample
        self.__FaceMatrix = np.array([])
        self.__EigenFaces = np.array([])

    def __getFilesList(self):
        """ Retrieve all files in a dir into a list """
        self.FilesList = os.listdir(self.__DirToInput)

    def __constructFaceMatrix(self):
        face_matrix = []
        for i in range(len(self.FilesList)):
            # read img as greyscale
            img = cv.imread(self.__DirToInput + "/" + self.FilesList[i], cv.IMREAD_GRAYSCALE)
            # resize img
            img = cv.resize(img, (100, 100))
            # convert img into vector
            numOfDimensions = img.shape[0] * img.shape[1]
            img = np.reshape(img, numOfDimensions)
            # add to face matrix
            face_matrix.append(img)
        self.__FaceMatrix = np.array(face_matrix)

    def __getZeroMeanMatrix(self) -> np.ndarray:
        """ get the mean sample and subtract it from all samples """
        self._mean_sample = np.mean(self.__FaceMatrix, axis=0)
        return np.subtract(self.__FaceMatrix, self._mean_sample)  # zero-mean array

    def __getCovarianceMatrix(self):
        cov = (self.zero_mean_arr.dot(self.zero_mean_arr.T)) / (len(self.FilesList) - 1)

        # get eigenvalues and eigenvectors of cov
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # print("Eigen Values", eigenvalues)
        # print("Eigen Vectors", eigenvectors)
        # Order eigenvalues by index desendingly
        idx = eigenvalues.argsort()[::-1]
        # sort eigenvectors according to eigen values order
        eigenvectors = eigenvectors[:, idx]
        # linear combination of each column of zero_mean_mat
        eigenvectors_c = self.zero_mean_arr.T @ eigenvectors
        # normalize the eigenvectors
        # normalize only accepts matrix with n_samples, n_feature. Hence the transpose.
        self.__EigenFaces = normalize(eigenvectors_c.T, axis=1)
        # print("EigenFaces", self.__EigenFaces)
        print(self.__EigenFaces.shape)

    def detect_face(self, img_path):
        found_flag = 0  # Flag to check if face is found in the dataset
        # testing image
        test_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        # resize the testing image. cv2 resize by width and height.
        test_img = cv.resize(test_img, (100, 100))
        # subtract the mean
        mean_subtracted_test_img = np.reshape(test_img, (100 * 100)) - self._mean_sample
        # print(mean_subtracted_test_img.shape)
        # the vector that represents the image with respect to the eigenfaces.
        vector_of_mean_subtracted_test_img = self.__EigenFaces.dot(mean_subtracted_test_img)
        # print(vector_of_mean_subtracted_test_img.shape)
        # chosen threshold for face detection
        alpha_1 = 3000
        # n^2 vector of the new face image represented as the linear combination of the chosen eigenfaces
        # 90 % of dataset is number of chosen eigenfaces
        projected_new_img_vector = self.__EigenFaces.T @ vector_of_mean_subtracted_test_img
        diff = mean_subtracted_test_img - projected_new_img_vector
        # distance between the original face image vector and the projected vector.
        beta = math.sqrt(diff.dot(diff))

        if beta < alpha_1:
            print(f"Face detected in the image!, beta = {beta}")
            found_flag = 1
        else:
            print(f"No face detected in the image!, beta = {beta} ")
        return vector_of_mean_subtracted_test_img

    def __faceRecognition(self, vector_of_mean_subtracted_test_img,percent):
        threshold = 3000
        label = 0
        #  start distance with 0
        smallest_distance = 0
        #  iterate over all image vectors until fit input image
        for i in range(len(self.FilesList)):
            # projecting each image in face space
            Edb = self.__EigenFaces.dot(self.zero_mean_arr[i])
            # print(Edb.shape)
            # calculating euclidean distance between vectors
            differnce = vector_of_mean_subtracted_test_img[:int(percent*len(self.FilesList))] - Edb[:int(percent*len(self.FilesList))]
            euclidean_distances = math.sqrt(differnce.dot(differnce))
            # get smallest distance
            if smallest_distance == 0:
                smallest_distance = euclidean_distances
                label = i
            if smallest_distance > euclidean_distances:
                smallest_distance = euclidean_distances
                label = i
        # comparing smallest distance with threshold
        if smallest_distance < threshold:
            k = 1
            # print("the input image fit :", self.FilesList[label])
            return self.FilesList[label],k
        else:
            k = 0
            # print("unknown Face")
            return "unknown Face", k

    def roc(self, folder_path,threshold):
        img_list = os.listdir(folder_path)
        output_images = []
        labels=[]
        for img in img_list:
            test_img = cv.imread(folder_path+'/'+img, cv.IMREAD_GRAYSCALE)
            # resize the testing image. cv2 resize by width and height.
            test_img = cv.resize(test_img, (100, 100))
            # subtract the mean
            mean_subtracted_test_img = np.reshape(test_img, (100 * 100)) - self._mean_sample
            # the vector that represents the image with respect to the eigenfaces.
            vector_of_mean_subtracted_test_img = self.__EigenFaces.dot(mean_subtracted_test_img)
            result, label = self.__faceRecognition(vector_of_mean_subtracted_test_img, threshold)
            output_images.append(result)
            labels.append(label)

        roc =self.compare(img_list, output_images,labels)
        return roc

    def compare(self, x, y,label):
        roc = []
        tpr = 0
        fpr = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 00
        for l in range(20):
            txt = x[l]
            I = txt.split('_')
            txt2 = y[l]
            O = txt2.split('_')
            if I[0] in ['yaleB01', 'yaleB02', 'yaleB03', 'yaleB04', 'yaleB05','yaleB06','yaleB07','yaleB08']:
                if I[0] == O[0]:
                    # print(I[0],O[0])
                    tp = tp + 1
                    # print("tp")
                else:
                    fn = fn + 1
                    # print("fn")
            else:
                if label[l] == 1:
                    fp = fp + 1
                    # print("fp")
                else:
                    tn = tn + 1
                    # print("tn")
        # print(tp, fn, fp, tn)
        tpr = tp/(tp+fn)
        fpr = tn/(tn+fp)
        roc.extend([tpr, fpr])
        return roc

    def plot(self,result):
        y = [result[0],result[2],result[4],result[6],result[8]]
        x = [result[1], result[3],result[5],result[7],result[9]]
        plt.plot(x, y)
        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
        # giving a title to my graph
        plt.title('roc')
        # function to show the plot
        plt.savefig("roc.png")



    def fit(self, test_img):
        self.__getFilesList()
        self.__constructFaceMatrix()
        self.zero_mean_arr = self.__getZeroMeanMatrix()
        self.__getCovarianceMatrix()
        vector_of_mean_subtracted_test_img = self.detect_face(test_img)
        result = self.__faceRecognition(vector_of_mean_subtracted_test_img, 0.9)
        return result

    def Roc(self):
        thrsholds = [0.01, 0.02, 0.05, 0.5, 0.9]
        result =[]
        self.__getFilesList()
        self.__constructFaceMatrix()
        self.zero_mean_arr = self.__getZeroMeanMatrix()
        self.__getCovarianceMatrix()
        for i in thrsholds:
             roc = self.roc("test",i)
             # print(roc)
             result.append(roc)
        results =np.array(result)
        self.plot(results.flatten())







