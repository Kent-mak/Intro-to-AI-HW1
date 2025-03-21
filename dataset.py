import os
import cv2
import glob
import numpy as np


def load_data_small():
    """
        This function loads images form the path: 'data/data_small' and return the training
        and testing dataset. The dataset is a list of tuples where the first element is the 
        numpy array of shape (m, n) representing the image the second element is its 
        classification (1 or 0).

        Parameters:
            None

        Returns:
            dataset: The first and second element represents the training and testing dataset respectively
    """

    # Begin your code (Part 1-1)
    '''
    1. Read all images from each folder in grayscale
    2. For images from "face" folders, label 1. For Images from "non-face" folders, label 0
    3. return list with first and second element as the training and testing dataset respectively
    '''
    path = "data/data_small"
    test_path_nonface = path + '/test/non-face/*.pgm'
    test_path_face = path + '/test/face/*.pgm'
    train_path_nonface = path + '/train/non-face/*.pgm'
    train_path_face = path + '/train/face/*.pgm'

    test_set = []
    training_set = []

    for file in glob.glob(train_path_face):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        tup = (img, 1)
        training_set.append(tup)
    
    
    for file in glob.glob(train_path_nonface):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        tup = (img, 0)
        training_set.append(tup)

    for file in glob.glob(test_path_face):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        tup = (img, 1)
        test_set.append(tup)
    
    for file in glob.glob(test_path_nonface):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        tup = (img, 0)
        test_set.append(tup)

    dataset = [training_set, test_set]
    # raise NotImplementedError("To be implemented")
    # End your code (Part 1-1)
    
    return dataset


def load_data_FDDB(data_idx="01"):
    """
        This function generates the training and testing dataset  form the path: 'data/data_small'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("data/data_FDDB/FDDB-folds/FDDB-fold-{}-ellipseList.txt".format(data_idx)) as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0

    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        img_gray = cv2.imread(os.path.join("data/data_FDDB", line_list[line_idx] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx + 1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
            x, y = coord[3] - coord[1], coord[4] - coord[0]            
            w, h = 2 * coord[1], 2 * coord[0]

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 2

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have alreadly save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
        for i in range(num_faces):
            # Begin your code (Part 1-2)
            '''
            for num_faces iterations:
                continue to generate random croppings of the image until the area doesn't overlap any of the faces.
                Implemented by generating a random top left coordinate and a random bottom right coordinate which has greater x and y values
                then check if it overlaps and of the areas in face_box_list and if it's area is grater than 0
            '''
            is_face = True
            valid = False

            # rand_left_top, rand_right_bottom = None, None
            while is_face and not valid:
                rand_left_top = (np.random.randint(0, img_gray.shape[1]) ,np.random.randint(0, img_gray.shape[0]))
                rand_right_bottom = (np.random.randint(rand_left_top[0], img_gray.shape[1]), np.random.randint(rand_left_top[1], img_gray.shape[0]))
                valid = True
                if (rand_left_top[0] - rand_right_bottom[0] >= 0) or (rand_left_top[1] - rand_right_bottom[1] >= 0):
                    valid = False
                    continue

                for face in face_box_list:
                    if face[0][0] > rand_right_bottom[0] or rand_left_top[0] > face[1][0] or face[0][1] > rand_right_bottom[1] or rand_left_top[1] > face[1][1]:
                        is_face = False
                        break

            img_crop = img_gray[rand_left_top[1]:rand_right_bottom[1], rand_left_top[0]:rand_right_bottom[0]].copy()
            # raise NotImplementedError("To be implemented")
            # End your code (Part 1-2)

            nonface_dataset.append((cv2.resize(img_crop, (19, 19)), 0))

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[:int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[int(SPLIT_RATIO * num_nonface_data):]

    return train_dataset, test_dataset


def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()
