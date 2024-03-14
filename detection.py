import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


def detect(dataPath, clf):
  """
  Please read detectData.txt to understand the format. Load the image and get
  the face images. Transfer the face images to 19 x 19 and grayscale images.
  Use clf.classify() function to detect faces. Show face detection results.
  If the result is True, draw the green box on the image. Otherwise, draw
  the red box on the image.
    Parameters:A
      dataPath: the path of detectData.txt
    Returns:
      No returns.
  """
  # Begin your code (Part 4)
  '''
  convert each line in the file into lists of elements 
  for each image: 
    read image in both grayscale and color
    read the name of the file and number of faces num_faces on the first line,
    the next num_faces lines are of format (x of left top coordinate, y of left top coordinate, width, height)
    crop regions read from each line from the grayscale image
    classify them, and draw green rectangles if classification is 1 and red if it is 0, on the colored image
    display the modified image

  '''

  with open(dataPath) as file:
    lines = [line.rstrip() for line in file] 
  
  i = 0
  while i < len(lines):
    first_line = lines[i].split()
    file, num_faces = first_line[0], int(first_line[1])
    print(file)
    img = cv2.imread(os.path.join("data/detect", file), cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(os.path.join("data/detect", file))

    for j in range(1,num_faces + 1):
      x, y, w, h = [int(num) for num in lines[i + j].split()]
      left_top = (x, y)
      right_bottom = (x + w, y + h)
      face = img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
      face = cv2.resize(face, (19, 19))

      classification = clf.classify(face)
      if classification == 1:
        img_color = cv2.rectangle(img_color, left_top, right_bottom, (0,255,0), 2)
      else:
        img_color = cv2.rectangle(img_color, left_top, right_bottom, (0,0,255), 2)

    if img is None:
      print('img is None')
    
    cv2.imshow(file ,img_color)
    cv2.waitKey()
    cv2.destroyWindow(file)      
    i += num_faces + 1
      
  # raise NotImplementedError("To be implemented")
  # End your code (Part 4)
