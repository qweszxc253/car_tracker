import cv2
import numpy as np

# im = cv2.imread('color_image.jpg')
# # 246 534 plus or minus 6 pixels
# sub_im = im[239:254,527:543]
# resized_sub_im = cv2.resize(sub_im, (300, 300), interpolation=cv2.INTER_LINEAR)
# cv2.imshow("Sub Image", resized_sub_im)
# cv2.waitKey(0) 

# hsv_im = cv2.cvtColor(resized_sub_im, cv2.COLOR_BGR2HSV)
# lower_dark = np.array([0, 0, 150])
# upper_dark = np.array([180, 255, 255]) # Adjust '60' to change darkness sensitivity

# # 3. Create a mask of the DARK pixels
# bright_mask = cv2.inRange(hsv_im, lower_dark, upper_dark)

# result = cv2.bitwise_and(resized_sub_im, resized_sub_im, mask=bright_mask)
# cv2.imshow("Bright Areas", result)
# cv2.waitKey(0)

# # find characteristic color in hsv
# hsv_mean = cv2.mean(hsv_im, mask=bright_mask)
# hsv_sd = cv2.meanStdDev(hsv_im, mask=bright_mask)
# print("Mean HSV:", hsv_mean)
# print("Std Dev HSV:", hsv_sd)



# im = cv2.imread('color_image_bright.jpg')
# # 474 445 plus or minus 6 pixels
# sub_im = im[441:453,469:479]
# resized_sub_im = cv2.resize(sub_im, (300, 300), interpolation=cv2.INTER_LINEAR)
# cv2.imshow("Sub Image", resized_sub_im)
# cv2.waitKey(0) 

# hsv_im = cv2.cvtColor(resized_sub_im, cv2.COLOR_BGR2HSV)
# lower_dark = np.array([0, 0, 150])
# upper_dark = np.array([25, 255, 255]) # Adjust '60' to change darkness sensitivity

# # 3. Create a mask of the DARK pixels
# bright_mask = cv2.inRange(hsv_im, lower_dark, upper_dark)

# result = cv2.bitwise_and(resized_sub_im, resized_sub_im, mask=bright_mask)
# cv2.imshow("Bright Areas", result)
# cv2.waitKey(0)

# # find characteristic color in hsv
# hsv_mean = cv2.mean(hsv_im, mask=bright_mask)
# hsv_sd = cv2.meanStdDev(hsv_im, mask=bright_mask)
# print("Mean HSV:", hsv_mean)
# print("Std Dev HSV:", hsv_sd)


# read data df from csv in recording folder, under data
# import pandas as pd
# df = pd.read_csv('recordings/data/data-20260416_160923-20260416_160926.csv')

# # calculate mean and sd for each column
# mean = df.mean()
# sd = df.std()

# print("Mean:\n", mean)
# print("Std Dev:\n", sd)

# # print sd as fraction of mean
# fraction_sd = sd / mean
# print("Fraction of Std Dev to Mean:\n", fraction_sd)


import math

tilt_arr = [0.2,0.2,-0.3,-0.5,-0.4,0.0,
            0.2,-0.1,0.0,-0.1,0.4,0.2,
            0.0,0.1,0.3,0.7,0.8,0.6]

# calculate height arr for tilt arr in degree

segment_length = 6  #inches

def calc_height_arr(tilt_arr):
    height_arr = [0]
    for i in range(len(tilt_arr)):
        #convert tilt degree to radians
        tilt_rad = tilt_arr[i] * math.pi / 180
        height_arr.append(height_arr[i] + math.sin(tilt_rad) * segment_length)
        
    return height_arr

height_arr = calc_height_arr(tilt_arr)
print(height_arr)

#graph the height arr graph

# each index is 6 inches appart

import matplotlib.pyplot as plt
import numpy as np

#make x array in inches

x = np.arange(len(height_arr)) * 6
y = height_arr

plt.plot(x, y)
plt.xlabel('Distance (in)')
plt.ylabel('Height (in)')
plt.title('Height vs Distance')
plt.show()




