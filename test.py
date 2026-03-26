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



im = cv2.imread('color_image_bright.jpg')
# 474 445 plus or minus 6 pixels
sub_im = im[441:453,469:479]
resized_sub_im = cv2.resize(sub_im, (300, 300), interpolation=cv2.INTER_LINEAR)
cv2.imshow("Sub Image", resized_sub_im)
cv2.waitKey(0) 

hsv_im = cv2.cvtColor(resized_sub_im, cv2.COLOR_BGR2HSV)
lower_dark = np.array([0, 0, 150])
upper_dark = np.array([25, 255, 255]) # Adjust '60' to change darkness sensitivity

# 3. Create a mask of the DARK pixels
bright_mask = cv2.inRange(hsv_im, lower_dark, upper_dark)

result = cv2.bitwise_and(resized_sub_im, resized_sub_im, mask=bright_mask)
cv2.imshow("Bright Areas", result)
cv2.waitKey(0)

# find characteristic color in hsv
hsv_mean = cv2.mean(hsv_im, mask=bright_mask)
hsv_sd = cv2.meanStdDev(hsv_im, mask=bright_mask)
print("Mean HSV:", hsv_mean)
print("Std Dev HSV:", hsv_sd)