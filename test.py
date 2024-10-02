import cv2
from PIL import Image
import numpy as np


image1 = Image.open('./app/resource/images/fishing/logImages/3.png')
imageCV = cv2.imread('./app/resource/images/fishing/logImages/4.png')
image1.show()

# 将图像转换为NumPy数组
img_np1 = np.array(image1)

# 将图像从RGB格式转换为BGR格式（OpenCV使用BGR）
bgr_image1 = cv2.cvtColor(img_np1, cv2.COLOR_RGB2BGR)

lower_yellow = np.array([255,255,20])
upper_yellow = np.array([255, 255, 23])
XK_yellow = np.array([30, 255, 255])

# 创建黄色掩膜
mask1 = cv2.inRange(bgr_image1, lower_yellow, upper_yellow)
mask2 = cv2.inRange(imageCV, lower_yellow, upper_yellow)
# mask1 = cv2.inRange(bgr_image1, XK_yellow, XK_yellow)
# mask2 = cv2.inRange(imageCV, XK_yellow, XK_yellow)


# 查找轮廓
contours1, _ = cv2.findContours(mask1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print('\n1号分割图像个数为：',len(contours1))
print('2号分割图像个数为：',len(contours2))

# 展示掩膜图片
cv2.imshow('Mask1', mask1)
cv2.imshow('Mask2', mask2)
cv2.waitKey(0)

