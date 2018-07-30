# 一. 
# 导入人脸图像数据集，文件所在路径存储在名为 human_files 的 numpy 数组。

import random
random.seedseed((86753098675309)

# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))


'''
二.
检测人脸
使用 OpenCV 中的 Haar feature-based cascade classifiers 来检测图像中的人脸。
OpenCV 提供了很多预训练的人脸检测模型，它们以XML文件保存在 github。
我们已经下载了其中一个检测模型，并且把它存储在 haarcascades 的目录中。
'''

import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# 提取预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# 加载彩色（通道顺序为BGR）图像
img = cv2.imread(human_files[3])

# 将BGR图像进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找出脸
faces = face_cascade.detectMultiScale(gray)

# 打印图像中检测到的脸的个数
print('Number of faces detected:', len(faces))

# 获取每一个所检测到的脸的识别框
for (x,y,w,h) in faces:
    # 在人脸图像中绘制出识别框
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# 将BGR图像转变为RGB图像以打印
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 展示含有识别框的图像
plt.imshow(cv_rgb)
plt.show()

'''
在使用任何一个检测模型之前，将图像转换为灰度图是常用过程。
detectMultiScale 函数使用储存在 face_cascade 中的的数据，对输入的灰度图像进行分类。
在上方的代码中，faces 以 numpy 数组的形式，保存了识别到的面部信息。
它其中每一行表示一个被检测到的脸，该数据包括如下四个信息：
前两个元素  x、y 代表识别框左上角的 x 和 y 坐标（参照上图，注意 y 坐标的方向和我们默认的方向不同）；
后两个元素代表识别框在 x 和 y 轴两个方向延伸的长度 w 和 d。
'''

# 三.
# 定义人脸识别器函数
# 该函数的输入为人脸图像的路径，当图像中包含人脸时，该函数返回 True，反之返回 False。该函数定义如下所示。
# 如果img_path路径表示的图像检测到了脸，返回"True" 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# 四.
# 评估人脸检测模型
'''
使用 face_detector 函数，计算：
human_files 的前100张图像中，能够检测到人脸的图像占比多少
dog_files 的前100张图像中，能够检测到人脸的图像占比多少
'''
human_files_short = human_files[:100] #human_files_short 和 dog_files_short存储的就是文件路径
dog_files_short = train_files[:100]


human_human = 0
dog_human = 0
for i in range(100):
    if face_detector(human_files_short[i]):
        human_human += 1
    if face_detector(dog_files_short[i]):
        dog_human +=1

print("The percentage of human detected in human_files_short is: %.2f%%" % (100*human_human/100))
print('The percentage of human detected in dog_files_short is: %.2f%%' % (100*dog_human/100))
















