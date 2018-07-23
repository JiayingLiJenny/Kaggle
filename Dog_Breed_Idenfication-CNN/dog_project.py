

Step 0: 导入数据集
Step 1: 检测人脸
Step 2: 检测狗狗
Step 3: 从头创建一个CNN来分类狗品种
Step 4: 使用一个CNN来区分狗的品种(使用迁移学习)
Step 5: 建立一个CNN来分类狗的品种（使用迁移学习）
Step 6: 完成你的算法
Step 7: 测试你的算法



步骤 0: 导入数据集

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

# 定义函数来加载train，test和validation数据集
def load_dataset(path):
    data = load_files(path)
    dog_file = np.array(data['filenames']) 
    # 子文件夹的名称作为label，可看http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
    # np.array(data['filenames'])返回的是地址
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133) #怎么知道是133（通过观察？）
    # 如何通过这个获取y_train,或者y_test的数据
    return dog_file, dog_targets

# 加载train，test和validation数据集
train_files, train_targets = load_dataset('dogImages/train') 
valid_files, valid_targets = load_dataset('dogImages/valid') # dogImages里面就是train/valid/test 3 个文件夹
test_files, test_targets = load_dataset('dogImages/test')

# 加载狗品种列表
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]  #item[20:-1]该怎么理解

# 打印数据统计描述
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))






import  random
random.seed(8675309)

# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("lfw/*/*")) 
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))


步骤1：检测人脸
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline

# 如果img_path路径表示的图像检测到了脸，返回"True" 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0 #返回的是True/False


human_files_short = human_files[:100] #human_files_short 和 dog_files_short存储的就是文件路径
dog_files_short = train_files[:100]


## TODO: 基于human_files_short和dog_files_short
## 中的图像测试face_detector的表现
human = []
human = (face_detector(human_files_short[i]) for i in range(1000))
print(human)
检测：可以看看上面这个代码对不对？？？？
####################################################################
human_human = 0
dog_human = 0
for i in range(100):
    if face_detector(human_files_short[i]):
    	human_human += 1
    if face_detector(dog_files_short[i]):
    	dog_human +=1

print("The percentage of human detected in human_files_short is: %.2f%%" % (100*human_human/100))
print('The percentage of human detected in dog_files_short is: %.2f%%' % (100*dog_human/100))


#试试下面这个
print('The percentage of human detected in human_files_short is: {:.2f%}'.format(human_human/100))
print('The percentage of human detected in dog_files_short is: {:.2f%}'.format(dog_human/100))


问题2：
回答：
不合理。因为用户会动，移动可能造成图像模糊。
#####################################################################


步骤 2: 检测狗狗
from keras.applications.resnet50 import ResNet50
# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')


数据预处理
from keras.preprocessing import image
from tqdm import tqdm
def path_to_tensor(img_path):
	# 1.该函数首先读取一张图像，然后将其缩放为 224×224 的图像。
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 2.随后，该图像被调整为具有4个维度的张量。
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 3.将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
# Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。

基于 ResNet-50 架构进行预测
from keras.applications.resnet50 import preprocess_input, decode_predictions
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    img = preprocess_input(path_to_tensor(img_path))
    # preprocess_input实现的功能如下：
    # 首先，这些图像的通道顺序为 RGB，我们需要重排他们的通道顺序为 BGR。
    # 其次，预训练模型的输入都进行了额外的归一化过程。
    # 因此我们在这里也要对这些张量进行归一化，即对所有图像所有像素都减去像素均值 [103.939, 116.779, 123.68]（以 RGB 模式表示，根据所有的 ImageNet 图像算出）。
    return np.argmax(ResNet50_model.predict(img))
# ResNet50_model.predict(img),返回一个向量，向量的第 i 个元素表示该图像属于第 i 个 ImageNet 类别的概率。
# 通过对预测出的向量取用 argmax 函数（找到有最大概率值的下标序号），我们可以得到一个整数，即模型预测到的物体的类别。

检测： 尝试输入 print(img)就可以查看是什么？？？？？？？？？？


完成狗检测模型
# 狗类别对应的序号为151-268。
# 因此，在检查预训练模型判断图像是否包含狗的时候，
# 我们只需要检查如上的 ResNet50_predict_labels 函数是否返回一个介于151和268之间（包含区间端点）的值。
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
# 如果从图像中检测到狗就返回 True，否则返回 False


【作业】评估狗狗检测模型

问题 3:
在下方的代码块中，使用 dog_detector 函数，计算：
human_files_short中图像检测到狗狗的百分比？
dog_files_short中图像检测到狗狗的百分比？

################################################
human_dog = 0
dog_dog = 0
for i in range(100):
    if dog_detector(human_files_short[i]):
        human_dog += 1
    if dog_detector(dog_files_short[i]):
        dog_dog += 1
print(human_dog/100)
print(dog_dog/100)

print('The percentage of dog detected in human_files_short is: {:.2f%}'.format(human_dog/100))
print('The percentage of dog detected in dog_files_short is: {:.2f%}'.format(dog_dog/100))

#################################################

步骤 3: 从头开始创建一个CNN来分类狗品种
我们也提到了随机分类将得到一个非常低的结果：不考虑品种略有失衡的影响，随机猜测到正确品种的概率是1/133，相对应的准确率是低于1%的。
请记住，在深度学习领域，实践远远高于理论。

数据预处理

归一化处理

from  PIL  import  ImageFile                             
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

检测： 
print(paths_to_tensor(valid_files)) 看它输出的是什么，每个都是不一样的吗？


【练习】模型架构
#################################################################################
问题 4:
在下方的代码块中尝试使用 Keras 搭建卷积网络的架构，并回答相关的问题。
你可以尝试自己搭建一个卷积网络的模型，那么你需要回答你搭建卷积网络的具体步骤（用了哪些层）以及为什么这样搭建。
你也可以根据上图提示的步骤搭建卷积网络，那么请说明为何如上的架构能够在该问题上取得很好的表现。
回答:
1. 搭建第一个卷积层，由于是第一层，添加input_shape为(224, 224, 3)，表示输入的是224*224像素且为3通道的彩色图片
2. 搭建第一个最大池化层，为了缩小维度减少参数数量，提高效率，同时pool_size与pool_size设为2，可使特征映射的宽和高都减小为原来的一半
3. 搭建第二个卷积层，filters设为第一卷积层的2倍，更多的过滤器能够获取图片中更多的规律。
4. 搭建第二个最大池化层，与第一个一样
5. 搭建第三个卷积层，filters设为第二卷积层的2倍，以获取更多图片规律。
6. 搭建第三个最大池化层，与第一个一样
7. 搭建全局平均池化层，将最后一个最大池化层的输出缩减为一个向量，作为最后一个密集层的输入
8. 搭建密集层，使用softmax为激活函数返回概率，使输出返回133类


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from keras.models import Sequential
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', 
                 activation='relu', input_shape=(224, 224, 3))) #检测：这里是224还是223？
model.add(MaxPooling2D(pool_size=2)) # strides默认为pool_size
model.add(Conv2D(filters=32, kernel_size=2, padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(GlobalAveragePooling2D()) #这里空着不写任何东西，可以吗？？ input_shape=(27,27,64) 可以不写
model.add(Dense(133, activation='softmax'))

model.summary()

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 224, 224, 16)      208       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 112, 112, 16)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 112, 112, 32)      2080      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 56, 56, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 56, 56, 64)        8256      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 133)               8645      
=================================================================
Total params: 19,189.0
Trainable params: 19,189.0
Non-trainable params: 0.0

#####################################################################################
编译模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


【练习】训练模型
from keras.callbacks import ModelCheckpoint  

### TODO: 设置训练模型的epochs的数量
###################################################
epochs = 5 (20)
###################################################
### 不要修改下方代码

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

## 加载具有最好验证loss的模型

model.load_weights('saved_models/weights.best.from_scratch.hdf5')

测试模型
在狗图像的测试数据集上试用你的模型。确保测试准确率大于1%。

# 获取测试数据集中每一个图像所预测的狗品种的index
dog_breed_predictions = [np.argmax(model.predict(np.expand_dim(tensor, axis=0))) for tensor in test_tensors]

检测： np.expand_dim（）在paths_to_tensor 里不是已经转换为4维张量了吗？为什么这里还要再扩展一次？？？？？

# 报告测试准确率
检测： type(dog_breed_predictions) 我觉得是array
type(test_targets)
print(test_targets[:10])

test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1)) / len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


步骤 4: 使用迁移学习（Transfer Learning）的方法来区分狗的品种,这里使用VGG16
使用 迁移学习（Transfer Learning）的方法，能帮助我们在不损失准确率的情况下大大减少训练时间

得到从图像中提取的特征向量（Bottleneck Features）
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


模型架构
该模型使用预训练的 VGG-16 模型作为固定的图像特征提取器，
其中 VGG-16 最后一层卷积层的输出被直接输入到我们的模型。
我们只需要添加一个全局平均池化层以及一个全连接层，
其中全连接层使用 softmax 激活函数，对每一个狗的种类都包含一个节点。

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
# 通过这种形式得到的train set都【0】都是表示样本数量
# 如下面这个在flatten中的利用：
# model.add(Flatten(input_shape = x_train.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))
VGG16_model.summary()

编译模型
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

训练模型
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
	                           verbose=1, save_best_only=True)
VGG16_model.fit(train_VGG16, train_targets,
	            validation_data=(valid_VGG16, valid_targets),
	            epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

加载具有最好验证loss的模型
VGG16_model.load_wights('saved_models/weights.best.VGG16.hdf5')

测试模型
现在，我们可以测试此CNN在狗图像测试数据集中识别品种的效果如何。我们在下方打印出测试准确率。
# 获取测试数据集中每一个图像所预测的狗品种的index
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)



步骤 5: （使用迁移学习）用另一个预训练模型来搭建一个 CNN，这里使用ResNet50
#############################################################################
bottleneck_features = np.load('bottleneck_features/DogResNet50Data.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']


### TODO: 定义你的框架
ResNet50_model = Sequential()
ResNet50_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet50_model.add(Dense(133, activation='softmax'))
ResNet50_model.summary()
### TODO: 编译模型
ResNet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
### TODO: 训练模型
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5',
	                           verbose=1, save_best_only=True)
ResNet50_model.fit(train_ResNet50, train_targets,
	            validation_data=(valid_InceptionV3, valid_targets),
	            epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
### TODO: 加载具有最佳验证loss的模型权重
ResNet50_model.load_wights('saved_models/weights.best.VGG16.hdf5')
### TODO: 在测试集上计算分类准确率
ResNet50_predictions = [np.argmax(ResNet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]
test_accuracy = 100*np.sum(np.array(ResNet50_predictions)==np.argmax(test_targets, axis=1))/len(ResNet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

检测：
改变epochs改变准确率，
多试几个model替换InceptionV3
##########################################################################

练习】使用模型测试狗的品种
函数应当包含如下三个步骤：

1. 根据选定的模型载入图像特征（bottleneck features）
2. 将图像特征输输入到你的模型中，并返回预测向量。注意，在该向量上使用 argmax 函数可以返回狗种类的序号。
3. 使用在步骤0中定义的 dog_names 数组来返回对应的狗种类名称。


extract_InceptionV3内部代码
#输入是张量
#predict该模型的n个类别（维度）概率向量
#输出：图片穿过这个model（InceptionV3）后的输出
#InceptionV3做了什么事情，它里面的参数表示什么意思
#include_top=False： 去掉最后一层
#weights='imagenet'：weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
def extract_InceptionV3(tensor):
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
问题 9:
##############################################################################
from extract_bottleneck_features import extract_InceptionV3
#获取图片路径
img_path = glob.glob("images/dog_01.jpg")
#把路径转换为张量
def path_to_tensor(img_path):
	# RGB --> PIL.image.image
    img = image.load_img(img_path, target_size=(229, 229)) # InceptionV3的默认input—size是229*229
    # PIL.image.image --> 3D tensor (229, 229, 3)
    x = image.img_to_array(img)
    # 3D tensor --> 4D tensor (1, 229, 229, 3)
    return np.expand_dims(x, axis=0)
             
#把pre_training的输出作为模型的输入
# 获取bottleneck的特征
bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
# 获取预测向量
predicted_vector = InceptionV3_model.predict(bottleneck_feature)
检测： 与下面这个有什么不同（下面这个是这里的写法）
InceptionV3_model.predict(np.expand_dims(bottleneck_feature, axis=0))

dog_names[np.argmax(predicted_vector)]
#输入新模型，并找出分类
print(np.argmax())
------------------------------------------
如果用的是extract_Resnet50
from extract_bottleneck_features import extract_Resnet50
def Renet50_predict_breed(img_path):
    bottlenect_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = ResNet50_model.predict(bottlenect_feature)
    return(dog_names[np.argmax(predicted_vector)])


##############################################################################
步骤 6: 完成你的算法

实现一个算法，它的输入为图像的路径，它能够区分图像是否包含一个人、狗或两者都不包含，然后：
如果从图像中检测到一只狗，返回被预测的品种。
如果从图像中检测到人，返回最相像的狗品种。
如果两者都不能在图像中检测到，输出错误提示。
####################################################
def detect_human_or_dogbreed(img_path):
	# 加载彩色（通道顺序为BGR）图像
    img = cv2.imread(img_path)
	# 将BGR图像转变为RGB图像以打印
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 展示含有识别框的图像
    plt.imshow(cv_rgb)
    plt.show()
    if face_detector(img_path):
    	print('hello, human!')
    	print('You look like a {}'.format(Renet50_predict_breed(img_path)))
    elif dog_detector(img_path):
    	print('hello, this is a {}'.format(Renet50_predict_breed(img_path)))
    else:
    	print('error input')
####################################################

步骤 7: 测试你的算法
问题 11:
在下方编写代码，用至少6张现实中的图片来测试你的算法。你可以使用任意照片，不过请至少使用两张人类图片（要征得当事人同意哦）和两张狗的图片。 同时请回答如下问题：
输出结果比你预想的要好吗 :) ？或者更糟 :( ？
提出至少三点改进你的模型的想法。
####################################################
detect_human_or_dogbreed('test_images/dog_Labrador.jpg')
detect_human_or_dogbreed('test_images/dog_Hasky.jpg')
detect_human_or_dogbreed('test_images/dog_BichonFrise.jpg')
detect_human_or_dogbreed('test_images/human_gaoyuanyuan.jpg')
detect_human_or_dogbreed('test_images/human_huge.jpg')
detect_human_or_dogbreed('test_images/cat.jpg')
detect_human_or_dogbreed('test_images/panda.jpg')

改进：
1. 改变epochs，找到valid loss最小的地方。
2. 添加dropout层，避免过拟合
3. 添加池化层，避免过拟合
####################################################



















