'''
一.
导入狗数据集
在下方的代码单元（cell）中，导入了一个狗图像的数据集。使用 scikit-learn 库中的 load_files 函数来获取一些变量：
train_files, valid_files, test_files - 包含图像的文件路径的numpy数组
train_targets, valid_targets, test_targets - 包含独热编码分类标签的numpy数组
dog_names - 由字符串构成的与标签相对应的狗的种类
'''
from sklearn.datasets  import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# 定义函数来加载train，test和validation数据集
# 分为features和labels
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# 加载train，test和validation数据集
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# 加载狗品种列表
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# 打印数据统计描述
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


'''
二.
检测狗狗
在这个部分中，我们使用预训练的 ResNet-50 模型去检测图像中的狗。
ImageNet 是目前一个非常流行的数据集，常被用来测试图像分类等计算机视觉任务相关的算法。
它包含超过一千万个 URL，每一个都链接到 1000 categories 中所对应的一个物体的图像。
任给输入一个图像，该 ResNet-50 模型会返回一个对图像中物体的预测结果。
'''
# 1. 
# 定义ResNet50模型
from keras.applications.resnet50 import ResNet50
ResNet50_model = ResNet50(weights='imagenet')
#下载了 ResNet-50 模型的网络结构参数，以及基于 ImageNet 数据集的预训练权重。

'''
2.
数据预处理

在使用 TensorFlow 作为后端的时候，在 Keras 中，CNN 的输入是一个4维数组（也被称作4维张量），
它的各维度尺寸为 (nb_samples, rows, columns, channels)。

其中 nb_samples 表示图像（或者样本）的总数，rows, columns, 和 channels 分别表示图像的行数、列数和通道数。
下方的 path_to_tensor 函数实现如下将彩色图像的字符串型的文件路径作为输入，
返回一个4维张量，作为 Keras CNN 输入。因为我们的输入图像是彩色图像，因此它们具有三个通道（ channels 为 3）。

paths_to_tensor 函数将图像路径的字符串组成的 numpy 数组作为输入，并返回一个4维张量，
各维度尺寸为 (nb_samples, 224, 224, 3)。 在这里，nb_samples是提供的图像路径的数据中的样本数量或图像数量。
也可以将 nb_samples 理解为数据集中3维张量的个数（每个3维张量表示一个不同的图像。
'''
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


# 3.
# 基于 ResNet-50 架构进行预测
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

# 4.
# 完成狗检测模型
# 狗类别对应的序号为151-268。
# 因此，在检查预训练模型判断图像是否包含狗的时候，
# 我们只需要检查如上的 ResNet50_predict_labels 函数是否返回一个介于151和268之间（包含区间端点）的值。
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
# 如果从图像中检测到狗就返回 True，否则返回 False


# 5.
# （结合人脸图片）评估狗狗检测模型
'''
在下方的代码块中，使用 dog_detector 函数，计算：
human_files_short中图像检测到狗狗的百分比
dog_files_short中图像检测到狗狗的百分比
'''
human_dog = 0
dog_dog = 0
for i in range(100):
    if dog_detector(human_files_short[i]):
        human_dog += 1
    if dog_detector(dog_files_short[i]):
        dog_dog += 1

print("The percentage of dog detected in human_files_short is: %.2f%%" % (100*human_dog/100))
print('The percentage of dog detected in dog_files_short is: %.2f%%' % (100*dog_dog/100))










