

'''
一.
数据预处理
通过对每张图像的像素值除以255，我们对图像实现了归一化处理。
'''
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255　
'''
#输出：
100%|██████████| 6680/6680 [02:18<00:00, 48.39it/s]
100%|██████████| 835/835 [00:15<00:00, 55.55it/s]
100%|██████████| 836/836 [00:15<00:00, 54.15it/s]
'''

# 二
#  模型架构
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: 定义你的网络架构
model.add(Conv2D(filters=16, kernel_size=2, padding='same', 
                 activation='relu', input_shape=(224, 224, 3))) 
model.add(MaxPooling2D(pool_size=2)) # strides默认为pool_size
model.add(Conv2D(filters=32, kernel_size=2, padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(GlobalAveragePooling2D()) 
model.add(Dense(133, activation='softmax'))
                 
model.summary()

# 三
# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 四
# 训练模型
from keras.callbacks import ModelCheckpoint  
#设置训练模型的epochs的数量
epochs = 5
#使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

# 五
# 加载具有最好验证loss的模型
model.load_weights('saved_models/weights.best.from_scratch.hdf5')

# 六
# 测试模型

# 获取测试数据集中每一个图像所预测的狗品种的index
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)







