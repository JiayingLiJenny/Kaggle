

# 从另一个预训练的CNN获取bottleneck特征
bottleneck_features = np.load('bottleneck_features/DogResNet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']


ResNet50_model = Sequential()
ResNet50_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet50_model.add(Dense(133, activation='softmax'))
ResNet50_model.summary()


'''
print(train_ResNet50.shape[1:])
(1, 1, 2048)
'''

# 三
# 编译模型
ResNet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# 四
## 训练模型
epochs = 20
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.ResNet50.hdf5',
                               verbose=1, save_best_only=True)
ResNet50_model.fit(train_ResNet50, train_targets,
                   validation_data=(valid_ResNet50, valid_targets),
                epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

# 五
# 加载具有最好验证loss的模型的权重
ResNet50_model.load_weights('saved_models/weights.best.ResNet50.hdf5')

# 六
# 测试模型
ResNet50_predictions = [np.argmax(ResNet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_ResNet50]
test_accuracy = 100*np.sum(np.array(ResNet50_predictions)==np.argmax(test_targets, axis=1))/len(ResNet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

# 七
# 使用模型预测狗品种

## 该函数将图像的路径作为输入
## 然后返回此模型所预测的狗的品种
from extract_bottleneck_features import extract_Resnet50
def Renet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = ResNet50_model.predict(bottleneck_feature)
    return(dog_names[np.argmax(predicted_vector)])


# 八

'''
它的输入为图像的路径，它能够区分图像是否包含一个人、狗或两者都不包含，然后：

如果从图像中检测到一只狗，返回被预测的品种。
如果从图像中检测到人，返回最相像的狗品种。
如果两者都不能在图像中检测到，输出错误提示。
'''














