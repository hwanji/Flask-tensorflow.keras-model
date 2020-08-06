# -*- coding: utf-8 -*-
"""
如何用Flask部署Keras深度学习模型
https://zhuanlan.zhihu.com/p/47349497

"""

import pandas as pd
from tensorflow.keras import models, layers

class_count = 2 #类别数量

# 加载样本数据集，划分为x和y DataFrame
df = pd.read_csv("games-expand.csv")
x = df.drop(['label'], axis=1)
y = df['label']

# 定义Keras模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10,)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(class_count, activation='softmax'))

model.summary()
# 编译并拟合模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x, y, epochs=100, batch_size=100,
                    validation_split = 0.2, verbose=1)

# 以H5格式保存模型
model.save("games.h5")




