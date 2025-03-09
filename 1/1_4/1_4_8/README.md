[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 예측 값과 실제 값 출력해 보기

## 손 글씨 숫자 인식 예제 예측 값과 실제 값 출력해 보기
여기서는 예측 값과 실제 값을 출력해 보겠다.

다음과 같이 예제를 수정한다.
```py
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'x_train:{x_train.shape} y_train:{y_train.shape} x_test:{x_test.shape} y_test:{y_test.shape}')

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_train[0])
plt.show()

for y in range(28):
    for x in range(28):
        print(f'{x_train[0][y][x]:4}', end='')
    print()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape(60000, 784), x_test.reshape(10000, 784)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

p_test = model.predict(x_test)
print(f'p_test[0] : {p_test[0]}')

import numpy as np

print(f'p_test[0] : {np.argmax(p_test[0])}, y_test[0] : {y_test[0]}')
```
- import문을 이용하여 numpy 모듈을 np라는 이름으로 불러온다.  
여기서는 numpy 모듈의 argmax 함수를 이용하여 p_test[0] 항목의 가장 큰 값의 항목 번호를 출력한다.
- print 함수를 호출하여 p_test[0] 항목의 가장 큰 값의 항목 번호와 y_test[0] 항목이 가리키는 실제 라벨 값을 출력한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/85e6d283-bb46-4965-bc6d-ec8df628fa73)  
p_test[0] 항목의 가장 큰 값의 항목 번호와 y_test[0] 항목이 가리키는 실제 라벨 값이 같다.  
x_test[0] 항목의 경우 예측 값과 실제 값이 같아 인공 신경망이 옳게 예측한다.
