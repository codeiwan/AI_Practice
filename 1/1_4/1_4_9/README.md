[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 시험 데이터 그림 그려보기

## 손 글씨 숫자 인식 예제 시험 데이터 그림 그려보기
여기서는 시험용 데이터의 그림을 화면에 출력해 보겠다.

다음과 같이 예제를 수정한다.
```python
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

x_test = x_test.reshape(10000, 28, 28)

plt.figure()
plt.imshow(x_test[0])
plt.show()
```
- reshape 함수를 호출하여 x_test가 가리키는 그림을 원래 모양으로 돌려놓는다.  
그래야 pyplot 모듈을 이용하여 그림을 화면에 표시할 수 있다.
- plt.figure 함수를 호출하여 새로운 그림을 만들 준비를 한다.  
figure 함수는 내부적으로 그림을 만들고 편집할 수 있게 해 주는 함수이다.
- plt.imshow 함수를 호출하여 x_test[0] 항목의 그림을 내부적으로 그린다.
- plt.show 함수를 호출하여 내부적으로 그린 그림을 화면에 그린다.


다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/6b649a3d-7e9e-49a8-b3b6-94863df73610)  
x_test[0] 항목의 손 글씨 숫자 그림은 7이다.
