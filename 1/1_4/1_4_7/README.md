[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 학습된 인공 신경망 시험하기

## 손 글씨 숫자 인식 예제 학습된 인공 신경망 시험하기
여기서는 학습된 신경망에 시험 데이터를 입력하여 예측해 보겠다.
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
```
- model.predict 함수를 호출하여 인공 신경망을 시험한다.  
여기서는 학습이 끝난 인공 신경망 함수에 x_test 값을 주어 그 결과를 예측해 보겠다.  
예측한 결과 값은 p_test 변수로 받는다.  
x_test는 1만개의 손 글씨 숫자를 가리키고 있으며, 따라서 1만개에 대한 예측을 수행한다.
- print 함수를 호출하여 x_test[0] 손 글씨 숫자의 예측 값을 출력한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/1f2e7694-e3e0-4a14-889a-72c849cc1d92)  
p_test[0]은 x_test[0]이 가리키는 손 글씨 숫자에 대해 0~9 각각의 숫자에 대한 확률값 리스트를 출력한다.  
x_test[0]은 실제로 숫자 7을 가리키고 있다.  
그래서 p_test[0]의 8번째 값의 확률이 가낭 높게 나타난다.  
8번째 값은 9.99882400e-01이며 99.9%로 7이라고 예측한다.  
p_test[0]의 1번째 값은 숫자 0일 확률을 나타낸다.
