[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 잘못된 예측한 그림 살펴보기

## 손 글씨 숫자 인식 예제 잘못된 예측한 그림 살펴보기
여기서는 시험 데이터 중 잘못 예측한 그림을 출력하겠다.

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

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(p_test[i]))
plt.show()


cnt_wrong = 0
p_wrong = []
for i in range(10000):
    if np.argmax(p_test[i]) != y_test[i]:
        p_wrong.append(i)
        cnt_wrong += 1

print(f'cnt_wrong : {cnt_wrong}')
print(f'predicted wrong 10 : {p_wrong[:10]}')

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[p_wrong[i]], cmap=plt.cm.binary)
    plt.xlabel(f'{p_wrong[i]} : p{np.argmax(p_test[p_wrong[i]])} y{y_test[p_wrong[i]]}')
plt.show()
```
- plt.figure 함수를 호출하여 새로운 그림을 만들 준비를 한다.  
figure 함수는 내부적으로 그림을 만들고 편집할 수 있게 해 주는 함수이다.  
figsize는 그림의 인치 단위의 크기를 나타낸다.  
여기서는 가로 10인치, 세로 10인치의 그림을 그린다는 의미이다.
- 0에서 24에 대해 plt.subplot 함수를 호출하여 그림 창을 분할하여 하위 그림을 그린다.
- 5, 5는 각각 행의 개수와 열의 개수를 의미한다.  
i+1은 하위 그림의 위치를 나타낸다.
- plt.xticks, plt.yticks 함수를 호출하여 x, y 축 눈금을 설정한다.  
여기서는 빈 리스트를 주어 눈금 표시를 하지 않는다.
- plt.imshow 함수를 호출하여 x_test[p_wrong[i]] 항목의 그림을 내부적으로 그린다.  
cmap은 color map의 약자로 binary는 그림을 이진화해서 표현해 준다.
- plt.xlabel 함수를 호출하여 x 축에 라벨을 붙여준다. 라벨의 값은 잘못 예측한 그림 번호, 인공 신경망이 예측한 숫자 값, 라벨에 표시된 숫자 값으로 구성된다.
- plt.show 함수를 호출하여 내부적으로 그린 그림을 화면에 그린다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/0933a212-2b75-44bf-9a5d-e475f7e0b07c)  
이번 경우 첫 번째 잘못 예측한 숫자는 149번째 숫자이며, 인공 신경망은 9로 예측하였으며, 실제 값은 2이다.  
언뜻 보면 사람이 보기에도 혼동될 수 있는 형태의 숫자들이다.

이상으로 손 글씨 숫자를 이용하여 인공 신경망을 학습시키고, 학습시킨 결과를 예측하는 과정을 자세히 살펴보았다.  
일반적으로 인공 신경망을 이용할 때, 달라지는 부분은 데이터의 종류와 인공 신경망의 구성 형태이다.  
나머지 부분은 프레임워크처럼 비슷한 형태로 작성될 가능성이 높다.
