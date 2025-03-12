[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 잘못된 예측 살펴보기

## 손 글씨 숫자 인식 예제 잘못된 예측 살펴보기
여기서는 시험 데이터 중 잘못된 예측이 몇 개나 되는지 또 몇 번째 그림이 잘못 예측되었는지 살펴 보도록 하겠다.

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
```
- cnt_wrong 변수를 선언한 후, 0으로 초기화 한다.  
cnt_wrong 변수는 잘못 예측된 그림의 개수를 저장할 변수이다.
- p_wrong 변수를 선언한 후, 빈 리스트로 초기화한다.  
p_wrong 변수는 잘못 예측된 그림의 번호를 저장할 변수이다.
- 0부터 10000미만까지 p_test[i] 항목의 가장 큰 값의 항목 번호와 y_test[0] 항목이 가리키는 실제 라벨 값이 다르면  
p_wrong 리스트에 해당 그림을 추가하고  
cnt_wrong 값을 하나 증가시킨다.
- print 함수를 호출하여 cnt_wrong 값을 출력한다.
- print 함수를 호출하여 p_wrong에 저장된 값 중, 앞에서 10개까지 출력한다.  
p_wrong[:10]은 p_wrong 리스트의 0번 항목부터 시작해서 10번 항목 미만인 9번 항목까지를 의미한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/472d893f-07a6-465e-b0b8-ad40805bf67e)  
학습이 끝난 인공 신경망은 시험 데이터에 대해 10000개 중 244개에 대해 잘못된 예측을 하였다.  
즉, 10000개 중 9756(=10000-244)개는 바르게 예측을 했으며, 나머지 244개에 대해서는 잘못된 예측을 하였다.  
예측 정확도는 97.56%, 예측 오류도는 2.44%이다.  
잘못 예측한 데이터 번호 10개에 대해서도 확인해보자.  
62번 데이터로 시작해서 613번 데이터까지 10개의 데이터 번호를 출력한다.
