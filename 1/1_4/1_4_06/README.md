[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 인공 신경망 학습시키기

## 손 글씨 숫자 인식 예제 인공 신경망 학습시키기
여기서는 이전 예제에서 살펴본 손 글씨 숫자 데이터를 이용하여 인공 신경망을 학습시켜 보겠다.  
인공 신경망은 앞에서 구성했던 신경망을 그대로 사용한다.

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
```
- x_train, x_test 변수가 가리키는 6만개, 1만개의 그림은 각각 28x28 픽셀로 구성된 그림이며, 1픽셀의 크기는 8비트로 0에서 255사이의 숫자를 가진다.  
모든 픽셀의 숫자를 255.0으로 나누어 각 픽셀을 0.0에서 1.0사이의 실수로 바꾸어 인공 신경망에 입력하게 된다.  
다음은 x_train[0] 그림의 픽셀 값을 출력한 그림이다.  
![Image](https://github.com/user-attachments/assets/2a43b16d-530d-4400-a1dc-84f0f5be947f)  
- x_train, x_test 변수가 가리키는 6만개, 1만개의 그림은 각각 28x28 픽셀, 28x28 픽셀로 구성되어 있다.  
이 예제의 인공 신경망의 경우 그림 데이터를 입력할 때 28x28 픽셀을 784(=28x28) 픽셀로 일렬로 세워서 입력하게 된다.  
그래서 33줄에 있는 InputLayer 클래스는 일렬로 세워진 784 픽셀을 입력으로 받도록 구성된다.  
![Image](https://github.com/user-attachments/assets/5f510769-136c-43d3-9fe7-59de9de49e1f)  
(출처: https://www.kdnuggets.com/2019/11/designing-neural-networks.html)
- tf.keras.Sequential 클래스를 이용하여 인공 신경망을 생성한다.  
여기서 생성한 인공 신경망은 138(=128+10)개의 인공 신경망으로 구성된다.  
입력 층에 표시된 노드는 입력 값의 개수를 표시하며 나머지 층에 있는 노드는 인공 신경을 나타낸다.  
인공 신경망의 내부 구조는 이후 자세히 살펴보겠다.  
생성된 인공 신경망은 일반적으로 모델이라고 한다.  
모델은 모형을 의미하며, 주어진 데이터에 맞추어진 원래 함수를 흉내 내는 함수인 근사 함수를 의미한다.  
여기서는 손 글씨 숫자를 구분하는 근사함수이다.
- tf.keras.layers.InputLayer 함수를 이용하여 내부적으로 keras 라이브러리에서 제공하는 tensor를 생성하고, 입력 노드의 개수를 정해준다.  
tensor는 3차원 이상의 행렬을 의미하며, 인공 신경망 구성 시 사용하는 자료 형이다.  
여기서는 784개의 입력 노드를 생성한다.
- tf.keras.layers.Dense 클래스를 이용하여 신경망 층을 생성한다.  
여기서는 단위 인공 신경 128개를 생성한다.  
activation은 활성화 함수를 의미하며 여기서는 'relu' 함수를 사용한다.  
다음은 relu 함수를 나타낸다.  
![Image](https://github.com/user-attachments/assets/89adb7d5-9b93-4fd9-ade3-a5f437fefbdd)  
활성화 함수와 'relu' 함수에 대해서는 이후 직접 구현해 보면서 자세히 살펴보겠다.  
여기서 Dense는 내부적으로 y = activation(x*w + b) 식을 생성하게 된다.  
이 식에 대해서는 뒤에서 실제로 구현해 보며 그 원리를 살펴보도록 할 것이다.
- tf.keras.layers.Dense 클래스를 이용하여 신경망 층을 생성한다.  
여기서는 단위 인공 신경 10개를 생성한다.  
activation은 활성화 함수를 의미하며 여기서는 'softmax' 함수를 사용한다.  
다음은 softmax 함수를 나타낸다.  
![Image](https://github.com/user-attachments/assets/fc700c53-8666-4d7a-bced-6506eaee5cd8)  
참고로 'softmax' 함수는 출력 층에서만 사용할 수 있는 활성화 함수이다.  
활성화 함수와 'softmax' 함수에 대해서는 추후 구현해 보면서 자세히 살펴보도록 할 것이다.
- model.compile 함수를 호출하여 내부적으로 인공 신경망을 구성한다.  
인공 신경망을 구성할 때에는 적어도 2개의 함수를 정해야 한다.  
loss 함수와 optimizer 함수 즉, 손실 함수와 최적화 함수를 정해야 한다.  
손실 함수와 최적화 함수에 대해서는 추후 자세히 살펴볼 것이다.  
손실 함수로는 sparse_categorical_crossentropy 함수를 사용하고, 최적화 함수는 adam 함수를 사용한다.  
sparse_categorical_crossentropy, adam 함수는 추후 살펴보도록 할 것이다.  
fit 함수 로그에는 기본적으로 손실 값만 표시된다.  
metrics 매개 변수는 학습 측정 항목 함수를 전달할 때 사용한다.  
'accuracy'는 학습의 정확도를 출력해 준다.
- model.fit 함수를 호출하여 인공 신경망에 대한 학습을 시작한다.  
fit 함수에는 x_train, y_train 데이터가 입력이 되는데 인공 신경망을 x_train, y_train 데이터에 맞도록 학습한다는 의미를 갖는다.  
fit 함수에는 학습을 몇 회 수행할지도 입력해 준다.  
epochs는 학습 횟수를 의미하며, 여기서는 5회 학습을 수행하도록 한다.  
일반적으로 학습 횟수에 따라 인공 신경망 근사 함수가 정확해진다.
- model.evaluate 함수를 호출하여 인공 신경망의 학습 결과를 평가한다.  
여기서는 학습이 끝난 인공 신경망 함수에 x_test 값을 주어 학습 결과를 평가한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/b7cad4c2-269e-4802-aa12-a0186102d76c)  
- loss는 손실 함수에 의해 측정된 오차 값을 나타낸다.  
학습 횟수가 늘어남에 따라 오차 값이 줄어든다.
- accuracy에 학습 진행에 따른 정확도가 표시된다.  
처음에 87.82%에서 시작해서 마지막엔 98.66%의 정확도로 학습이 끝난다.  
즉, 100개의 손 글씨가 있다면 98.66개를 맞춘다는 의미이다.
- 학습이 끝난 후에, evalueate 함수로 시험 데이터를 평가한 결과가 나온다.  
학습 데이터의 예측 결과에 비해 시험 데이터의 예측 결과에서는 손실 값이 늘어났고, 정확도가 97.05%로 약간 떨어진 상태이다.
- 마지막으로 평가한 결과가 표시된다.
