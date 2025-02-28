[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [인공 신경망과 근사 함수](./../README.md) | 인공 신경망 소스 살펴보기

## 인공 신경망 소스 살펴보기
다음은 지금까지 실습한 인공 신경망 관련 루틴을 정리한 내용이다.
```python
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(0.1, 5, NUM_SAMPLES)
np.random.shuffle(xs)

ys = 1.0/xs
ys += 0.1*np.random.randn(NUM_SAMPLES)

NUM_SPLIT = int(0.8*NUM_SAMPLES)

x_train, x_test = np.split(xs, [NUM_SPLIT])
y_train, y_test = np.split(ys, [NUM_SPLIT])

import tensorflow as tf

model_f = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_f.compile(optimizer='sgd', loss='mean_squared_error')

model_f.fit(x_train, y_train, epochs=600)

p_test = model_f.predict(x_test)

plt.plot(x_test, y_test, 'b.', label='actual')
plt.plot(x_test, p_test, 'r.', label='predicted')
plt.legend()
plt.show()
```
- 인공 신경망 학습에 사용할 데이터를 생성한다.
- 데이터를 훈련 데이터와 실험 데이터로 나눈다.
- 인공 신경망 구성에 필요한 입력 층, 은닉 층, 출력 등을 구성한다.
- 인공 신경망 내부 망을 구성하고, 학습에 필요한 오차함수, 최적화함수를 설정한다.
- 인공 신경망을 학습시킨다.
- 학습시킨 인공 신경망을 이용하여 새로 들어온 데이터에 대한 예측을 수행한다.
