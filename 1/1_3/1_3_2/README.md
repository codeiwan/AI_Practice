[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [인공 신경망과 근사 함수](./../README.md) | 5차 함수 근사해 보기

## 5차 함수 근사해 보기
이번에는 다음과 같은 5차 함수를 근사하도록 인공 신경망 함수를 학습시켜 보자.  
![Image](https://github.com/user-attachments/assets/c99f2453-bcdd-49be-89ec-33d1801cdc1e)  
$y=(x+1.7)(x+0.7)(x-0.3)(x-1.3)(x-1.9)+0.2$ 　　　 $(-2 \leq x \leq 2)$  

x 좌표의 범위는 -2에서 2까지이다.

이전 예제를 수정하여 다음과 같이 작성한다.
```python
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(-2, 2, NUM_SAMPLES)
np.random.shuffle(xs)
print(xs[:5])

ys = (xs + 1.7)*(xs + 0.7)*(xs - 0.3)*(xs - 1.3)*(xs - 1.9) + 0.2
print(ys[:5])

plt.plot(xs, ys, 'b.')
plt.show()

ys += 0.1*np.random.randn(NUM_SAMPLES)

plt.plot(xs, ys, 'g.')
plt.show()

NUM_SPLIT = int(0.8*NUM_SAMPLES)

x_train, x_test = np.split(xs, [NUM_SPLIT])
y_train, y_test = np.split(ys, [NUM_SPLIT])

plt.plot(x_train, y_train, 'b.', label='train')
plt.plot(x_test, y_test, 'r.', label="test")
plt.legend()
plt.show()

import tensorflow as tf

model_f = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_f.compile(optimizer='rmsprop', loss='mse')

p_test = model_f.predict(x_test)

plt.plot(x_test, y_test, 'b.', label='actual')
plt.plot(x_test, p_test, 'r.', label='predicted')
plt.legend()
plt.show()

model_f.fit(x_train, y_train, epochs=600)

p_test = model_f.predict(x_test)

plt.plot(x_test, y_test, 'b.', label='actual')
plt.plot(x_test, p_test, 'r.', label='predicted')
plt.legend()
plt.show()
```
- np.random.uniform 함수를 호출하여 (-2, 2) 범위에서 NUM_SAMPLES 만큼의 임의 값을 차례대로 고르게 추출하여 xs 변수에 저장한다.
- $y=(x+1.7)(x+0.7)(x-0.3)(x-1.3)(x-1.9)+0.2$ 식을 이용하여 추출된 x 값에 해당하는 y 값을 얻어내어 ys 변수에 저장한다.  
y 값도 NUM_SAMPLES 개수만큼 추출된다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/36ece390-4284-40e3-9104-f21bf3a4ac2b)  
![Image](https://github.com/user-attachments/assets/61c9c226-0fb1-4dd4-89ae-84c25101fe83)  
![Image](https://github.com/user-attachments/assets/5e8b6a31-beb5-4f6c-9dfc-9ce8d5dbd5a4)  
![Image](https://github.com/user-attachments/assets/da01e851-cc90-48ed-85a8-f402964a7782)  
![Image](https://github.com/user-attachments/assets/d336dbf6-b03d-47f4-a255-098dfc6e580c)

인공 신경망이 학습을 수행한 이후에는 x_test 값에 대한 예측 값을 실제 함수에 근사해서 생성해 내는 것을 볼 수 있다.
