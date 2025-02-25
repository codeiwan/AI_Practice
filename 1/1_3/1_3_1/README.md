[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [인공 신경망과 근사 함수](./../README.md) | 2차 함수 근사해 보기

## 2차 함수 근사해 보기
여기서는 먼저 다음 2차 함수를 근사하는 인공 신경망 함수를 생성해 보자.  
![Image](https://github.com/user-attachments/assets/34719ff1-aab5-4bf8-b4a1-10fadf9fafd8) $y = 2x^2+3x+5$ $(-2 \leq x \leq 0.5)$

x 좌표의 범위는 -2에서 0.5까지입니다.
<br/><br/>
### 2차 함수 그리기
다음과 같이 예제를 작성해보자.  
```python
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(-2, 0.5, NUM_SAMPLES)
np.random.shuffle(xs)
print(xs[:5])

ys = 2*xs**2 + 3*xs + 5
print(ys[:5])

plt.plot(xs, ys, 'b.')
plt.show()
```
- import문을 이용하여 numpy 모듈을 np라는 이름으로 불러온다. numpy 모듈을 이용하여 x, y 값의 집합을 동시에 처리한다.
- import문을 이용하여 time 모듈을 불러온다. 임의 숫자(난수) 생성 초기화에 사용된다.
- import문을 이용하여 matplotlib.pyplot 모듈을 plt라는 이름으로 불러온다. 그래프를 그리는 데 사용된다.
- NUM_SAMPLES 변수를 생성한 후, 1000으로 초기화한다. NUM_SAMPLES 변수는 생성할 데이터의 개수 값을 가지는 변수이다.
- np.random.seed 함수를 호출하여 임의 숫자 생성을 초기화 한다.  
time.time 함수를 호출하여 현재 시간을 얻어낸 후, 정수 값으로 변환하여 np.random.seed 함수의 인자로 준다.  
이렇게 하면 현재 시간에 맞춰 임의 숫자 생성이 초기화 된다.
- np.random.uniform 함수를 호출하여 (-2, 0.5) 범위에서 NUM_SAMPLES 만큼의 임의 값을 차례대로 고르게 추출하여 xs 변수에 저장한다.
- np.random.shuffle 함수를 호출하여 임의 추출된 x 값을 섞어준다.  
이렇게 하면 임의로 추출된 x 값의 순서가 뒤섞이게 된다.  
인공 신경망 학습 시에 데이터는 임의 순서로 입력되는 것이 중요하다.  
데이터가 임의 순서로 입력될 때 모델의 정확도가 높아지기 때문이다.
- print 함수를 호출하여 xs에 저장된 값 중, 앞에서 5개까지 출력한다.  
xs[:5]는 xs 리스트의 0번 항목부터 시작해서 5번 항목 미만인 4번 항목까지를 의미한다.
- $y=2x^2+3x+5$ 식을 이용하여 추출된 x 값에 해당하는 y 값을 얻어내어 ys 변수에 저장한다.  
y 값도 NUM_SAMPLES 개수만큼 추출된다. 파이썬에서 *는 곱셈기호, **는 거듭제곱기호를 나타낸다.
- print 함수를 호출하여 ys에 저장된 값 중, 앞에서 5개까지 출력한다.
- plt.plot 함수를 호출하여 xs, ys 좌표 값에 맞추어 그래프를 내부적으로 그린다.  
그래프의 색깔은 파란색으로 그린다. 'b.'는 파란색을 의미한다.
- plt.show 함수를 호출하여 화면에 그래프를 표시한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/f1c20756-7fa3-43ae-9a82-be6bcca42ab4)  
첫째 줄은 xs에 저장된 값 중 앞에서 5개까지 출력 결과이다. 이 값은 실행할 때마다 달라진다.  
둘째 줄은 ys에 저장된 값 중 앞에서 5개까지 출력 결과이다. 이 값은 실행할 때마다 달라진다.  
$y=2x^2+3x+5$ 함수의 (-2, 0.5) 범위에서의 그래프이다.
<br/><br/>
### 실제 데이터 생성하기
이번엔 y값을 일정한 범위에서 위아래로 흩뜨려 실제 데이터에 가깝게 만들어 보자.  
이 과정은 y값에 잡음을 섞어 실제 데이터에 가깝게 만드는 과정이다.  

다음과 같이 예제를 수정하자.
```python
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(-2, 0.5, NUM_SAMPLES)
np.random.shuffle(xs)
print(xs[:5])

ys = 2*xs**2 + 3*xs + 5
print(ys[:5])

plt.plot(xs, ys, 'b.')
plt.show()

ys += 0.1*np.random.randn(NUM_SAMPLES)

plt.plot(xs, ys, 'g.')
plt.show()
```
- np.random.randn 함수를 호출하여 정규분포에 맞춰 임의 숫자를 NUM_SAMPLES의 개수만큼 생성한다.  
정규분포는 가우스분포라고도 하며, 종 모양과 같은 형태의 자연적인 분포 곡선이다.  
예를 들어, 키의 분포나 체중의 분호와 같이 자연적인 분포를 의미한다.  
생성된 숫자에 0.1을 곱해 ys에 더해준다.  
이렇게 하면 ys값은 워래 값을 기준으로 상하로 퍼진 형태의 자연스런 값을 갖게 된다.
- plt.plot 함수를 호출하여 xs, ys 좌표 값에 맞추어 그래프를 내부적으로 그린다.  
그래프의 색깔은 초록색으로 그린다. 'g.'은 초록색을 의미한다.
- plt.show 함수를 호출하여 화면에 그래프를 표시한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/09392e37-1a3b-4d98-92d0-c2bd5c6bf6d8)  
마지막에 표시된 그래프의 모양이 원래 모양에서 상하로 퍼진 형태로 나타나게 된다.  
여기서 생성된 데이터는 인공 신경망 학습에 사용되며 원래 곡선에 가까운 근사 곡선을 생성하는 인공 신경망 함수를 만들게 된다.
<br/><br/>
### 훈련, 실험 데이터 분리하기
여기서는 앞에서 생성한 x, y 데이터를 훈련 데이터와 실험 데이터로 분리해 보자.  
훈련 데이터는 인공 신경망을 학습시키는데 사용하는 데이터이며, 실험 데이터는 학습이 잘 되었는지 확인하는 데이터로 사용한다.

다음과 같이 예제를 수정하자.
```python
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(-2, 0.5, NUM_SAMPLES)
np.random.shuffle(xs)
print(xs[:5])

ys = 2*xs**2 + 3*xs + 5
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
```
- NUM_SAMPLES에 0.8을 곱한 후, 정수로 변경하여 NUM_SPLIT 변수에 할당한다.  
현재 예제의 경우 NUM_SPLIT 변수는 800의 값을 가진다.  
1000개의 x, y 데이터 값 중 800개는 훈련 데이터로, 200개는 실험 데이터로 사용한다.
- np.split 함수를 호출하여 1000개의 값을 가진 xs를 800개, 200개로 나누어 각각 x_train, x_test에 할당한다. x_train 변수는 1000개의 값 중 앞부분 800개의 값을 할당 받고 x_test 변수는 나머지 200개의 값을 할당받는다.
- np.split 함수를 호출하여 1000개의 값을 가진 ys를 800개, 200개로 나누어 각각 y_train, y_test에 할당한다. y_train 변수는 1000개의 값 중 앞부분 800개의 값을 할당 받고 y_test 변수는 나머지 200개의 값을 할당받는다.
- plt.plot 함수를 호출하여 x_train, y_train 좌표 값에 맞추어 그래프를 내부적으로 그린다.  
그래프의 색깔은 파란색으로 그린다. 'b.'은 파란색을 의미한다.  
label 매개변수에는 'train' 문자열을 넘겨준다.  
이 문자열은 plt.legend 함수에 의해 그래프에 표시된다.
- plt.plot 함수를 호출하여 x_test, y_test 좌표 값에 맞추어 그래프를 내부적으로 그린다.  
그래프의 색깔은 빨간색으로 그린다. 'r.'은 파란색을 의미한다.  
label 매개변수에는 'test' 문자열을 넘겨준다.  
이 문자열은 plt.legend 함수에 의해 그래프에 표시된다.
- plt.legend 함수를 호출하여 범례를 표시한다.
- plt.show 함수를 호출하여 화면에 그래프를 표시한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/39006c2d-7280-4ecc-9710-552798faffd4)  
파란색 점은 x_train, y_train의 분포를 나타내며, 빨간색 점은 x_test, y_test의 분포를 나타낸다.  
x_train, y_train 데이터는 인공 신경망 학습에 사용되며 원래 곡선에 가까운 근사 곡선을 생성하는 인공 신경망 함수를 만들게 된다.  
x_test, y_test 데이터는 학습이 끝난 인공 신경망 함수를 시험하는데 사용된다.
<br/><br/>
### 인공 신경망 구성하기
이번엔 인공 신경망 함수를 구성한 후, 학습을 수행하지 않은 상태로 시험 데이터를 이용하여 예측을 수행한 후, 그래프를 그려보자.  
여기서는 다음과 같은 모양의 인공 신경망을 구성한다.  
입력 층 xs, 출력 층 ys 사이에 단위 인공 신경 16개로 구성된 은닉 층 2개를 추가하여 인공 신경망을 구성한다.  
![Image](https://github.com/user-attachments/assets/845e7e69-3923-4192-86a7-1f9c28ab5512)

다음과 같이 예제를 수정하자.
```python
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(-2, 0.5, NUM_SAMPLES)
np.random.shuffle(xs)
print(xs[:5])

ys = 2*xs**2 + 3*xs + 5
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
```
- import문을 이용하여 tensorflow 모듈을 if라는 이름으로 불러온다.  
tensorflow 모듈은 구글에서 제공하는 인공 신경망 라이브러리이다.
- tf.keras.Sequential 클래스를 이용하여 인공 신경망을 생성한다.  
여기서 생성한 인공 신경망은 33(=16+16+1)개의 인공 신경망으로 구성된다.  
model_f는 모델 함수를 의미하는 변수이다.
- tf.keras.layers.InputLayer 함수를 이용하여 내부적으로 keras 라이브러리에서 제공하는 tensor를 생성하고, 입력 노드의 개수를 정해준다.  
tensor는 3차원 이상의 행렬을 의미하며, 인공 신경망 구성 시 사용하는 자료 형이다.
- tf.keras.layers.Dense 클래스를 이용하여 신경망 층을 생성한다.  
여기서는 각 층별로 단위 인공 신경 16개를 생성한다.  
activation은 활성화 함수를 의미하며 여기서는 'relu' 함수를 사용한다.  
다음은 relu 함수를 나타낸다.  
![Image](https://github.com/user-attachments/assets/48f6ac8b-98ae-4209-afbf-c970ea1496a6)  
활성화 함수와 'relu' 함수에 대해서는 이후 자세히 살펴보겠다.  
여기서 Dense는 내부적으로 y=activation(x*w + b) 식을 생성하게 된다.  
이 식에 대해서는 실제로 구현해 보며 그 원리를 살보도록 하겠다.
- tf.keras.layer.Dense 클래스를 이용하여 신경망 층을 생성한다.  
여기서는 단위 인공 신경 1개를 생성한다.  
마지막에 생성한 신경망 층은 출력 신경망이 된다.
- model_f.compile 함수를 호출하여 내부적으로 인공 신경망을 구성한다.  
인공 신경망을 구성할 때에는 적으도 2개의 함수를 정해야 한다.  
loss 함수와 optimizer 함수, 즉, 손실 함수와 최적화 함수를 정해야 한다.  
손실 함수와 최적화 함수에 대해서는 이후 자세히 살펴보겠다.  
손실 함수로는 mse 함수를 사용하고 최적화 함수는 rmsprop 함수를 사용한다.  
mse, rmsprop 함수는 이후 자세히 살펴보겠다.
- model_f.predict 함수를 호출하여 인공 신경망을 사용해 보자.  
여기서는 학습을 수행하지 않은 상태에서 인공 신경망 함수에 x_test 값을 주어 그 결과를 예측 할 것이다.  
예측한 결과 값은 p_test 변수로 받는다.
- plt.plot 함수를 호출하여 x_test, y_test 좌표 값에 맞추어 그래프를 내부적으로 그린다.  
그래프의 색깔은 파란색으로 그린다. 'b.'은 파란색을 의미한다.  
label 매개변수에는 'actual' 문자열을 넘겨준다.  
이 문자열은 plt.legend 함수에 의해 그래프에 표시된다.
- plt.plot 함수를 호출하여 x_test, p_test 좌표 값에 맞추어 그래프를 내부적으로 그린다.  
그래프의 색깔은 빨간색으로 그린다. 'r.'은 빨간색을 의미한다.  
label 매개변수에는 'predicted' 문자열을 넘겨준다.  
이 문자열은 plt.legend 함수에 의해 그래프에 표시된다.
- plt.legend 함수를 호출하여 범례를 표시한다.
- plt.show 함수를 호출하여 화면에 그래프를 표시한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/105cc74d-8204-4eae-b2fd-a035f154e6c8)  
파란색 점은 x_test, y_test의 분포를 나타내며, 빨간색 점은 x_test, p_test의 분포를 나타낸다.  
인공 신경망이 학습을 수행하기 전 상태라 x_test 값에 대한 예측 값을 정확히 생성해 내지 못하는 것을 볼 수 있다.
<br/><br/>
### 인공 신경망 학습시키기

다음과 같이 예제를 수정하자.
```python
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(-2, 0.5, NUM_SAMPLES)
np.random.shuffle(xs)
print(xs[:5])

ys = 2*xs**2 + 3*xs + 5
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
- model_f.fit 함수를 호출하여 인공 신경망에 대한 학습을 시작한다.  
fit 함수에는 x_train, y_train 데이터가 입력이 되는데 인공 신경망을 x_train, y_train 데이터에 맞도록 학습한다는 의미를 갖는다.  
즉, x_train, y_train 데이터에 맞도록 인공 신경망을 학습한다는 의미이다.  
fit 함수에는 학습을 몇 회 수행할지도 입력해준다.  
epochs는 학습 횟수를 의미하며, 여기서는 600회 학습을 수행하도록 한다.  
일반적으로 학습 횟수에 따라 인공 신경망 근사 함수가 정확해진다.
- model_f.predict 함수를 호출하여 인공 신경망을 사용한다.  
여기서는 학습이 끝난 인공 신경망 함수에 x_test 값을 주어 그 결과를 예측한다.  
예측한 결과 값은 p_test 변수로 받는다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/839fa369-6cdd-44a1-8bd5-23ff604284de)

파란색 점은은 x_test, y_test의 분포를 나타내며, 빨간색 점은 x_test, p_test의 분포를 나타낸다.  
인공 신경망이 학습을 수행한 이후에는 x_test 값에 대한 예측 값을 실제 함수에 근사해서 생성해 내는 것을 볼 수 있다.
