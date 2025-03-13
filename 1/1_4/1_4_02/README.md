[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 데이터 모양 살펴보기

## 손 글씨 숫자 인식 예제 데이터 모양 살펴보기
![Image](https://github.com/user-attachments/assets/1066525f-5243-4b35-86ce-bc027557964d)  
앞서 전체적으로 실행해 본 예제를 앞으로 단계별로 살펴볼 것이다.  
즉, 입력 데이터의 모양도 살펴보고 학습을 수행한 후, 학습에 사용하지 않은 손 글씨 숫자를 얼마나 잘 인식하는지 살펴 볼 것이다.  
또 잘못 인식한 숫자를 직접 확인하며, 인공 신경망의 인식 성능을 확인해 볼 것이다.

여기서는 먼저 MNIST 손 글씨 숫자 데이터의 개수와 형식을 살펴보자.

다음과 같이 예제를 작성한다.
```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'x_train:{x_train.shape} y_train:{y_train.shape} x_test:{x_test.shape} y_test:{y_test.shape}')
```
- import문을 이용하여 tensorflow 모듈을 tf라는 이름으로 불러온다.  
tensorflow 모듈은 구글에서 제공하는 인공 신경망 라이브러리 이다.
- mnist 변수를 생성한 후, tf.keras.datasets.mnist 모듈을 가리키게 한다.  
mnist 모듈은 손 글씨 숫자 데이터를 가진 모듈이다.  
mnist 모듈에는 6만개의 학습용 손 글씨 숫자 데이터와 1만개의 시험용 손 글씨 숫자 데이터가 있다.
- mnist.load_data 함수를 호출하여 손 글씨 숫자 데이터를 읽어와 x_train, y_train, x_test, y_test 변수가 가리키게 한다.
- print 함수를 호출하여 x_train, y_train, x_test, y_test의 데이터 모양을 출력한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/533a079f-bd53-49b7-b42e-01f652c08d72)  
x_train, y_train 변수는 각각 6만개의 학습용 손 글씨 숫자 데이터와 숫자 라벨을 가리킨다.  
x_test, y_test 변수는 각각 1만개의 시험용 손 글씨 숫자 데이터와 숫자 라벨을 가리킨다.  
x_train, x_test 변수가 가리키는 6만개, 1만개의 그림은 각각 28x28 픽셀, 28x28 픽셀로 구성되어 있다.
