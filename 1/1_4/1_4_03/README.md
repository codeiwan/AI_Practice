[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 학습 데이터 그림 그려보기

## 손 글씨 숫자 인식 예제 학습 데이터 그림 그려보기
여기서는 학습용 데이터의 그림을 화면에 출력해 보겠다.

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
```
- import문을 이용하여 matplotlib.pyplot 모듈을 plt라는 이름으로 불러온다.  
여기서는 matplotlib.pyplot 모듈을 이용하여 그래프를 그린다.
- plt.figure 함수를 호출하여 새로운 그림을 만들 준비를 한다.  
figure 함수는 내부적으로 그림을 만들고 편집할 수 있게 해 주는 함수이다.
- plt.imshow 함수를 호출하여 x_train[0] 항목의 그림을 내부적으로 그린다.
- plt.show 함수를 호출하여 내부적으로 그린 그림을 화면에 그린다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/ab11ce8f-2757-49d2-86ed-62b74aa04634)  
x_train[0] 항목의 손 글씨 숫자 그림은 5이다.
