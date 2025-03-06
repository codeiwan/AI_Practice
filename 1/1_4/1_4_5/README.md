[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 학습 데이터 그림 그려보기 2

## 손 글씨 숫자 인식 예제 학습 데이터 그림 그려보기 2
여기서는 학습 데이터의 그림 25개를 화면에 출력해 보겠다.

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
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()
```
- plt.figure 함수를 호출하여 새로운 그림을 만들 준비를 한다.  
figure 함수는 내부적으로 그림을 만들고 편집할 수 있게 해 주는 함수이다.  
figsize는 그림의 인치 단위의 크기를 나타낸다.  
여기서는 가로 10인치, 세로 10인치의 그림을 그린다는 의미이다.
- 0에서 24에 대해 plt.subplot 함수를 호출하여 그림 창을 분할하여 하위 그림을 그린다.  
5, 5는 각각 행의 개수와 열의 개수를 의미한다.  
i + 1은 하위 그림의 위치를 나타낸다.
- plt.xtics, plt.ytics 함수를 호출하여 x, y 축 눈금을 설정한다.  
여기서는 빈 리스트를 주어 눈금 표시를 하지 않는다.
- plt.imshow 함수를 호출하여 x_train[i] 항목의 그림을 내부적으로 그린다.  
cmap은 color map의 약자로 binary는 그림을 이진화해서 표현해 준다.
- plt.xlabel 함수를 호출하여 x 축에 라벨을 붙여준다.  
라벨의 값은 y_train[i]이다.
- plt.show 함수를 호출하여 내부적으로 그린 그림을 화면에 그린다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/7bea2097-67c4-4f53-83d5-701151f01ca3)  
x_train 변수가 가리키는 손 글씨 숫자 그림 25개를 볼 수 있다.  
x_train 변수는 이런 그림을 6만개를 가키리고 있다.
