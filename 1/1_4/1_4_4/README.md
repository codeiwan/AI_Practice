[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [딥러닝 활용 맛보기](./../README.md) | 손 글씨 숫자 인식 예제 그림 픽셀 값 출력해 보기

## 손 글씨 숫자 인식 예제 그림 픽셀 값 출력해 보기
여기서는 앞서 출력한 그림의 픽셀 값을 출력해 보겠다.

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
```
- 그림의 세로 28 줄에 대한 가로 28 픽셀에 대해 각 픽셀 값을 출력한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/7717e54b-181c-4de2-9f97-da94f19bebd5)  
각 픽셀의 값이 0~255 사이의 값에서 출력되는 것을 확인한다.
