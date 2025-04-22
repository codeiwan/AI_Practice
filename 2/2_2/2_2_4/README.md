[Home](./../../../README.md) | [인공 지능의 딥러닝 알고리즘](./../../README.md) | [다양한 인공 신경망 구현해 보기](./../README.md) | 2입력 2은닉 2출력 인공 신경망 구현하기

## 2입력 2은닉 2출력 인공 신경망 구현하기
여기서는 은닉 신경을 추가한 인공 신경망을 구현해 보겠다.  
은닉 신경이 추가된 경우에도 순전파, 역전파 수식을 구하는 방식은 이전과 같다.  
다음 그림은 입력2 은닉2 출력2로 구성된 인공 신경망을 나타낸다.

![Image](https://github.com/user-attachments/assets/35b0e5fc-5a1a-4b78-b5b5-2e74be0893c0)

[그림1]은 순전파 과정에 필요한 변수와 수식을 나타낸다.  
[그림2]는 역전파에 필요한 변수이다. 순전파에 대응되는 변수가 모두 필요하다.  
[그림3]은 입력의 역전파에 필요한 변수와 수식을 나타낸다.  
[그림4]는 가중치와 편향의 역전파에 필요한 변수와 수식을 나타낸다.  
(※ i1b, i2b값은 앞부분에 또 다른 인공 신경와 연결되어 있을 경우 h1b, h2b처럼 해당 인공 신경으로 역전파 되는 값이다.  
역전파된 i1b, i2b값은 해당 인공 신경의 가중치와 편향 학습에 사용된다.  
여기서 i1, i2는 은닉 층에 연결된 입력 층이므로 i1b, i2b의 수식은 필요치 않다.)

이상에서 필요한 수식을 정리하면 다음과 같다.  
![Image](https://github.com/user-attachments/assets/09320a51-fbd4-434d-8117-d9b007b5cbee)

지금까지 정리한 수식을 구현을 통해 살펴보겠다.  
![Image](https://github.com/user-attachments/assets/64bb7392-5938-4448-bd77-35c537f4e3cf)

입력 값, 가중치 값, 편향 값은 그림을 참조한다.  
i1, i2를 상수로 고정한 채 w1~w8, b1~b4에 대해 학습을 수행해 보겠다.

다음과 같이 예제를 작성한다.
```python
i1, i2 = .05, .10
t1, t2 = .01, .99

w1, w3 = .15, .25
w2, w4 = .20, .30
b1, b2 = .35, .35

w5, w7 = .40, .50
w6, w8 = .45, .55
b3, b4 = .60, .60

for epoch in range(2000):

    print(f'epoch = {epoch}')

    h1 = i1 * w1 + i2*w2 + 1*b1
    h2 = i1 * w3 + i2*w4 + 1*b2
    o1 = h1 * w5 + h2*w6 + 1*b3
    o2 = h1 * w7 + h2*w8 + 1*b4
    print(f' h1,  h2  = {h1:6.3f}, {h2:6.3f}')
    print(f' o1,  o2  = {o1:6.3f}, {o2:6.3f}')

    E = (o1 - t1)**2/2 + (o2 - t2)**2/2
    print(f' E  = {E:.7f}')
    if E < 0.0000001:
        break

    o1b, o2b = o1 - t1, o2 - t2
    h1b, h2b = o1b*w5 + o2b*w7, o1b*w6 + o2b*w8
    w1b, w3b = i1*h1b, i1*h2b
    w2b, w4b = i2*h1b, i2*h2b
    b1b, b2b = 1*h1b, 1*h2b
    w5b, w7b = h1*o1b, h1*o2b
    w6b, w8b = h2*o1b, h2*o2b
    b3b, b4b = 1*o1b, 1*o2b
    print(f' w1b, w3b = {w1b:6.3f}, {w3b:6.3f}')
    print(f' w2b, w4b = {w2b:6.3f}, {w4b:6.3f}')
    print(f' b1b, b2b = {b1b:6.3f}, {b2b:6.3f}')
    print(f' w5b, w7b = {w5b:6.3f}, {w7b:6.3f}')
    print(f' w6b, w8b = {w6b:6.3f}, {w8b:6.3f}')
    print(f' b3b, b4b = {b3b:6.3f}, {b4b:6.3f}')

    lr = 0.01
    w1, w3 = w1 - lr*w1b, w3 - lr*w3b
    w2, w4 = w2 - lr*w2b, w4 - lr*w4b
    b1, b2 = b1 - lr*b1b, b2 - lr*b2b
    w5, w7 = w5 - lr*w5b, w7 - lr*w7b
    w6, w8 = w6 - lr*w6b, w8 - lr*w8b
    b3, b4 = b3 - lr*b3b, b4 - lr*b4b
    print(f' w1,  w3  = {w1:6.3f}, {w3:6.3f}')
    print(f' w2,  w4  = {w2:6.3f}, {w4:6.3f}')
    print(f' b1,  b2  = {b1:6.3f}, {b2:6.3f}')
    print(f' w5,  w7  = {w5:6.3f}, {w7:6.3f}')
    print(f' w6,  w8  = {w6:6.3f}, {w8:6.3f}')
    print(f' b3,  b4  = {b3:6.3f}, {b4:6.3f}')
```

다음은 실행 결과 화면이다.
```
epoch = 664
 h1,  h2  =  0.239,  0.226
 o1,  o2  =  0.010,  0.990
 E  = 0.0000001
 w1b, w3b = -0.000,  0.000
 w2b, w4b = -0.000,  0.000
 b1b, b2b = -0.000,  0.000
 w5b, w7b =  0.000, -0.000
 w6b, w8b =  0.000, -0.000
 b3b, b4b =  0.000, -0.000
 w1,  w3  =  0.143,  0.242
 w2,  w4  =  0.186,  0.284
 b1,  b2  =  0.213,  0.186
 w5,  w7  =  0.203,  0.533
 w6,  w8  =  0.253,  0.583
 b3,  b4  = -0.095,  0.730
epoch = 665
 h1,  h2  =  0.239,  0.226
 o1,  o2  =  0.010,  0.990
 E  = 0.0000001
```
