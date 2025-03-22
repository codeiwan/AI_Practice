[Home](./../../README.md) | [인공 지능의 딥러닝 알고리즘](./../README.md) | 기본 인공 신경 동작 구현해 보기

## 기본 인공 신경 동작 구현해 보기
지금까지의 과정을 그림과 수식을 통해 정리한 후, 구현을 통해 확인해 보자.  
다음 그림은 지금까지 살펴본 입력1 출력1로 구성된 인공 신경을 나타낸다.  
![Image](https://github.com/user-attachments/assets/93ee29b4-53c9-4b16-801f-fd5449da1bbd)  
[그림1]은 순전파 과정에 필요한 변수와 수식을 나타낸다.  
[그림2]는 역전파에 필요한 변수이다. 순전파에 대응되는 변수가 모두 필요하다.  
[그림3]은 입력의 역전파에 필요한 변수와 수식을 나타낸다.  
[그림4]는 가중치와 편향의 역전파에 필요한 변수와 수식을 나타낸다.  
(※ xb값은 앞부분에 또 다른 인공 신경과 연결되어 있을 경우 yb처럼 해당 인공 신경으로 역전파되는 값이다.  
역전파된 xb값은 해당 인공 신경의 가중치와 편향 학습에 사용된다.)  

이상에서 필요한 수식을 정리하면 다음과 같다.  
- 순전파  
$xw + 1b = y$
- 입력 역전파  
$y_bw = x_b$
- 가중치, 편향 역전파  
$xy_b = w_b$  
$1y_b = b_b$
- 인공 신경망 학습  
$w = w - aw_b$  
$b = b - ab_b$

지금까지의 과정을 구현을 통해 살펴보자.  
![Image](https://github.com/user-attachments/assets/3531f2f8-088f-4746-8bfd-81812936233b)  
이 그림에서 입력 값 x, 가중치 w, 편향 b는 각각 2, 3, 1이고 목표 값 t는 10이다.  
(※ 이 값을은 임의의 값들이다. 다른 값들을 사용하여 학습을 수행할 수도 있다.)

다음과 같이 예제를 작성한다.
```python
x = 2
t = 10
w = 3
b = 1

y = x*w + 1*b
print(f'y = {y:6.3f}')

yb = y - t
xb = yb*w
wb = yb*x
bb = yb*1
print(f'xb = {xb:6.3f}, wb = {wb:6.3f}, bb = {bb:6.3f}')

lr = 0.01
w = w - lr*wb
b = b - lr*bb
print(f'x = {x:6.3f}, w = {w:6.3f}, b = {b:6.3f}')
```
- 변수 x를 선언한 후, 2로 초기화한다.
- 변수 t를 선언한 후, 10으로 초기화한다.
- 가중치 변수 w를 선언한 후, 3으로 초기화한다.  
가중치 w는 입력 값의 강도, 세기라고도 하며 입력 값을 증폭시키거나 감소시키는 역할을 한다.  
인공 신경도 가지 돌기의 두께에 따라 입력 신호가 증폭되거나 감소될 수 있는데, 이런 관점에서 가중치는 가지 돌기의 두께에 해당되는 변수로 생각할 수 있다.  
![Image](https://github.com/user-attachments/assets/c52cd915-22fc-4b91-b2ff-e39f27b67ac3)  
- 편향 변수 b를 선언한 후, 1로 초기화한다.  
편향은 가중치를 거친 입력 값의 합(=전체 입력 신호)에 더해지는 값으로 입력신호를 좀 더 세게 해주거나 약하게 하는 역할을 한다.
- 순전파 수식을 구한다.
- print 함수를 호출하여 순전파 결과 값 y를 출력한다.  
소수점 이하 3자리까지 출력한다.
- yb 변수를 선언한 후, 순전파 결과 값에서 목표 값을 빼 오차 값을 넣어준다.  
- xb 변수를 선언한 후, 입력 값에 대한 역전파 값을 받는다.  
이 부분은 이 예제에서 필요한 부분은 아니며, 역전파 연습을 위해 추가하였다.
- wb 변수를 선언한 후, 가중치 값에 대한 역전파 값을 받는다.  
- bb 변수를 선언한 후, 편향 값에 대한 역전파 값을 받는다.  
- print 함수를 호출하여 역전파 결과 값 wb, bb를 출력한다.  
소수점 이하 3자리까지 출력한다.
- 학습률 변수 lr을 선언한 후, 0.01로 초기화한다.  
- wb 역전파 값에 학습률을 곱한 후, w값에서 빼준다.  
이 과정에서 w 변수에 대한 학습이 이루어진다.
- bb 역전파 값에 학습률을 곱한 후, b값에서 빼준다.  
이 과정에서 b 변수에 대한 학습이 이루어진다.
- print 함수를 호출하여 학습된 결과 값 w, b를 출력한다.  
소수점 이하 3자리까지 출력한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/1e9253d1-637e-4d21-b0c6-7d180faf20fc)  
현재 y값은 7이다. wb, bb 값을 확인한다. 또 w, b 값을 확인한다.
<br>
<br>

### 반복 학습 2회 수행하기
여기서는 반복 학습 2회를 수행해 보겠다.

다음과 같이 예제를 수정한다.
```python
x = 2
t = 10
w = 3
b = 1

for epoch in range(2):

    print(f'epoch = {epoch}')

    y = x*w + 1*b
    print(f' y  = {y:6.3f}')

    yb = y - t
    xb = yb*w
    wb = yb*x
    bb = yb*1
    print(f' xb = {xb:6.3f}, wb = {wb:6.3f}, bb = {bb:6.3f}')

    lr = 0.01
    w = w - lr*wb
    b = b - lr*bb
    print(f' x  = {x:6.3f}, w  = {w:6.3f}, b  = {b:6.3f}')
```
- epoch값을 0에서 2 미만까지 바꾸어가며 2회 반복을 수행한다.
- print 함수를 호출하여 epoch 값을 출력한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/f4f799b7-26d6-4809-830b-130f0b05e3ab)  
y 값이 7에서 7.150으로 바뀌는 것을 확인한다. wb, bb 값을 확인한다. 또, w, b 값을 확인한다.
<br>
<br>

### 반복 학습 20회 수행하기
여기서는 반복 학습 20회를 수행해 보겠다.

다음과 같이 예제를 수정한다.
```python
x = 2
t = 10
w = 3
b = 1

for epoch in range(20):

    print(f'epoch = {epoch}')

    y = x*w + 1*b
    print(f' y  = {y:6.3f}')

    yb = y - t
    xb = yb*w
    wb = yb*x
    bb = yb*1
    print(f' xb = {xb:6.3f}, wb = {wb:6.3f}, bb = {bb:6.3f}')

    lr = 0.01
    w = w - lr*wb
    b = b - lr*bb
    print(f' x  = {x:6.3f}, w  = {w:6.3f}, b  = {b:6.3f}')
```
- epoch값을 0에서 20 미만까지 수행한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/0863c6ec-e0c4-4ce0-b249-9f329dcb6bcd)  
y 값이 8.868까지 접근하는 것을 확인한다.
<br>
<br>

### 반복 학습 200회 수행하기
여기서는 반복 학습 200회를 수행해 보겠다.

다음과 같이 예제를 수정한다.
```python
x = 2
t = 10
w = 3
b = 1

for epoch in range(200):

    print(f'epoch = {epoch}')

    y = x*w + 1*b
    print(f' y  = {y:6.3f}')

    yb = y - t
    xb = yb*w
    wb = yb*x
    bb = yb*1
    print(f' xb = {xb:6.3f}, wb = {wb:6.3f}, bb = {bb:6.3f}')

    lr = 0.01
    w = w - lr*wb
    b = b - lr*bb
    print(f' x  = {x:6.3f}, w  = {w:6.3f}, b  = {b:6.3f}')
```
- epoch값을 0에서 200 미만까지 수행한다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/2ed7b827-b279-4215-8dc4-8ca5278560cb)  
y 값이 10.000에 수렴하는 것을 확인한다.  
이 때, 가중치 w는 4.2, 편향 b는 1.6에 수렴한다.

### 오차 값 계산하기
여기서는 인공 신경망을 통해 얻어진 예측 값과 목표 값의 오차를 계산하는 부분을 추가해 보겠다.  
오차(error)는 손실(loss) 또는 비용(cost)이라고도 한다.  
오차 값이 작을수록 예측을 잘하는 인공 신경망이다.

다음과 같이 예제를 수정한다.
```python
x = 2
t = 10
w = 3
b = 1

for epoch in range(200):

    print(f'epoch = {epoch}')

    y = x*w + 1*b
    print(f' y  = {y:6.3f}')

    E = (y-t)**2/2
    print(f' E  = {E:.7f}')
    if E < 0.0000001:
        break

    yb = y - t
    xb = yb*w
    wb = yb*x
    bb = yb*1
    print(f' xb = {xb:6.3f}, wb = {wb:6.3f}, bb = {bb:6.3f}')

    lr = 0.01
    w = w - lr*wb
    b = b - lr*bb
    print(f' x  = {x:6.3f}, w  = {w:6.3f}, b  = {b:6.3f}')
```
- 변수 E를 선언한 후, 다음과 같은 현태의 수식을 구현한다.  
$E=\frac{1}{2}(y-t)^2$  
y의 값이 t에 가까울수록 E의 값은 0에 가까워진다.  
즉, 오차 값이 0에 가까워진다.  
이 수식을 오차함수 또는 손실함수 또는 비용함수라고 한다.
- print 함수를 호출하여 오차 값 E를 출력한다.  
소수점 이하 7자리까지 출력한다.
- 오차 값 E가 0.0000001(1천만분의1)보다 작으면 break문을 수행하여 6줄의 for문을 빠져 나간다.

다음은 실행 결과 화면이다.  
![Image](https://github.com/user-attachments/assets/733286b9-1014-4f66-94ec-9d800825364a)  
epoch 값이 172(173회 째)일 때 for 문을 빠져 나간다.  
y값은 10에 수렴한다.
<br>
<br>

### 학습률 변경하기
여기서는 학습률 값을 변경시켜 보면서 학습의 상태를 살펴보겠다.

다음과 같이 예제를 수정한다.
```python
x = 2
t = 10
w = 3
b = 1

for epoch in range(200):

    print(f'epoch = {epoch}')

    y = x*w + 1*b
    print(f' y  = {y:6.3f}')

    E = (y-t)**2/2
    print(f' E  = {E:.7f}')
    if E < 0.0000001:
        break

    yb = y - t
    xb = yb*w
    wb = yb*x
    bb = yb*1
    print(f' xb = {xb:6.3f}, wb = {wb:6.3f}, bb = {bb:6.3f}')

    lr = 0.05
    w = w - lr*wb
    b = b - lr*bb
    print(f' x  = {x:6.3f}, w  = {w:6.3f}, b  = {b:6.3f}')
```
- 학습률 값을 0.05로 변경한다.

다음은 실행 결과 화면이다.
```
epoch = 30
 y  =  9.999
 E  = 0.0000001
 xb = -0.002, wb = -0.001, bb = -0.001
 x  =  2.000, w  =  4.200, b  =  1.600
epoch = 31
 y  = 10.000
 E  = 0.0000001
```
32회 째 학습이 완료되는 것을 볼 수 있다.
