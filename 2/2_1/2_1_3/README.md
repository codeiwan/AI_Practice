[Home](./../../../README.md) | [인공 지능의 딥러닝 알고리즘](./../../README.md) | [딥러닝 동작 원리 이해하기](./../README.md) | $y = 3 \times x + 1$ 학습시켜 보기

## $y = 3 \times x + 1$ 학습시켜 보기
여기서는 다음과 같은 숫자들의 집합 X, Y를 이용하여, 단일 인공 신경을 합습시켜 보겠다.  
![Image](https://github.com/user-attachments/assets/463bb056-4c92-4d5e-b705-dd835804d136)  
![Image](https://github.com/user-attachments/assets/408627ee-0d03-497d-bb58-c7ea75d02f03)  

그래서 다음 함수를 근사하는 인공 신경 함수를 만들어 보도록 하자.

$y = f(x) = 3 \times x + 1$ ($x$는 실수)  

인공 신경을 학습시키는 과정은 w, b 값을 X, Y 값에 맞추어 가는 과정이다.  
그래서 학습이 진행됨에 따라 w 값은 3에 가까운 값으로, b 값은 1에 가까운 값으로 이동하게 된다.

다음과 같이 예제를 작성하자.
```python
xs = [-1., 0., 1., 2., 3., 4.]
ys = [-2., 1., 4., 7., 10., 13.]
w = 10.
b = 10.

y = xs[0]*w + 1*b
print(f'x  = {xs[0]:6.3f}, y  = {y:6.3f}')

t = ys[0]
E = (y - t)**2/2
print(f'E  = {E:.7f}')

yb = y - t
wb = yb*xs[0]
bb = yb*1
print(f'wb = {wb:6.3f}, bb = {bb:6.3f}')

lr = 0.01
w = w - lr*wb
b = b - lr*bb
print(f'w  = {w:6.3f}, b  = {b:6.3f}')
```
