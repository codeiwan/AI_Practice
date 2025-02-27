[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [인공 신경망과 근사 함수](./../README.md) | 다양한 함수 근사해 보기

## 다양한 함수 근사해 보기
여기서는 이전과 같이 예제를 수정해 가며, 간단한 함수들을 인공 신경망을 학습시켜 근사시켜 보자.

### 분수 함수 근사해 보기
다음은 분수 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.  
![Image](https://github.com/user-attachments/assets/d7ed8542-8d7e-4314-8baf-0e7e8b648b54)  
$y = \frac{1}{x}$ 　 $(0.1 \leq x \leq 5)$

x 좌표의 범위는 $0.1$에서 $5$까지이다.  
분수함수의 경우 x 값 0에 대해 정의되지 않는다.  
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(0.1, 5, NUM_SAMPLES)
```
```python
ys = 1.0/xs
```
<br/><br/>
### sin 함수 근사해 보기
다음은 sin 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.  
![Image](https://github.com/user-attachments/assets/edf2219a-7807-450f-93a2-e9a704408ffc)  
$y = sin(x)$ 　 $(0 \leq x \leq 2 \pi)$

x 좌표의 범위는 $0$에서 $2\pi$까지 이다.  
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(0, 2*np.pi, NUM_SAMPLES)
```
```python
ys = np.sin(xs)
```
<br/><br/>
### tanh 함수 근사해 보기
다음은 tanh 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.
![Image](https://github.com/user-attachments/assets/dee8437f-ba63-4f04-a13a-d9bad632c86b)
$y = tanh(x)$ 　 $(-5 \leq x \leq 5)$

x 좌표의 범위는 $-5$에서 $5$까지이다.  
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(-5, 5, NUM_SAMPLES)
```
```python
ys = np.tanh(xs)
```
<br/><br/>
### e 지수함수 근사해 보기
다음은 e 지수 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.  
![Image](https://github.com/user-attachments/assets/8824875d-e84d-4d66-913d-71acf12015b1)
$y = e^x$ 　 $(-5 \leq x \leq 5)$

x 좌표의 범위는 $-5$에서 $5$까지이다.   
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(-5, 5, NUM_SAMPLES)
```
```python
ys = np.exp(xs)
```
<br/><br/>
### sigmoid 함수 근사해 보기
다음은 sigmoid 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.  
![Image](https://github.com/user-attachments/assets/8a5bba28-30d7-434e-9cd1-2f6a3dc20668)  
$y = \frac{1}{1-e^{-x}}$ 　 $(-5 \leq x \leq 5)$

x 좌표의 범위는 $-5$에서 $5$까지이다.  
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(-5, 5, NUM_SAMPLES)
```
```python
ys = 1.0/(1.0 + np.exp(-xs))
```
<br/><br/>
### 로그 함수 근사해 보기
다음은 로그 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.  
![Image](https://github.com/user-attachments/assets/abf6b26f-2251-4989-bb58-90bc0f82154f)  
$y = log(x)$ 　 $(0 \leq x \leq 5)$

x 좌표의 범위는 $0$에서 $5$까지이다.  
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(0, 5, NUM_SAMPLES)
```
```python
ys = np.log(xs)
```
<br/><br/>
### 제곱근 함수 근사해 보기
다음은 제곱근 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.  
![Image](https://github.com/user-attachments/assets/2afa31b1-29d6-4ad6-ad7f-7c9939546a96)  
$y = \sqrt{x}$ 　 $(0 \leq x \leq 5)$

x 좌표의 범위는 $0.1$에서 $5$까지이다.  
제곱근 함수의 경우 음수 x 값에 대해 정의되지 않는다.  
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(0, 5, NUM_SAMPLES)
```
```python
ys = np.sqrt(xs)
```
<br/><br/>
### relu 함수 근사해 보기
다음은 relu 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.  
![Image](https://github.com/user-attachments/assets/37e08325-3c38-4582-8b7c-05c0de222e80)  
$y = \begin{cases} x (x > 0)\\ 0　 (x \leq 0) \end{cases}$ 　 $(-3 \leq x \leq 3)$

x 좌표의 범위는 $-3$에서 $3$까지이다.  
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(-3, 3, NUM_SAMPLES)
```
```python
ys = (xs > 0)*xs
```
<br/><br/>
### leaky relu 함수 근사해 보기
다음은 leaky relu 함수에 대한 그래프와 인공 신경망 학습 후, 예측 그래프이다.  
![Image](https://github.com/user-attachments/assets/7c95b6bf-801a-46ba-98bb-e2f05d8b04ab)
$y = \begin{cases} x　 (x > 0)\\ ax　 (x \leq 0) \end{cases}$ 　 $(-3 \leq x \leq 3, 　a=0.1)$

x 좌표의 범위는 $-3$에서 $3$까지이다.  
위 그래프에서 a는 0.1이다.  
일반적으로 0.01을 사용한다.  
이전 예제를 다음과 같이 수정한 후, 테스트를 수행한다.
```python
xs = np.random.uniform(-3, 3, NUM_SAMPLES)
```
```python
ys = (xs > 0)*xs + (xs <= 0)*0.1*xs
```

이상으로 우리가 배웠던 함수들에 대해 인공 신경망을 학습시켜 근사 함수를 만들어 보았다.  
또 몇 가지 활성화 함수들에 대해서도 인공 신경망을 학습시켜 근사 함수를 만들어 보았다.  
실제로 인공 신경망 함수는 앞에서 살펴본 함수로 표현하기 어려운 복잡한 형태의 입출력 데이터에 대한 근사 함수를 만들 때 사용한다.  
예를 들어, 자동차 번호판을 인식하는 함수라든지 사람이나 자동차를 인식하는 함수를 만들 때 사용한다.
