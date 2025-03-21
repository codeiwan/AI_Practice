[Home](./../../../README.md) | [인공 지능 딥러닝의 이해](./../../README.md) | [인공 신경망의 이해](./../README.md) | 인공 신경망의 학습 방법

## 인공 신경망의 학습 방법
전통적인 알고리즘과는 달리 인공 신경망은 프로그래머의 의도대로 작업하도록 '프로그램 되거나' 또는 '구성되거나' 할 수 없다.<br>
인간의 뇌처럼 인공 신경망은 하나의 일을 수행할 방법을 직접 배워야 한다.<br>
일반적으로 인공 신경망의 학습 방법에는 3가지 전략이 있다.

### 지도 학습
가장 간단한 학습 방법이다.<br>
미리 알려진 결과들이 있는 충분히 많은 데이터가 있을 때 사용하는 방법이다.<br>
지도 학습은 다음처럼 진행된다.

1. 하나의 입력 데이터를 처리한다.
2. 출력값을 미리 알려진 결과와 비교한다.
3. 인공 신경망을 수정한다.
4. 이 과정을 반복한다.

이것이 지도 학습 방법이다.

예를 들어 부모가 어린 아이에게 그림판을 이용하여 사물을 학습시키는 방법은 지도 학습과 같다.<br>
한글, 숫자 등에 대한 학습도 지도 학습의 형태이다.<br>
아래에 있는 그림판에는 동물, 과일, 채소 그림이 있고 해당 그림에 대한 이름이 있다.<br>
아이에게 고양이를 가리키면서 '고양이'라고 알려주는 과정에서 아이는 학습을 하게 된다.<br>
이와 같은 방식으로 인공 신경망도 학습을 시킬 수 있으며, 이런 방법을 지도 학습이라고 한다.

![Image](https://github.com/user-attachments/assets/e8e16ff6-f1f5-441c-89c7-d38aecf96979)

### 비지도 학습
비지도 학습은 입력 값이 목표 값과 같을 때 사용하는 학습 방법이다.<br>
예를 들어, 메모리 카드 게임을 하는 방식을 생각해 보자.<br>
1. 그림에 표현된 사물의 이름을 모르는 상태로 사물의 형태를 통째로 기억한다.
2. 같은 그림을 찾아 내며 게임을 진행한다.

이와 같이 입력 값과 출력 값이 같은 형태의 데이터를 학습할 때, 즉, 입력 값을 그대로 기억해 내야 하는 형태의 학습 방법을 비지도 학습이라고 한다.

![Image](https://github.com/user-attachments/assets/214d7266-d334-4028-8426-3701babd0d25)

### 강화 학습
인공 신경망이 익숙하지 않은 환경에서 시행착오를 통해 이익이 되는 동작을 취할 확률은 높이고 손해가 되는 동작을 취할 확률을 낮추게 하는 학습 방법이다.<br>
즉, 이익이 되는 동작을 강화해가는 학습 방법이다.<br>
예를 들어, 우리가 익숙하지 않은 환경에서 어떤 동작을 취해야 하는지 모를 때, 일단 할 수 있는 동작을 취해보고 그 동작이 유리한지 불리한지를 체득하는 형태의 학습 방식과 같다.<br>
이 과정에서 유리한 동작은 기억해서 점점 더 하게 되고 물리한 동작도 기억해서 점점 덜 하게 된다.
