---
title: 머신러닝 야학 - Tensorflow 101 정리-3
author: fast01
date: 2021-01-09 18:00:00 +0800
categories: [challenge,머신러닝야학]
tags: [challenge,ML]
toc: false
---

<h2><span style="color:red">머신러닝 야학 2기 </span></h2>
----------
https://opentutorials.org/course/4548
생활코딩 - 머신러닝 코스 


<h2><span style="color:red"> 머신러닝 야학 </span></h2>
----------
생활코딩에서 주최하는 야학에 참가하게됐습니다.
https://ml.yah.ac/
일정은 2021.1.4 : 개강 ~ 2021.1.15 : 종강 
총 10일 동안 진행되는 야학입니다

<h2><span style="color:red"> 학습 커리큘럼</span></h2>
----------
머신러닝에 대해 알고는 있지만 다시 한번 상기시킬겸 머신러닝1 수업을 듣고,
텐서플로우( python )을 들을 계획이다.

사진
<span style="color:red">실습 환경 </span>
----------

**Google Colaboratory**을 사용합니다.

<h2><span style="color:red"> 아이리스 품종 비교 </span></h2>
----------
아이리스 품종을 분류하는 딥러닝 모델을 텐서플로우를 이용하여 만들어 보고, 
분류모델과 회귀모델의 차이점을 이해합니다. 
범주형 변수의 처리 방법인 원핫인코딩을 해야하는 이유와 활성화함수 softmax를 사용하는 이유를 학습합니다.

<span style="color:green"> Tensorflow 101 정리-1~2와 3 비교  </span>
----------

앞 레모네이드 판매량 예측, 보스톤 집값 예측과 아이리스 품종 비교의 가장 큰 차이점은 결과값의 자료형입니다. 앞에서 본 두 모델은 결과값이 수치값으로 독립변수를 통해 종속변수를 예측하면 되는 모델이였습니다.
하지만 아이리스 품종 예측은 어떠한 데이터로 Versicolor, Setosa, Virginica등 문자열데이터로 예측해야합니다.
문자열 데이터로 예측하는 모델의 문제점은 **범주형 데이터**라는 것 입니다.

꽃잎길이와 꽃잎의 폭, 꽃받침의 길이와 폭 등 여러 수치를 범주화 하여 
꽃잎길이20~21은 Versicolor / 꽃잎길이22~23은 Setosa 등 범주를 가지고 비교해야합니다.

<span style="color:green">원핫인코딩 </span>
----------
위에서 설명한 것 처럼
우리가 이번에 작업할 데이터는 범주형 데이터입니다.
하지만 컴퓨터는 결과값으로 나올 Versicolor, Setosa, Virginica를 인식하지 못합니다. 
모두 숫자로 바꿔줘야한다는 의미죠, 이 작업을 **원핫 인코딩**이라고 합니다.
사진 

위 사진과 같이 품종.Versicolor 식으로 컬럼을 만들어 해당 행이 어떤 범주에 속하는지 표현합니다.


<span style="color:green">Softmax</span>
----------
위에서 만든 범주형데이터를 이제 확률로써 예측하는 시간입니다.
예전에는 단순히 x에 가중치를 곱해 만든 수치가 결과값이였다면 
이제는 3개의 아이리스 품종에서 하나를 골라야 합니다. 따라서 3개의 아이리스 확률을 각각 골라야하고
3개의 아이리스 확률의 합은 1이여야 합니다. 

복잡한 수식을 도와주는 함수중 softmax라는 활성화 함수가 있습니다.
사진
위 사진 처럼 softmax는 0~1사이의 데이터로 출력되게 만들어주는 함수입니다.


<span style="color:green">과거의 데이터를 준비합니다. </span>
----------

	# 1.과거의 데이터를 준비합니다.
	파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
	아이리스 = pd.read_csv(파일경로)
	아이리스.head()
	 
	# 원핫인코딩
	아이리스 = pd.get_dummies(아이리스)
	 
	# 종속변수, 독립변수
	독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
	종속 = 아이리스[['품종_setosa', '품종_versicolor', '품종_virginica']]
	print(독립.shape, 종속.shape)
데이터를 준비합니다


<span style="color:green">모델의 구조를 만듭니다. </span>
----------

	# 2. 모델의 구조를 만듭니다
	X = tf.keras.layers.Input(shape=[4])
	Y = tf.keras.layers.Dense(3, activation='softmax')(X)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='categorical_crossentropy',
	              metrics='accuracy')

모델의 구조를 만듭니다.
총 4개의 독립변수로 3개의 출력값을 계산합니다.
위에서 설명한 것 처럼 Softmax를 통해 확률값(0~1)을 출력합니다.

또한 위 코드에서 중요한 부분이 하나 더 있는데 Compile부분 입니다. 
loss(손실값)이 그 전엔 Mse(결과값 오차)였지만 회귀문제에선 mse를 써도 되지만 
이번 분류문제에서는 이 손실값함수를 사용할 수 없습니다.
깊게 들어가진 않겠지만 궁금하신 분은 찾아보세요~!
또한 metrics='accuracy' 이 부분은 이 모델의 정확도를 표현하는 코드입니다.



<span style="color:green">데이터로 모델을 학습(FIT)합니다. </span>
----------

    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=10)
위 코드에서 중요한 부분은 epochs입니다.
epochs는 학습의 횟수입니다. 



<span style="color:green">모델을 이용합니다.</span>
----------

	# 4. 모델을 이용합니다
	# 맨 처음 데이터 5개
	print(model.predict(독립[:5]))
	print(종속[:5])
	 
	# 맨 마지막 데이터 5개
	print(model.predict(독립[-5:]))
	print(종속[-5:])
모델을 사용하여 출력합니다.

	print(model.get_weights())
모델의 수식 확인


<span style="color:green">숙제 Sigmoid vs Softmax </span>
----------
- Sigmoid
	-	특징
		-	확률의 총합이 1이 아니다!
		-	출력 확률값이 크면 해당 Class가 유력하지만 실제 확률은 아니다
		-	binary-classification에서 사용
- Softmax
	-	특징
		-	확률의 총합이 1이다.
		-	출력 확률값이 크면 해당 Class가 유력하고 실제 확률 값이다.
		-	multi-classification에서 사용
	
	즉 두개의 차이점은 어디서 사용하냐에 따라 나뉜다.
	binary-classification vs multi-classification
	- multi는 위에서 본 여러 범주안에서 하나를 고르는 모델
	- binary는 어떤 데이터가 어떤 class를 표현하냐 아니냐, 즉 Yes or No문제 이다
	
<span style="color:green">전체 코드</span>
----------

	###########################
	# 라이브러리 사용
	import tensorflow as tf
	import pandas as pd
	 
	###########################
	# 1.과거의 데이터를 준비합니다.
	파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
	아이리스 = pd.read_csv(파일경로)
	아이리스.head()
	 
	# 원핫인코딩
	아이리스 = pd.get_dummies(아이리스)
	 
	# 종속변수, 독립변수
	독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
	종속 = 아이리스[['품종_setosa', '품종_versicolor', '품종_virginica']]
	print(독립.shape, 종속.shape)
	 
	###########################
	# 2. 모델의 구조를 만듭니다
	X = tf.keras.layers.Input(shape=[4])
	Y = tf.keras.layers.Dense(3, activation='softmax')(X)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='categorical_crossentropy',
	              metrics='accuracy')
	 
	###########################
	# 3.데이터로 모델을 학습(FIT)합니다.
	model.fit(독립, 종속, epochs=1000, verbose=0)
	model.fit(독립, 종속, epochs=10)
	 
	###########################
	# 4. 모델을 이용합니다
	# 맨 처음 데이터 5개
	print(model.predict(독립[:5]))
	print(종속[:5])
	 
	# 맨 마지막 데이터 5개
	print(model.predict(독립[-5:]))
	print(종속[-5:])
	 
	###########################
	# weights & bias 출력
	print(model.get_weights())

----------
모든 내용은 아래 링크에서 학습한 내용이고 문제시 글 내리겠습니다.
https://opentutorials.org/module/4966/28974

