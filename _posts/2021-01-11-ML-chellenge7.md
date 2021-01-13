---
title: 머신러닝 야학 - Tensorflow 101 정리-5
author: fast01
date: 2021-01-11 18:00:00 +0800
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

<h2><span style="color:red"> 데이터를 위한 팁 </span></h2>
----------

데이터를 처리하다보면 여러 오류가 발생한다.
초보자가 할 수 있는 오류중 대표적인 몇가지가 있는데 
이번 포스팅에서는 그 오류를 소개하고 잡는 방법을 설명할 것입니다.

1.	원핫인코딩에서 범주형 데이터 인식 오류
2.	NA값 처리

( NA값이란 들어오지 않은 값으로 이 경우에는 여러 방법으로 NA값을 채울 수 있다. )

1번 해결 방법

	# 품종 타입을 범주형으로 바꾸어 준다. 
	아이리스['품종'] = 아이리스['품종'].astype('category')
	print(아이리스.dtypes)
	
	# 카테고리 타입의 변수만 원핫인코딩
	인코딩 = pd.get_dummies(아이리스)
	인코딩.head()

2번 해결 방법

	# NA값을 체크해 봅시다. 
	아이리스.isna().sum()
	아이리스.tail()
	 
	###########################
	# NA값에 꽃잎폭 평균값을 넣어주는 방법
	mean = 아이리스['꽃잎폭'].mean()
	print(mean)
	아이리스['꽃잎폭'] = 아이리스['꽃잎폭'].fillna(mean)
	아이리스.tail()

NA값을 체크하고 NA값 항목에 평균값을 채워 넣는다.
( NA값에는 여러 값이 들어갈 수 있는데 모델의 종류 역활에 따라서 바뀐다.  )


<span style="color:green">전체 코드 </span>
----------

	###########################
	# 라이브러리 사용
	import pandas as pd
	 
	###########################
	# 파일 읽어오기
	파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
	아이리스 = pd.read_csv(파일경로)
	아이리스.head()
	 
	###########################
	# 칼럼의 데이터 타입 체크
	print(아이리스.dtypes)
	 
	# 원핫인코딩 되지 않는 현상 확인
	인코딩 = pd.get_dummies(아이리스)
	인코딩.head()
	 
	###########################
	# 품종 타입을 범주형으로 바꾸어 준다. 
	아이리스['품종'] = 아이리스['품종'].astype('category')
	print(아이리스.dtypes)
	 
	# 카테고리 타입의 변수만 원핫인코딩
	인코딩 = pd.get_dummies(아이리스)
	인코딩.head()
	 
	###########################
	# NA값을 체크해 봅시다. 
	아이리스.isna().sum()
	아이리스.tail()
	 
	###########################
	# NA값에 꽃잎폭 평균값을 넣어주는 방법
	mean = 아이리스['꽃잎폭'].mean()
	print(mean)
	아이리스['꽃잎폭'] = 아이리스['꽃잎폭'].fillna(mean)
	아이리스.tail()

<h2><span style="color:red"> 모델을 위한 팁 </span></h2>
----------
BatchNormalization layer를 사용하여 보다 학습이 잘되는 모델을 만들어 봅니다.

아래 전체 코드 1과 2는 사실 별거 없다 
전 포스팅의 보스턴집값 예측과 아이리스 품종 판단 코드에 히든 레이어 생성 부분을

	H = tf.keras.layers.Dense(8, activation='swish')(H)

위 코드 한줄의 코드를 3줄의 코드로 늘린것뿐인다.👇

		H = tf.keras.layers.Dense(8)(X)
		H = tf.keras.layers.BatchNormalization()(H)
		H = tf.keras.layers.Activation('swish')(H)
더 자세히 말하자면 

	1.
	H = tf.keras.layers.Dense(8, activation='swish')(H)
	
	2.
	H = tf.keras.layers.Dense(8)(X)
	H = tf.keras.layers.Activation('swish')(H)
1번과 2번은 같은 의미이다 활성함수를 Dense안에 서술하냐 한줄을 더 써서 서술하냐 차이이다.
이렇게 서술한 이유는 저 두 함수사이에 Nomalization이 들어가야하기 때문이다.
( 사이에 들어가는 이유는 그래야 학습이 잘된다고 한다 )

우리가 집중해서 봐야하는 코드는  아래 한줄의 코드이다

	H = tf.keras.layers.BatchNormalization()(H)
이 코드는 Batch Normalization을 해주는 코드로 

Batch Normalization은 학습하는 과정을 안정화하여 가속시켜주는 정규화 방법이다.

<span style="color:Yellow"> Batch Normalization</span>
----------
이 알고리즘이 나오게 된 계기는 멀티 레이어 환경에서 나올 수 있는 오류인 **Internal Covariance Shift** 때문이다.

이 불안정화는 각 레이어의 활성화함수마다 입력값의 분산이 달라져서 생긴 현상인데 간단히 말해서 
이전 레이어의 변화로 생긴 파라미터(입력값)변화로 현재 레이어의 입력 분포가 바뀐는 현상이다.
그리고 이런 레이어의 변화가 레이어를 통과할 때 마다 생기며 입력 분포가 계속 변화하는 현상을 **Internal Covariance Shift** 이라고 한다.

설명이 좀 어려웠지만 결국에는 배치 정규화를 통해 위 불안정화를 안정화시킬 수 있어 모델이 더 성능 좋게 학습될 수 있다라는 결론이다.

TMI.
이 코드가 왜 굳이 중간에 들어가야하나 궁금한 사람이 있을것같아 적는다.
배치 정규화의 원리는 레이어 중간 중간마다 생기는 감마 베타를 구해 정규화를 해주는 역활을 통해 불안정성을 해결한다. 
따라서 레이어의 생성 중간 중간 마다 감마 베타를 구하는 코드가 필요하고 그것이 바로 

	H = tf.keras.layers.BatchNormalization()(H)
이 코드인 것이다.

<span style="color:green">전체 코드 1 </span>
----------

	###########################
	# 라이브러리 사용
	import tensorflow as tf
	import pandas as pd
	 
	###########################
	# 1.과거의 데이터를 준비합니다.
	파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
	보스턴 = pd.read_csv(파일경로)
	 
	# 종속변수, 독립변수
	독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox', 
	            'rm', 'age', 'dis', 'rad', 'tax',
	            'ptratio', 'b', 'lstat']]
	종속 = 보스턴[['medv']]
	print(독립.shape, 종속.shape)
	 
	###########################
	# 2. 모델의 구조를 만듭니다
	X = tf.keras.layers.Input(shape=[13])
	H = tf.keras.layers.Dense(8, activation='swish')(X)
	H = tf.keras.layers.Dense(8, activation='swish')(H)
	H = tf.keras.layers.Dense(8, activation='swish')(H)
	Y = tf.keras.layers.Dense(1)(H)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='mse')
	 
	# 2. 모델의 구조를 BatchNormalization layer를 사용하여 만든다.
	X = tf.keras.layers.Input(shape=[13])
	 
	H = tf.keras.layers.Dense(8)(X)
	H = tf.keras.layers.BatchNormalization()(H)
	H = tf.keras.layers.Activation('swish')(H)
	 
	H = tf.keras.layers.Dense(8)(H)
	H = tf.keras.layers.BatchNormalization()(H)
	H = tf.keras.layers.Activation('swish')(H)
	 
	H = tf.keras.layers.Dense(8)(H)
	H = tf.keras.layers.BatchNormalization()(H)
	H = tf.keras.layers.Activation('swish')(H)
	 
	Y = tf.keras.layers.Dense(1)(H)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='mse')
	 
	###########################
	# 3.데이터로 모델을 학습(FIT)합니다.
	model.fit(독립, 종속, epochs=1000)
	
<span style="color:green">전체 코드 2 </span>
----------

	###########################
	# 라이브러리 사용
	import tensorflow as tf
	import pandas as pd
	 
	###########################
	# 1.과거의 데이터를 준비합니다.
	파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
	아이리스 = pd.read_csv(파일경로)
	 
	# 원핫인코딩
	아이리스 = pd.get_dummies(아이리스)
	 
	# 종속변수, 독립변수
	독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
	종속 = 아이리스[['품종_setosa', '품종_versicolor', '품종_virginica']]
	print(독립.shape, 종속.shape)
	 
	###########################
	# 2. 모델의 구조를 만듭니다
	X = tf.keras.layers.Input(shape=[4])
	H = tf.keras.layers.Dense(8, activation='swish')(X)
	H = tf.keras.layers.Dense(8, activation='swish')(H)
	H = tf.keras.layers.Dense(8, activation='swish')(H)
	Y = tf.keras.layers.Dense(3, activation='softmax')(H)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='categorical_crossentropy',
	              metrics='accuracy')
	 
	###########################
	# 2. 모델의 구조를 BatchNormalization layer를 사용하여 만든다.
	X = tf.keras.layers.Input(shape=[4])
	 
	H = tf.keras.layers.Dense(8)(X)
	H = tf.keras.layers.BatchNormalization()(H)
	H = tf.keras.layers.Activation('swish')(H)
	 
	H = tf.keras.layers.Dense(8)(H)
	H = tf.keras.layers.BatchNormalization()(H)
	H = tf.keras.layers.Activation('swish')(H)
	 
	H = tf.keras.layers.Dense(8)(H)
	H = tf.keras.layers.BatchNormalization()(H)
	H = tf.keras.layers.Activation('swish')(H)
	 
	Y = tf.keras.layers.Dense(3, activation='softmax')(H)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='categorical_crossentropy',
	              metrics='accuracy')
	               
	###########################
	# 3.데이터로 모델을 학습(FIT)합니다.
	model.fit(독립, 종속, epochs=1000)

----------
모든 내용은 아래 링크에서 학습한 내용이고 문제시 글 내리겠습니다.
https://opentutorials.org/module/4966/28974

