---
title: 머신러닝 야학 - Tensorflow 101 정리-4
author: fast01
date: 2021-01-10 18:00:00 +0800
categories: [challenge,머신러닝야학]
tags: [challenge,ML]
toc: true
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

<h2><span style="color:red"> 신경망의 완성:히든레이어 </span></h2>
----------
히든레이어와 멀티레이어의 구조를 이해하고, 히든레이어를 추가한 멀티레이어 인공신경망 모델을 완성해 봅니다. 

사진

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
	H = tf.keras.layers.Dense(8, activation="swish")(X)
	H = tf.keras.layers.Dense(8, activation="swish")(H)
	H = tf.keras.layers.Dense(8, activation="swish")(H)
	Y = tf.keras.layers.Dense(3, activation='softmax')(H)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='categorical_crossentropy',
	              metrics='accuracy')
 
위에서 input으로 독립변수 4개를 받고 
히든 레이어가 총 3개 그리고 마지막 출력 레이어 한개 
총 5개의 레이어로 이루어진 모델입니다.


	# 모델 구조 확인
	model.summary()
모델의 구조를 확인할 수 있는 코드입니다.



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
	 
	# 원핫인코딩
	아이리스 = pd.get_dummies(아이리스)
	 
	# 종속변수, 독립변수
	독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
	종속 = 아이리스[['품종_setosa', '품종_versicolor', '품종_virginica']]
	print(독립.shape, 종속.shape)
	 
	###########################
	# 2. 모델의 구조를 만듭니다
	X = tf.keras.layers.Input(shape=[4])
	H = tf.keras.layers.Dense(8, activation="swish")(X)
	H = tf.keras.layers.Dense(8, activation="swish")(H)
	H = tf.keras.layers.Dense(8, activation="swish")(H)
	Y = tf.keras.layers.Dense(3, activation='softmax')(H)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='categorical_crossentropy',
	              metrics='accuracy')
	 
	# 모델 구조 확인
	model.summary()
	 
	###########################
	# 3.데이터로 모델을 학습(FIT)합니다.
	model.fit(독립, 종속, epochs=100)
	 
	###########################
	# 4. 모델을 이용합니다
	print(model.predict(독립[:5]))
	print(종속[:5])

----------
모든 내용은 아래 링크에서 학습한 내용이고 문제시 글 내리겠습니다.
https://opentutorials.org/module/4966/28974

