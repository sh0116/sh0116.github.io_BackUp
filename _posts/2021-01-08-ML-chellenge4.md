---
title: 머신러닝 야학 - Tensorflow 101 정리-2
author: fast01
date: 2021-01-08 18:00:00 +0800
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

<h2><span style="color:red"> 보스턴 집값 예측</span></h2>
----------
보스턴 집값을 예측하는 딥러닝 모델을 텐서플로우를 이용하여 만들어 보고, 
모델을 구성하는 퍼셉트론에 대해 이해합니다.

<span style="color:green"> Tensorflow 101 정리-1과 비교  </span>
----------

바로 전 포스팅에서 설명한 레모네이드 판매량 예측과 보스턴 집값 예측을 비교하자면
데이터에서 큰 차이가 있다. 
레모네이드는 한 독립변수( 온도 ) 가 판매량을 결정했지만 
보스턴 집값 예측은 여러 독립변수(범죄율, 방수, 재산세 세율, 학생/교사비율 등)이 판매량을 결정한다.
즉 독립변수는 여러개지만 종속변수는 하나인 셈이다.

-> 위 내용을 정리하면 온도*2가 판매량이라는 공식을 y= x*2라고 표현 할 수 있다.
-> 다시 정리하면 **y(예측값 )= x1(한 독립변수) * w1(가중치) + x2(한 독립변수) * w2(가중치) ... +b**  인 셈이다
-> b는 편향으로 위 강의에선 자세히 설명하지않았지만 알고리즘과 원하는 값이 얼마나 떨어져있는지 나타내는 값이다. 즉 편향이 클수록 알고리즘과 답이 멀어져있는 것 이다.

위 설명을 모두 포함에서 퍼셉트론이라는 하나의 뉴런을 구성한다.
사진


<span style="color:green">과거의 데이터를 준비합니다. </span>
----------

	파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
	보스턴 = pd.read_csv(파일경로)
	print(보스턴.columns)
	보스턴.head()
	#독립 종속변수 설정
	독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
				'ptratio', 'b', 'lstat']]
	종속 = 보스턴[['medv']]
	print(독립.shape, 종속.shape)
데이터를 준비합니다


<span style="color:green">모델의 구조를 만듭니다. </span>
----------

	X = tf.keras.layers.Input(shape=[13])
	Y = tf.keras.layers.Dense(1)(X)
	model = tf.keras.models.Model(X, Y)
	model.compile(loss='mse')

모델의 구조를 만듭니다.
총 13개의 독립변수로 1개의 출력값을 보냅니다.


<span style="color:green">데이터로 모델을 학습(FIT)합니다. </span>
----------

    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=10)
위 코드에서 중요한 부분은 epochs입니다.
epochs는 학습의 횟수입니다. 






<span style="color:green">모델을 이용합니다.</span>
----------

	print(model.predict(독립[5:10]))
	# 종속변수 확인
	print(종속[5:10])
모델을 사용하여 출력합니다.

	print(model.get_weights())
모델의 수식 확인

<span style="color:green">전체 코드</span>
----------

    ###########################
    # 라이브러리 사용
    import tensorflow as tf
    import pandas as pd
     
    ###########################
    # 1.과거의 데이터를 준비합니다.
    파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
    보스턴 = pd.read_csv(파일경로)
    print(보스턴.columns)
    보스턴.head()
     
    # 독립변수, 종속변수 분리 
    독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                'ptratio', 'b', 'lstat']]
    종속 = 보스턴[['medv']]
    print(독립.shape, 종속.shape)
     
    ###########################
    # 2. 모델의 구조를 만듭니다
    X = tf.keras.layers.Input(shape=[13])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(loss='mse')
     
    ###########################
    # 3.데이터로 모델을 학습(FIT)합니다.
    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=10)
     
    ###########################
    # 4. 모델을 이용합니다
    print(model.predict(독립[5:10]))
    # 종속변수 확인
    print(종속[5:10])
     
    ###########################
    # 모델의 수식 확인
    print(model.get_weights())

----------
모든 내용은 아래 링크에서 학습한 내용이고 문제시 글 내리겠습니다.
https://opentutorials.org/module/4966/28974

