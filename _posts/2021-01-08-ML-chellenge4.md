---
title: 머신러닝 야학 - Tensorflow 101 정리-2
author: fast01
date: 2021-01-08 18:00:00 +0800
categories: [challenge,머신러닝야학]
tags: [challenge,ML]
toc: True
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

<h2><span style="color:red"> 레모네이드 판매 예측</span></h2>
----------
기본적인 지도학습 순서

-	과거의 데이터를 준비합니다.
-	모델의 구조를 만듭니다.
-	데이터로 모델을 학습(FIT)합니다.
-	모델을 이용합니다.

<span style="color:green">과거의 데이터를 준비합니다. </span>
----------

    데이터를 준비합니다.
    파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
    레모네이드 = pd.read_csv(파일경로)
    레모네이드.head()
    독립 = 레모네이드[['온도']]
    종속 = 레모네이드[['판매량']]
    print(독립.shape, 종속.shape)
데이터를 준비하고 독립변수와 종속변수를 설정해줍니다.


<span style="color:green">모델의 구조를 만듭니다. </span>
----------

    # 모델을 만듭니다.
    X = tf.keras.layers.Input(shape=[1])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(loss='mse')
모델의 구조를 만듭니다.
위 코드에서 중요한 부분은 "shape=[1]" , "Dense(1)" 이 부분입니다.
shape=[1] 부분은 	-> 온도라는 컬럼 한개여서 1을 적어줍니다.
Dense(1) 부분은   	-> 판매량이라는 컬럼 한대여서 1을 적어줍니다.
각 부분의 의미는 독립변수 , 종속 변수의 양을 뜻합니다.

<span style="color:green">데이터로 모델을 학습(FIT)합니다. </span>
----------

    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=10)
위 코드에서 중요한 부분은 epochs입니다.
epochs는 학습의 횟수입니다. 

<span style="color:red">
지금 이 시간은 이 정도로만 알고있으면 될꺼같습니다.  (머신러닝 입문 수업이기 때문에)
epochs를 더 깊게 알고 싶다면 epoch에 대해 검색하여 알아보는것도 좋은 방법일꺼같습니다.
 </span>

<span style="color:green">손실( loss )</span>
각 학습이 끝날때마다 얼마나 정확히 모델을 생성하고 있는지 평가하는 지표
공식사진

즉 loss가 0에 가까워질수록 학습이 잘되는 것이다.




<span style="color:green">모델을 이용합니다.</span>
----------

    print(model.predict(독립))
    print(model.predict([[15]]))
모델을 사용하여 출력합니다.
입력 -> 독립 -> 온도
출력 -> 종속 -> 판매량   

<span style="color:green">전체 코드</span>
----------

    ###########################
    # 라이브러리 사용
    import tensorflow as tf
    import pandas as pd
     
    ###########################
    # 데이터를 준비합니다.
    파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
    레모네이드 = pd.read_csv(파일경로)
    레모네이드.head()
    # 종속변수, 독립변수
    독립 = 레모네이드[['온도']]
    종속 = 레모네이드[['판매량']]
    print(독립.shape, 종속.shape)
     
    ###########################
    # 모델을 만듭니다.
    X = tf.keras.layers.Input(shape=[1])
    Y = tf.keras.layers.Dense(1)(X)
    model = tf.keras.models.Model(X, Y)
    model.compile(loss='mse')
     
    ###########################
    # 모델을 학습시킵니다. 
    model.fit(독립, 종속, epochs=1000, verbose=0)
    model.fit(독립, 종속, epochs=10)
     
    ###########################
    # 모델을 이용합니다. 
    print(model.predict(독립))
    print(model.predict([[15]]))

----------
모든 내용은 아래 링크에서 학습한 내용이고 문제시 글 내리겠습니다.
https://opentutorials.org/module/4966/28974
