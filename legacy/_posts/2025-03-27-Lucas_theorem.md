---
layout: post
title: 뤼카의 정리
date: 2025-03-27
description: 뤼카의 정리
tags: theorem 
categories: theorem
featured: true
---

더 어려운 수학 category를 들어가기에 앞서 뤼카의 정리를 한 번 살펴보고자 한다.

뤼카의 정리는 어떤 조합의 수를 소수 p에 대해 법 p 상에서 구할 때 간편한 계산 방식을 제공한다. 다시 말해 ``작은 값들의 조합을 통해 해를 빠르게 계산``할 수 있다.

<br>

#### 공식화
임의의 음이 아닌 정수 m과 n, 소수 p에 대하여 뤼카의 정리는 다음과 같이 합동식으로 표현할 수 있다.

$$
\binom {m}{n} \equiv \prod_{i=0}^{k}{\binom{m_i}{n_i} \pmod p}  
$$

여기서 첨자가 붙은 수들은 m과 n을 소수 p에 대해 다음과 같이 p진법으로 전개했을 때 얻어지는 것들이다.

1. $m=m_kp^k+m_{k-1}p^{k-1}+\cdots+m_1p+m_0,$
2. $n=n_kp^k+n_{k-1}p^{k-1}+\cdots+n_1p+n_0$

이상과 같은 뤼카의 정리는 임의의 자연수 q에 대해 법 p의 q제곱 형태로 일반화가 가능하다.

<br>

#### 증명

##### 1. 다항식 증명

$$
(1+x)^p
$$

해당 식에 대하여

$$
(1+x)^p = \binom p 0 + \binom p 1 x + \binom p 2 x^2 + \cdots + \binom p p x^p
$$

위의 식으로  표현이 가능하고 이는 $\binom p 1$부터 $\binom p {p-1}$까지 모두 p를 인수로 가진다는 것을 알 수 있다. 이는 $\pmod p$연산을 적용하면 초항과 마지막 항을 제외하고는 모든 항이 제거된다.

정리하면

$$
(1+x)^p \equiv 1 + x^p \pmod p
$$

위 합동으로 해당 명제가 성립한다.

$$
(1+x)^{p^{n+1}} =  ((1+x)^p)^{p^n} \equiv 1 + x^p \pmod p
$$

위 식 역시 귀납적으로 정의 될 수 있으며 해당 명제가 성립한다.

<br>

##### 2. 뤼카의 정리 증명

이를 이용해서 다음과 같이 전개가 가능하다.

$$
\sum_{n=0}^{m} \binom m n x^n \equiv (1+x)^m \equiv \prod_{i=0}^{k} \left[(1+x)^{p^i}  \right]^{m_i} \equiv \prod_{i=0}^{k} \left[ 1+x^{p^i} \right]^{m_i} \pmod p
$$

다시 이항 정리를 써서 안쪽의 식을 풀어내면,

{% raw %}
$$
\equiv \prod_{i=0}^k \left[ \sum_{{n_i}=0}^{m_i} \binom {m_i} {n_i} {x^{n_ip^i}} \right] \equiv \sum_{n=0}^m \left[ \prod_{i=0}^k \binom {m_i}{n_i} \right] x^n \pmod p
$$
{% endraw %}

이 된다. 모든 차수마다 계수는 같으므로 위 뤼카의 정리가 성립하게 된다.