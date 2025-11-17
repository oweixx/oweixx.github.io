---
layout: post
title: "&#91;Papers&#93; Learning a model of facial shape and expression from 4D scans &#40;SIGGRAPH 2017&#41;"
date: 2025-11-16
description: Paper Review
tags: Paper
categories: Paper
featured: true
---


## Learning a model of facial shape and expression from 4D scans
### [[Paper]](https://dl.acm.org/doi/10.1145/3130800.3130813)[[Github]](https://github.com/soubhiksanyal/FLAME_PyTorch)[[Project]](https://flame.is.tue.mpg.de/)

>**Title:** Learning a model of facial shape and expression from 4D scans  
**Journal name & Publication Date:** SIGGRAPH 2017-11-16  
**Authors:** Tianye Li, Timo Bolkart, Michael J. Black

---
>## 1. Abstract & Introduction

3D Face Modeling에서 face capture를 통한 아주 detail한 face reconstruction 방법이 있지만 이는 종종 mesh를 제대로 구현하지 못하는 artifacts들이 발생한다. (맨 위의 그림)
반대로 low resolution일 경우 face expression에 대한 표현이 제대로 이루어지지 못하는 현상이 발생하게 된다. (맨 아래)
해당 논문에서는 이 지점들의 middle ground를 목표로 하는 FLAME model (Faces Learned with an Articulated Model and Expressions)를 제안한다.

![](https://velog.velcdn.com/images/lowzxx/post/373cfb1e-268a-48fa-9b94-1a532e10d947/image.png)

- FLAME은 low-dimensional이지만 기존의 FaceWarehouse, Basel face model보다 more expressive한 FLAME Model을 제안한다.
- FLAME Model은 더 정확하며 추후 연구에 활용될 수 있게 더 적합한 모델이다. 
- Face mesh model을 parameterization하여 더 유연하게 다룰 수 있게 만들었다.

---
>## 3. Model Formulation

![](https://velog.velcdn.com/images/lowzxx/post/49220311-d522-4d52-8ee3-2097b6f0450c/image.png)

Flame mesh model은 기본적으로 SMPL model formulation을 따른다. SMPL이 Facial에 집중적으로 formulation 되어 있지는 않지만, 이를 이용하면 더 computationally efficient하며 더 경쟁력 있는 부분이라 사용했다고 한다.

Flame은 정확한 blendshapes을 위해 linear blend skinning(LBS)를 사용하고 $N = 5023$ 개의 vertices(정점), $K = 4$ 개의 joints(neck, jaw and eyeballs)을 사용한다고 한다.

<img src="https://velog.velcdn.com/images/lowzxx/post/7041c752-51ac-4538-9ee9-e9a101bcd712/image.png" width="900"/>
>### Parameter descrition
- **shape coefficients:** $\vec\beta \in \R^{\left\vert \vec\beta \right\vert}$
- **pose coefficients:** $\vec\theta \in \R^{\left\vert \vec\theta \right\vert}$
- **expression coefficients:** $\psi \in \R^{\left\vert \vec\psi \right\vert}$
- $J(\vec{\beta})$: shape에 따라 달라지는 joint 위치
- $\mathcal{W}$: 각 vertex에 대한 skinning weight 행렬

아래는 최종 FLAME Model에 대한 equation이다.
$$
T_P(\vec{\beta}, \vec{\theta}, \vec{\psi}) 
= \bar{T} 
+ B_S(\vec{\beta}; \mathcal{S}) 
+ B_P(\vec{\theta}; \mathcal{P}) 
+ B_E(\vec{\psi}; \mathcal{E}).
$$
해당 부분 부터 보자면, 기본 모델 $\bar{T}$에서 model을 변형하는 parameter shape, pose, expression에 대한 parameter들을 통해 blendshape을 하여 변형된 model을 반환하게 된다.

$$
M(\vec{\beta}, \vec{\theta}, \vec{\psi}) 
= W\big(T_P(\vec{\beta}, \vec{\theta}, \vec{\psi}), 
       J(\vec{\beta}), 
       \vec{\theta}, 
       \mathcal{W}\big),
$$

그렇게 반환된 mesh model $T_P$는 관절 위치 Joint와 관절 회전, blendshape 가중치와 함께 최종 모델로 변환이 된다.

여기까지는 매우 간단한 설명이었고, 이후 부터는 해당 model을 직접적으로 변형하는 blendshape에 대한 설명이다. 해당 부분에 대해서는 직접적으로 이해가 되는 부분들이 많지 않아서 gpt와 함께 확인하였다.
### blendshapes
**Shape blendshapes**
$$
B_S(\vec{\beta}; \mathcal{S})
  = \sum_{n=1}^{|\beta|}
      \beta_n \, \mathbf{S}_n,

$$
$$
\vec{\beta}
  = [\beta_1, \dots, \beta_{|\beta|}]^\mathsf{T},
\quad
\mathcal{S}
  = [\mathbf{S}_1, \dots, \mathbf{S}_{|\beta|}]
    \in \mathbb{R}^{3N \times |\beta|}.
$$
$\vec{\beta}$는 사람의 얼굴형을 결정하는 shape coefficient이고 $\mathbf{S}$는 n번째 vertices의 basis로 각 basis 방향으로 $\beta_n$만큼의 이동하는 형태를 의미한다.

**Pose blendshapes**

$$
R(\vec{\theta}) : \mathbb{R}^{|\theta|} \to \mathbb{R}^{9K},
$$
회전 함수 $R(\vec\theta)$ joint의 회전 행렬 요소들을 한 벡터로 정의한다.

$$
B_P(\vec{\theta}; \mathcal{P})
  = \sum_{n=1}^{9K}
      \big(
        R_n(\vec{\theta}) - R_n(\vec{\theta}^\ast)
      \big)\, \mathbf{P}_n,
$$

$$
\mathcal{P}
  = [\mathbf{P}_1, \dots, \mathbf{P}_{9K}]
    \in \mathbb{R}^{3N \times 9K}.
$$

zero pose를 의미하는 $R_n(\vec{\theta}^\ast)$와 현재포즈 사이에서의 차이를 통해 얼마나 회전 행렬이 바뀌었는지를 계산하고, 회전 요소가 변할 때 생기는 vertex 보정 $\mathbf{P}_n$을 선형조합해 Pose를 표현하게 된다. 

**Expression blendshapes**
Expression blendshape은 "웃음, 찡그림, 놀람"과 같은 표정의 표현에 대한 부분이고 **non-rigid facial deformation**을 설명하는 pose와 독립적인 공간이다.

$$
B_E(\vec{\psi}; \mathcal{E})
  = \sum_{n=1}^{|\psi|}
      \psi_n \, \mathbf{E}_n,
$$

$\mathbf{E}_n$은 특정 표정 방향에 해당하는 expression basis에 해당하여 여러 표정 basis들을 가중치로 섞어서 만든 expression deformation을 의미하는 blendshape이다.

$$
\vec{\psi}
  = [\psi_1, \dots, \psi_{|\psi|}]^\mathsf{T},
\quad
\mathcal{E}
  = [\mathbf{E}_1, \dots, \mathbf{E}_{|\psi|}]
    \in \mathbb{R}^{3N \times |\psi|}.
$$

$\mathcal{E}$는 orthonormal expression basis를 의미한다고 한다.

---
>## 4. Temporal Registration

### Initial model

FLAME model은 shape $\{\bar{T},\mathcal{S}\}$, pose $\{\mathcal{P}, \mathcal{W}, \mathcal{J}\}$, expression $\mathcal{E}$ parameter들을 통해 변형이 되기 때문에 이들에게 일관성있게 변화해야한다.

**Shape**
초반의 initial model을 만들기 위해 SMPL의 full-body model에서 head model을 refined하여 이용하였다고 한다.
더 안정적이고 visual quality를 위해 눈 양쪽에 eyeball을 추가하여 현실성과 디테일을 더하였다.

**Pose**
blendweights $\mathcal{W}$와 joint regressor$\mathcal{J}$는 사용자에 의해 정해지며, eyeball의 pose는 기본적으로 geometric center에 해당하는 pose로 initial된다.

**Expression**
expression parameter $\mathcal{E}$를 초기화하기 위해 기본 head와 artist generated FACS-based blendshape과 비교하여 초기화를 하고, deformation transfer를 통해 model 변환을 하게 된다. 

### Single-frame registration

위에서 정의한 FLAME Model을 실제 3D Scan, Multi-View image에 대하여 어떻게 맞추는지에 대한 일련의 과정을 설명한다.

**Our model-baed registration of a face scan consists of three steps.**
**Model-only -> Coupled -> Texture-based**

**1. Model-only**

$$
E(\vec{\beta}, \vec{\theta}, \vec{\psi})
= E_D + \lambda_L E_L + E_P.
$$

먼저 model coefficients만을 이용하여 scan과 landmark가 잘 맞을 수 있게 대략적인 optimizing을 하는 단계이다.

$$
E_D
= \lambda_D
  \sum_{v_s \in \mathcal{S}}
    \rho \!\left(
      \min_{v_m \in \mathcal{M}(\vec{\beta}, \vec{\theta}, \vec{\psi})}
      \left\| v_s - v_m \right\|_2
    \right),
$$

스캔의 각 점을 FLAME mesh 표면에 closest point로 매칭시키고 robust penalty를 걸어 outlier에 강건할 수 있게 한다.

$$
E_L
= \sum_{i=1}^{N_L}
    \left\|
      \Pi\!\big(\mathbf{v}_i(\vec{\beta}, \vec{\theta}, \vec{\psi})\big)
      - \mathbf{l}_i
    \right\|_2^2,
$$

multi-view에서 landmark가 잘 align 되도록 하는 역할을 한다.

$$
E_P
= \lambda_{\theta} E_{\theta}
+ \lambda_{\beta} E_{\beta}
+ \lambda_{\psi} E_{\psi},
$$

Prior term $E_P$에서는 파라미터들이 너무 큰 값으로 튀지 않게 정규항을 사용한다.

**2. Coupled**

$$
E(T, \vec{\beta}, \vec{\theta}, \vec{\psi})
= E_D + E_C + E_R + E_P.
$$

이번엔 model이 설명하지 못하는 부분에 대하여 T가 deform될 수 있도록 하는 부분이다.

$$
E_C
= \sum_{e}
    \lambda_e
    \left\|
      T_e
      - M(\vec{\beta}, \vec{\theta}, \vec{\psi})_e
    \right\|_2^2,
$$

단순히 vertex의 위치 차이를 줄이는 것 보다 edge 차이를 줄이는 것을 optimize objective로 두어 더 자연스럽고 매끄러운 deformation을 할 수 있게 한다.

$$
E_R
= \sum_{k}
    \lambda_k \left\|
      U(\mathbf{v}_k)
    \right\|_2^2,
$$

$$
U(\mathbf{v}_k)
= \frac{1}{|\mathcal{N}(k)|}
  \sum_{r \in \mathcal{N}(k)} (\mathbf{v}_r - \mathbf{v}_k),
$$

outlier, fold-over나 찌그러지는 현상을 피하고, noisy regularization을 위해 사용한다.

**3. Texture-based**

$$
E(T, \vec{\beta}, \vec{\theta}, \vec{\psi})
= E_D + E_C + \lambda_T E_T + E_R + E_P.
$$

앞에 두 단계에서는 geometry (scan, landmark)기반의 optimize 기반이의 과정이었다. 실제 detail에 관한 부분을 보강하고 expression/pose를 더 정밀하게 optimize한다.

$$
E_T
= \sum_{l = 0}^{L-1}
    \sum_{v = 1}^{V}
      \left\|
        \Gamma\!\big(I_l^{(v)}\big)
        -
        \Gamma\!\big(\hat{I}_l^{(v)}\big)
      \right\|_F^2, 
$$

직관적으로는 실제 이미지와 렌더된 이미지가 비슷해지도록 template parameter들을 조정하고, 해당 과정을 coarse level에서 큰 misalignment를 먼저 맞추고 finer level에서 detail들을 refine할 수 있게 유도한다.

### Sequential registration

여기서는 두 section으로 나뉜다.
Personalization에서 개인화된 template을 생성하고, Sequence fitting에서 해당 template을 부드럽게 프레임별로 움직일 수 있도록 해준다.

**Personalization**
multiple sequence에서 neutral pose, expression을 구하기 위해 $T_i$의 average를 구한다.
그다음 neutral result결과 중 하나의 $T_i$를 랜덤으로 골라 UV texture map을 만들고 이는 나중에 texture-based registration에서 사용된다.

**Sequence fitting**
간단하게 말해 그 template을 고정한 채, 이전 프레임 결과로 초기화된 single-frame registration을 프레임마다 반복해서 시간적으로 일관성을 가진 4D Face sequences를 등록하게 된다.


---
---