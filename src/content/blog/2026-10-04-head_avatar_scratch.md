---
title: "[Curious] Head Avatar scratch"
slug: head_avatar_scratch
date: 2026-10-04
category: Research
tags: []
description: ELITE 논문 기반으로 간단한 Head Avatar를 구현한다.
draft: true
---

### 들어가며
motivation 자체는 최근의 여러 3D Head Avatar 논문을 보다가 공통적으로 발견한 부분으로 Canonical UV Map을 통해 Appeaance를 High-Quality로 표현하는 부분들이 보였다.

[ELITE (CVPR 2026)](https://youwangk.github.io/elite), [FiCA (Meta Tech Report 2026)](https://youwangk.github.io/FiCA)
:::gallery
![ELITE_fig5](/assets/blog/head_avatar_scratch/elite_fig5-6c80c154.png)

![FiCA_fig2](/assets/blog/head_avatar_scratch/FiCA_fig2-e1854c82.png)
:::

기존에 3D Head Avatar를 표현하는 대표적인 방법으로 [GaussianAvatars (CVPR 2024 Highlight)](https://shenhanqian.github.io/gaussian-avatars) 논문과 같이 FLAME Mesh 기반에서 gaussian을 initial하면서 rigging하며 기존 3DGS 방식처럼 update하는 방법이 가장 익숙하다.

결론적으로 이번 Head Avatar scratch를 통해서 UV Map을 이용한 3D Head Avatar Reconstruction에 한 단계씩 알아보는 것을 목표로 한다. 특히 Canonical UV Map -> Gaussian UV Map -> Gaussian Avatar라는 이 흐름 자체를 이해할 수 있으면 나중에 요긴하게 사용할 수 있을 것 같다. 동시에 UV Map 사용의 기존 방법론 보다 어떤 장점을 가지고 있는지에 대해서도 살펴보면 좋을 것 같다.

### 1. Video → Tracking (FLAME)
먼저 Input Video로부터 FLAME Mesh를 얻기 위한 초반 과정이 필요하다. 아주 정확하진 않지만, 하나씩 순서대로 생각나는 것을 적어보자면 

1단계 (ffmpeg, matting)
Video → Frame분리 → Alpha값 추출
![level1](/assets/blog/head_avatar_scratch/level1-78131242.jpg)

2단계 (VHAP)
→ detect landmarks → tracking → export(?)
:::gallery
![level2_1](/assets/blog/head_avatar_scratch/level2_1-50710fa4.jpg)

![level2_2](/assets/blog/head_avatar_scratch/level2_2-f08a4542.jpg)
:::

여기까지는 기존에 널리 쓰이는 [VHAP](https://github.com/ShenhanQian/VHAP)을 사용하는 것이라 익숙하기도 하고 어렵지 않다.

### 2. Tracking result → Canonical UV Map
앞에서 Mesh Tracking을 통해서 추출한 정보들을 이용해 Canonical Mesh UV Maps을 만든다. 이는 실제 FLAME 표면을 UV pixel로 1:1로 대응시켜 색상(RGB), 위치(xyz), 대응점(index), 커버되지 않은 부분 등 다양하게 UV 형태로 Unwrapping시킬 수 있다.

:::gallery
![xyz_visualization](/assets/blog/head_avatar_scratch/xyz_visualization-485a0abc.png)

![texture](/assets/blog/head_avatar_scratch/texture-ce905644.png)

![normal](/assets/blog/head_avatar_scratch/normal-81749efb.png)

![tangent](/assets/blog/head_avatar_scratch/tangent-78f1da03.png)

![valid](/assets/blog/head_avatar_scratch/valid-c98843f0.png)

![bitangent](/assets/blog/head_avatar_scratch/bitangent-7cb362c7.png)

![coverage](/assets/blog/head_avatar_scratch/coverage-0f1c2640.png)

![face_index](/assets/blog/head_avatar_scratch/face_index-be843745.png)
:::

이렇게 완전히 대응되게 unwrapping 할 수 있는건 기존 FLAME Mesh가 가지고 있는 faces, vt, ft 덕분이며 이미 vertex의 정보들, 각 triangl mesh를 구성하는 vertex 조합, UV vertex의 2D 위치를 가지고 있게 된다.

이중 faces[k]와 ft[k]는 같은 표면 삼각형을 표현하게 되고 3D Vertex가 2D UV Vertex에 자연스럽게 대응이 되게 된다.

이제 UV좌표를 Rasterizer가 존재하는 clip 공간으로 바꾸어 UV평면 위의 삼각형들을 그리게 되다.
texel 중심의 UV 좌표와 해당 texel이 대응하는 삼각형 번호를 얻게 되고 Barycentric을 이용해서 삼각형 내부 위치를 파악하게 되면, 이를 3D Vertex에 해당하는 점에 똑같이 대응하게 되어 해당 점에서의 대응하는 x,y,z를 얻게 된다.

```xyz[row, col] = 해당 texel에 대응하는 표면 위치 (x,y,z)```


```python
pixel, depth = L03.project(points, K, w2c)
```

```text
UV texel
   ↓ face + barycentric
프레임의 3D 표면 위치
   ↓ 카메라 projection
원본 영상의 이미지 좌표
   ↓ RGB sampling
그 texel에 들어갈 관측 색
```

### 3. UV represent → Gausian Map 


### 4. Gaussian Map → Gaussian Avatar


### 5. Animation & Eval 


---
### 정리하며
