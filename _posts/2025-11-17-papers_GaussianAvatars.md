---
layout: post
title: "&#91;Papers&#93; GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians &#40;CVPR 2024 Highlight&#41;"
date: 2025-11-17
description: Code Review
tags: Paper
categories: Paper
featured: true
---


## GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians
### [[Paper]](https://arxiv.org/abs/2312.02069)[[Github]](https://github.com/ShenhanQian/GaussianAvatars)[[Project]](https://shenhanqian.github.io/gaussian-avatars)

>**Title:** GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians 
**Journal name & Publication Date:** CVPR 2023-12-04  
**Authors:** Shenhan Qian, Tobias Kirschstein, Liam Schoneveld

---
>## 0. Parameters & Functions

먼저 아래와 같이 train.py에서 gaussian을 학습한다고 하면, 그냥 주의할 parsing으로는 ```bind_to_mesh``` 정도인 것 같다.

{% highlight python linenos %}
SUBJECT=306
python train.py \
-s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--eval --bind_to_mesh --white_background --port 60000
{% endhighlight %}

{% highlight python linenos %}
if dataset.bind_to_mesh:
	gaussians = FlameGaussianModel(...)
    mesh_renderer = NVDiffRenderer()
{% endhighlight %}

training 초반 ```bind_to_mesh``` argument를 통해 아마 강제적으로 실행하는 부분인 것 같다.
train에서 사용하는 gaussians는 FlameGaussianModel로 초기화 한다.


{% highlight python linenos %}
# scene/flame_gaussian_model.py
class FlameGaussianModel(GaussianModel):
{% endhighlight %}


애초에 FlameGaussianModel은 GaussianModel을 상속받고 있기 때문에 GaussianModel 자체도 봐야하는 것 같다. 아마 FlameGaussianModel은 Flame Model과 관련된 부분들 관련한 function, parameters 들을 관리하려고 만들어 놓은 것 같다.

{% highlight python linenos %}
self.face_center = None
self.face_scaling = None
self.face_orien_mat = None
self.face_orien_quat = None
self.binding = None
self.binding_counter = None
self.timestep = None
{% endhighlight %}

scene/gaussian_model.py에 늘 익숙한 parameter들을 제외하고 어색한 parameter들이 보인다.
해당 parameter들이 mesh binding에 필요한 GaussianModel parameter인 것 같다. 논문에서도 볼 수 있는 triangle mesh의 global parameter들을 의미하는 것 같다.
binding에서 gaussian과 binding되는 부모 triangle mesh의 index를 부여하는 것 같다. 만약 None이라면 binding되지 않은 gaussian인 것 같다.
binding_counter 역시 해당 gaussian이 묶여있는 triangle에서의 gaussian counter인 것 같은데, 이는 pruning을 할 때 mesh에서 1개 이상의 gaussian을 유지한다는 정책을 이행하기 위한 부분인 것 같다.

### 3D Gaussian parameters

{% highlight python linenos %}
@property
def get_scaling(self):
    if self.binding is None:
        return self.scaling_activation(self._scaling)
    else:
        # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
        if self.face_scaling is None:
            self.select_mesh_by_timestep(0)

        scaling = self.scaling_activation(self._scaling)
        return scaling * self.face_scaling[self.binding]
{% endhighlight %}

gaussian parameter 부분의 global scaling 같은 경우는 face scaling k와 local gaussian parameter s와의 곱 연산으로 이루어진다.

$$
\mathbf{s}' = k\mathbf{s}
$$

{% highlight python linenos %}
@property
def get_xyz(self):
    if self.binding is None:
        return self._xyz
    else:
        # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
        if self.face_center is None:
            self.select_mesh_by_timestep(0)
        
        xyz = torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1)
        return xyz * self.face_scaling[self.binding] + self.face_center[self.binding]
{% endhighlight %}

gaussian의 global position도 아래와 같은 triangle의 global parameter와 local gaussian parameter의 조합으로 이루어 진다. self.binding 자체가 Gaussian별로 대응되는 triangle의 파라미터를 한 번에 gather해서 연산한다

$$
\mathbf{\mu}' = k\mathbf{R}\mathbf{\mu} + \mathbf{T}
$$

{% highlight python linenos %}
@property
def get_rotation(self):
    if self.binding is None:
        return self.rotation_activation(self._rotation)
    else:
        # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
        if self.face_orien_quat is None:
            self.select_mesh_by_timestep(0)

        # always need to normalize the rotation quaternions before chaining them
        rot = self.rotation_activation(self._rotation)
        face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding])
        return quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))  # roma
        # return quaternion_multiply(face_orien_quat, rot)  # pytorch3d
{% endhighlight %}

rotation도 같은 개념으로 사용된다.

$$
\mathbf{r}' = \mathbf{R}\mathbf{r}
$$


---

>## 1. Training Iteration

해당 부분에서는 train.py의 61번째 줄 부터 시작하는 train iteration에서의 부분을 살펴본다. 기본 GaussianModel과 같은 부분은 생략하고 특징적인 부분만 순차적으로 살펴본다.

{% highlight python linenos %}
# 118 line
if gaussians.binding != None:
	gaussians.select_mesh_by_timestep(...)
{% endhighlight %}

먼저 118번째 줄의 if문인데 초반에 FlameGaussianModel이 init되면서 bbinding을 flame_model.faces수만큼 초기화 되고 counter도 각 face마다 1로 초기화 되게 된다. 이 부분으로 인해 gaussians.binding은 None값을 가지지는 않으니 해당 부분이 실행된다.

{% highlight python linenos %}
    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
{% endhighlight %}

select_mesh_by_timestep은 4D 특성상 timestep이 변할 때 마다 flame_model의 parameter들로 vertex와 triangle, faces에 대한 정보들을 받아오는 것 같다. 동시에 update_mesh_properties를 통해 mesh 정보들(verts, faces...) update를 진행한다.

이후에는 loss dict를 이용해서 계산한다.

{% highlight python linenos %}
losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim
losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
{% endhighlight %}

논문에서 아래와 같은 loss항들이 열겨되는데 해당 부분을 계산하는데 해당 부분을 loss dict형태로 계산하는 부분인 것 같다.

$$
\mathcal{L}_{\text{rgb}} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}\\
\mathcal{L}_{\text{position}} = \lvert\lvert \text{max}(\mu, \epsilon_{\text{position}}) \rvert\rvert_2 \\
\mathcal{L}_{\text{scaling}} = \lvert\lvert \text{max}(\mu, \epsilon_{\text{scaling}}) \rvert\rvert_2 \\
\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{position}}\mathcal{L}_{\text{position}} + \lambda_{\text{scaling}}\mathcal{L}_{\text{scaling}}
$$

---