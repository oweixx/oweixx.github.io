---
layout: post
title: "&#91;Papers&#93; Generalizable and Animatable Gaussian Head Avatar &#40;NeurIPS 2024&#41;"
date: 2025-11-19
description: Code Review
tags: Paper
categories: Paper
featured: true
---


## Generalizable and Animatable Gaussian Head Avatar 
### [[Paper]](https://arxiv.org/abs/2410.07971)[[Github]](https://github.com/xg-chu/GAGAvatar)[[Project]](https://xg-chu.site/project_gagavatar/)

>**Title:** Generalizable and Animatable Gaussian Head Avatar    
**Journal name & Publication Date:** NeurIPS 2024-10-10  
**Affiliation:** The University of Tokyo, RIKEN AIP

---
>## 0. inference

inference는 아래와 같이 수행할 수 있다고 한다.

```inference.py``` 부분을 살펴보면, build_model함수로 GAGAvatars model을 만들어서 model을 불러오는데 해당 부분이 아마 중요한 의미들을 많이 담고 있을 것 같은데 해당 부분은 다음 section에서 따로 더 찾아보는 것으로 해야겠다. 일단은 GAGAvatars model을 불러왔다는 가정으로 시작한다.


# Driven by another image:
# This will track the images online, which is slow.
python inference.py -d ./demos/examples/2.jpg -i ./demos/examples/1.jpg

# Driven by a tracked video:
python inference.py -d ./demos/drivers/obama -i ./demos/examples/1.jpg

# Driven by a tracked image_lmdb
python inference.py -d ./demos/drivers/vfhq_demo -i ./demos/examples/1.jpg
{% endhighlight %}

아래와 같이 GAGAvatars model과 monocular face trucker인 GAGAvatar_Track model을 초기화 해준다.

결국 face trucker model은 단일 이미지, 카메라에 대해 FLAME parameters와 camera parameters를 예측하여 반환하는 모델이다.

그렇게 reference image와 track_engine을 이용하여 ```feature_data```를 반환 받게 된다.

{% highlight python linenos %}
model = build_model(model_cfg=meta_cfg.MODEL)
...
track_engine = TrackEngine(focal_length=12.0, device=device)

feature_name = os.path.basename(image_path).split('.')[0]
feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
{% endhighlight %}

driving image에 대해서도 위와 같은 일련의 과정으로 tacking을 하여 driver_data를 준비한다.

{% highlight python linenos %}
if os.path.isdir(driver_path):
    driver_name = os.path.basename(driver_path[:-1] if driver_path.endswith('/') else driver_path)
    driver_dataset = DriverData(driver_path, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
    driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
else:
    driver_name = os.path.basename(driver_path).split('.')[0]
    driver_data = get_tracked_results(driver_path, track_engine, force_retrack=force_retrack)
    if driver_data is None:
        print(f'Finish inference, no face in driver: {image_path}.')
        return
    driver_dataset = DriverData({driver_name: driver_data}, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
    driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
{% endhighlight %}

---
>## 1. GAGAvatars Model

위에서 확인했던 inference.py에는 module화된 조각들을 잘 처리하는 pipeline자체를 보여주고 있기 때문에 사실상 어떠한 과정이 진행되고 있는지에 대한 부분을 확인하기는 어렵다.

model config file을 보더라도 model을 불러올 때 ```models/GAGAvatar/models.py```에서의 GAGAvatar class를 불러오고 있다. 해당 model의 init 속성을 보았을 때 논문의 Reconstruction Branch를 수행하는 것으로 예상이 된다.

{% highlight python linenos %}
class GAGAvatar(nn.Module):
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()
        self.base_model = DINOBase(output_dim=256)
        for param in self.base_model.dino_model.parameters():
            param.requires_grad = False
        # dir_encoder
        n_harmonic_dir = 4
        self.direnc_dim = n_harmonic_dir * 2 * 3 + 3
        self.harmo_encoder = HarmonicEmbedding(n_harmonic_dir)
        # pre_trained
        self.head_base = nn.Parameter(torch.randn(5023, 256), requires_grad=True)
        self.gs_generator_g = LinearGSGenerator(in_dim=1024, dir_dim=self.direnc_dim)
        self.gs_generator_l0 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
        self.gs_generator_l1 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
        self.cam_params = {'focal_x': 12.0, 'focal_y': 12.0, 'size': [512, 512]}
        self.upsampler = StyleUNet(in_size=512, in_dim=32, out_dim=3, out_size=512)
        self.percep_loss = FacePerceptualLoss(loss_type='l1', weighted=True)
{% endhighlight %}

하나씩 살펴보자. 먼저 ```self.gs_generator_g```는 ```LinearGSGenerator``` model을 사용한다. 해당 class model은 밑에 정의 되어 있다.

보면 Lienar,ReLU의 조합으로 구성된 Sequential 형태이다. 입력으로 들어오는 input_features와 plane_direnc에 대하여 gaussian parameter를 추출 할 수 있는 형태로 설계 되어 있다. 아마 이 부분은 train에서 충분히 학습되어 weight가 담겨있을 것 같다. 

그리고 해당 부분은 위 Expression Branch에서 사용되는 global part형태에 사용되는 gs로 확인이 된다.

{% highlight python linenos %}
# global part
gs_params_g = self.gs_generator_g(
        torch.cat([
            self.head_base[None].expand(batch_size, -1, -1), f_feature1[:, None].expand(-1, 5023, -1), 
        ], dim=-1
    ), plane_direnc
)
gs_params_g['xyz'] = t_points
{% endhighlight %}

{% highlight python linenos %}
class LinearGSGenerator(nn.Module):
    def __init__(self, in_dim=1024, dir_dim=27, **kwargs):
        super().__init__()
        # params
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
        )
        layer_in_dim = in_dim//4 + dir_dim
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 32, bias=True),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 3, bias=True)
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 4, bias=True),
        )

    def forward(self, input_features, plane_direnc):
        input_features = self.feature_layers(input_features)
        plane_direnc = plane_direnc[:, None].expand(-1, input_features.shape[1], -1)
        input_features = torch.cat([input_features, plane_direnc], dim=-1)
        # color
        colors = self.color_layers(input_features)
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = self.opacity_layers(input_features)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_layers(input_features)
        # scales = torch.exp(scales) * 0.01
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = self.rotation_layers(input_features)
        rotations = nn.functional.normalize(rotations)
        return {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations}
{% endhighlight %}

밑에 해당 ```self.gs_generator_10``` 과 ```self.gs_generator_l1```은 local DINO에서 추출된 local feature에 대한 dual-lifting에 관한 코드 부분이다. feature와 plane_direnc에 대하여 양쪽으로 gaussian을 예측하여 Gaussian을 두 층으로 예측을 하게 된다. Gaussian의 parameter를 예측할 때는 Global과는 다르게 얇은 Conv층으로 ```32+1+3+4+1```의 gaussian parameter들을 예측한다. 

{% highlight python linenos %}
 # local part
gs_params_l0 = self.gs_generator_l0(f_feature0, plane_direnc)
gs_params_l1 = self.gs_generator_l1(f_feature0, plane_direnc)
gs_params_l0['xyz'] = f_planes['plane_points'] + gs_params_l0['positions'] * f_planes['plane_dirs'][:, None]
gs_params_l1['xyz'] = f_planes['plane_points'] + -1 * gs_params_l1['positions'] * f_planes['plane_dirs'][:, None]
{% endhighlight %}

궁금한 부분은 해당 두께의 Conv층으로 충분히 해당 Gaussian Parameter들이 잘 예측이 될 수 있는건지이다.

GPT에게 물어봤을 때는 이 부분은 당연히 가능하다는 의견이다. 애초에 DINO에서 추출한 feature형태가 저차원의 형태가 아닌 고차원의 형태이고 지금 현재 상황에서는 아예 다른 차원으로써의 이동이 아닌 mapping에 가까운 의도이기 때문에 이정도 깊이의 conv로 해당 역할을 수행하는 것은 일반적인 쪽에 가깝다고 한다.

{% highlight python linenos %}
class ConvGSGenerator(nn.Module):
    def __init__(self, in_dim=256, dir_dim=27, **kwargs):
        super().__init__()
        out_dim = 32 + 1 + 3 + 4 + 1 # color + opacity + scale + rotation + position
        self.gaussian_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input_features, plane_direnc):
        plane_direnc = plane_direnc[:, :, None, None].expand(-1, -1, input_features.shape[2], input_features.shape[3])
        input_features = torch.cat([input_features, plane_direnc], dim=1)
        gaussian_params = self.gaussian_conv(input_features)
        # color
        colors = gaussian_params[:, :32]
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = gaussian_params[:, 32:33]
        opacities = torch.sigmoid(opacities)
        # scale
        scales = gaussian_params[:, 33:36]
        # scales = torch.exp(scales) * 0.01
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = gaussian_params[:, 36:40]
        rotations = nn.functional.normalize(rotations)
        # position
        positions = gaussian_params[:, 40:41]
        positions = torch.sigmoid(positions)
        results = {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations, 'positions':positions}
        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).reshape(results[key].shape[0], -1, results[key].shape[1])
        return results
{% endhighlight %}

그럼 그렇게 위에서 global하게 나온 head의 gaussian과 밑에서 dual-lifting으로 나온 두 개의 gaussian을 concat하여 하나의 gaussian scene으로 구성하게 된다. 그렇게 image를 생성해서 dict형태의 results로 반환이 되게 된다.

이게 위에 dual-lifting으로 plane에 예측되는 gaussian points의 개수가 $(H,W) = 296 \times 296$ 형태로 나오고 2개의 층으로 나오기 때문에 175,232개의 point가 나온다고 한다.

{% highlight python linenos %}
gs_params = {
    k:torch.cat([gs_params_g[k], gs_params_l0[k], gs_params_l1[k]], dim=1) for k in gs_params_g.keys()
}
gen_images = render_gaussian(
    gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
)['images']
sr_gen_images = self.upsampler(gen_images)
results = {
    't_image':t_image, 't_bbox':t_bbox, 't_points': t_points, 
    'p_points': torch.cat([gs_params_l0['xyz'], gs_params_l1['xyz']], dim=1),
    'gen_image': gen_images[:, :3], 'sr_gen_image': sr_gen_images
}
return results
{% endhighlight %}