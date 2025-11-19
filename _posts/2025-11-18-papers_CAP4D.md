---
layout: post
title: "&#91;Papers&#93; CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models &#40;CVPR 2025 Oral&#41;"
date: 2025-11-18
description: Code Review
tags: Paper
categories: Paper
featured: true
---


## CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models
### [[Paper]](https://arxiv.org/abs/2412.12093)[[Github]](https://github.com/felixtaubner/cap4d/)[[Project]](https://felixtaubner.github.io/cap4d/)

>**Title:** CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models  
**Journal name & Publication Date:** CVPR 2024-12-16  
**Affiliation:** University of Toronto, Vector Institute, LG Electronics  

---
>## 0. Download FLAME and MDMM weights

먼저 FLAME model과 pretrained된 mmdm weights를 다운 받을 수 있는 shell script를 실행해준다.
이때 flame model은 ```flame2023_no_jaw.pkl```을 사용하는 것 같다.




{% highlight python linenos %}
# 1. Download FLAME blendshapes
# set your flame username and password
bash scripts/download_flame.sh 

# 2. Download CAP4D MMDM weights
bash scripts/download_mmdm_weights.sh
{% endhighlight %}

이후 해당 pkl file을 해당 환경 설정 version과 같은 numpy version으로 수행하기 위해 아래 code를 실행해준다.

{% highlight python linenos %}
python scripts/fixes/fix_flame_pickle.py --pickle_path data/assets/flame/flame2023_no_jaw.pkl
{% endhighlight %}

이후 test와 inference는 아래와 같은 scripts로 수행하면 된다고 한다.

{% highlight python linenos %}
# for check installation with a test run
bash scripts/test_pipeline.sh

# for inference
bash scripts/generate_felix.sh
bash scripts/generate_lincoln.sh
bash scripts/generate_tesla.sh
{% endhighlight %}

---
>## 1. Custom inference

논문의 저자가 작년 CVPR 2024에 냈었던 FlowFace 방법론을 여기서 Flame Tracking이자 MMDM Estimator로 사용했었는데, 아직 해당 FlowFace Model의 코드 공개가 되어있지 않아 있는 상태라 Pixel3DMM tracking의 코드를 이용한다. 

![](https://velog.velcdn.com/images/lowzxx/post/c503d688-c5ea-4b86-a0a0-c309a9b3661a/image.png)

해당 논문도 FLAME tracking을 지원하고 있다. 신기하게도 1저자 이신분이 GaussianAvatar와 Nersemble을 작성하셨던 분이었다.

아래가 example로 제공된 felix 선생님으로 tracking을 돌려볼 수 있는 부분이다. 아마 돌리면 위와 같이 FLAME mesh를 추출할 수 있는 것 같다. (나중에 시간이 나면 해봐야겠다...)

{% highlight python linenos %}
export PIXEL3DMM_PATH=$(realpath "../PATH/TO/pixel3dmm")
export CAP4D_PATH=$(realpath "./") 

mkdir examples/output/custom/

# For more information on arguments
bash scripts/track_video_pixel3dmm.sh --help

# Process a directory of (reference) images
bash scripts/track_video_pixel3dmm.sh examples/input/felix/images/cam0/ examples/output/custom/reference_tracking/

# Optional: process a driving (or reference) video
bash scripts/track_video_pixel3dmm.sh examples/input/animation/example_video.mp4 examples/output/custom/driving_video_tracking/
{% endhighlight %}

아래와 같은 cli로 MMDM을 통한 generated image를 할 수 있는 부분이다. ```cap4d/inference/generate_images.py``` 부분은 뒤에서 한 번 더 살펴봐야겠다.

{% highlight python linenos %}
# Generate images with single reference image
python cap4d/inference/generate_images.py --config_path configs/generation/default.yaml --reference_data_path examples/output/custom/reference_tracking/ --output_path examples/output/custom/mmdm/
{% endhighlight %}

이후에 GaussianAvatars를 base로 하는 code를 기반으로 Fit Gaussian avatar를 진행한다.

{% highlight python linenos %}
python gaussianavatars/train.py --config_path configs/avatar/default.yaml --source_paths examples/output/custom/mmdm/reference_images/ examples/output/custom/mmdm/generated_images/ --model_path examples/output/custom/avatar/ --interval 5000
{% endhighlight %}

---
>## 2. Generate Images (with MMDM)

해당 Section에서는 ```cap4d/inference/generate_images.py``` 부분에 대해 알아볼 예정이다. Diffusion 분야에 대해서는 Code딴으로 다뤄본적이 많지 않아서 자세하게는 못 볼 수 있지만 논문 본문과 비교하여 예상이 가는 곳과 특징적인 곳들을 살펴볼 예정이다.

{% highlight python linenos %}
from omegaconf import OmegaConf
gen_config = OmegaConf.load(gen_config_path)
{% endhighlight %}
omegaconf라는 라이브러리 모듈을 사용하면 config를 ```.``` 속성이 아닌 dictionary 스타일로 접근할 수도 있다고 한다.

먼저 56번째 줄에서 본격적으로 이미지 생성을 위한 Dataset을 만들기 시작한다.
먼저 ```GenerationDataset```에서는 생성할 이미지 sample수와 flame dict등에 대한 정보를 담고 있다

{% highlight python linenos %}
# line 56
# ./generate_images.py
genset = GenerationDataset(...)
{% endhighlight %}

{% highlight python linenos %}
#./data/generation_data.py
class GenerationDataset(CAP4DInferenceDataset):
	...
{% endhighlight %}

{% highlight python linenos %}
#./data/inference_data.py
class CAP4DInferenceDataset(Dataset):
	...
{% endhighlight %}

상속받고 있는 ```CAP4DInferenceDataset```에서 이제 기본적인 전체적인 baseline형태의 dataset class를 제공한다. 여기서는 ```__getitem__```에서 idx를 받으면 condition 정보들과 reference 정보를 dict형태로 반환해준다. 자세하게는 아래와 같은 형태로 반환해준다.

{% highlight python linenos %}
cond_dict = {
    "out_crop_mask": out_crop_mask[None],
    "reference_mask": reference_mask[None],
    "ray_map": ray_map[None],
    "verts_2d": verts_2d[None],
    "offsets_3d": offsets_3d[None],
}  # [None] is for fake time dimension

out_dict = {
    "jpg": img[None],  # jpg names comes from controlnet implementation
    "hint": cond_dict,
    "flame_params": flame_item,
}

return out_dict
{% endhighlight %}

윗 부분은 reference의 dataloader가 아닌 generation에 대한 data이다. 그렇다 보니 일종의 계산된 dataset을 사용해서 840개의 condition을 이용하는 느낌인 것 같다.

이제 여기서부터 본격적으로 논문에 수도코드 형태로 제공된 것과 같이 StochasticIOSampler를 돌리는 것 같다. sample의 형태는 reference와 gen에 대한 condition과 여러 parameter들이 들어간다.

해당 부분의 main generate 마지막 부분이다.

{% highlight python linenos %}
stochastic_io_sampler = StochasticIOSampler(device_model_map)

z_gen = stochastic_io_sampler.sample(
    S=gen_config["n_ddim_steps"],
    ref_cond=ref_data["cond_frames"],
    ref_uncond=ref_data["uncond_frames"],
    gen_cond=gen_data["cond_frames"],
    gen_uncond=gen_data["uncond_frames"],
    latent_shape=(4, gen_config["resolution"] // 8, gen_config["resolution"] // 8),
    V=gen_config["V"],
    R_max=gen_config["R_max"],
    cfg_scale=gen_config["cfg_scale"],
)
{% endhighlight %}

---
>## 3. StochasticIOSampler

해당 논문에서 어느 부분이 가장 중요하냐고 하면 나는 당연하게도.. MMDM을 학습할 때 특정 domaion의 detail을 위한 condition들이 라고 생각한다. 결국 MMDM이 완벽하게 잘 학습되었고 좋은 성능을 보여주기 때문에, 뒤에 부분들과 같이 4D Avatar로 만들어질 수 있는 것이라고 생각하기 때문이다.

해당 논문에서도 MMDM의 중요성을 중심으로 설명해주고 있긴 하지만, 동시에 Stochastic부분도 그 다음으로 중요한 부분으로 설명하고 있다.

해당 부분은 논문에서 의도한 바로는, 결국 **reference images를 4장 밖에 사용하지 못하니 이를 diffusion의 각 timestep에 reference images를 4장씩 random으로 sampling하여 diffusion에 적용하여 더 general한 image들을 생성**하자라는 목적이다.

해당 Section에서는 기본적으로 DDIM을 기반으로 하는  Stochastic I/O conditioning에 대한 설명을 간략하게 해본다.

{% highlight python linenos %}
# cap4d/mmdm/sampler.py
class StochasticIOSampler(object)
  def sample(...)
    Parameters:
        S (int): Number of diffusion steps.
        ref_cond (Dict[str, torch.Tensor]): Conditioning images used for reference (ref latents, pose maps, reference masks etc.).
        ref_uncond (Dict[str, torch.Tensor]): Unconditional conditioning images used for reference (zeroed conditioning).
        gen_cond (Dict[str, torch.Tensor]): Conditioning images used for reference (pose maps, reference masks etc.).
        gen_uncond (Dict[str, torch.Tensor]): Unconditional conditioning images used for reference (pose maps, reference masks etc.).
        latent_shape (Tuple[int]): Shape of the latent to be generated (B, C, H, W).
        V (int): Number of views supported by the MMDM.
        R_max (int, optional): Maximum number of reference images to use. Defaults to 4.
        cfg_scale (float, optional): Classifier-free guidance scale. Higher values increase conditioning strength. Defaults to 1.0.
        eta (float, optional): Noise scaling factor for DDIM sampling. 0 means deterministic sampling. Defaults to 0.
        verbose (bool, optional): Whether to print detailed logs during sampling. Defaults to False.    
    Returns:
        torch.Tensor: A tensor representing the generated sample(s) in latent space.
{% endhighlight %}

timestep은 기본적으로 250에서 1까지 reverse로 돌아가게 되고, ```shuffle generated latents```를 수행하는 부분으로 np.random.permuation을 사용하는 것 같다.

{% highlight python linenos %}
	for i, step in enumerate(iterator):
		index = total_steps - i - 1
		ref_batches = np.stack([
        	np.random.permutation(np.arange(n_all_ref))[:R] for _ in range(n_its)], axis=0)
{% endhighlight %}

아래와 같이 batch 단위로 돌면서 위에서 자연스럽게 shuffle된 sampliing을 이용하여 ref와 gen의 condition들의 sample을 뽑아서 사용한다.

{% highlight python linenos %}
for dev_batches in batch_indices:
    for dev_id, dev_batch in enumerate(dev_batches):
        dev_key = list(self.device_model_map)[dev_id]
        dev_device = self.device_model_map[dev_key].device

        curr_ref_cond = dict_sample(ref_cond, ref_batches[dev_batch], device=dev_device)
        curr_ref_uncond = dict_sample(ref_uncond, ref_batches[dev_batch], device=dev_device)

        curr_gen_cond = dict_sample(gen_cond, gen_batches[dev_batch], device=dev_device)
        curr_gen_uncond = dict_sample(gen_uncond, gen_batches[dev_batch], device=dev_device)	
{% endhighlight %}

이제 위에서 sampling된 batch들을 이용하여 $\mathbf{Z}_{\text{ref}}^{\prime},\mathbf{C}_{\text{ref}}^{\prime},\mathbf{C}_{\text{gen}}^\prime$ 값들을 뽑아서 MMDM Model에 적용시켜 e_t를 얻어 latent를 이후에 한꺼번에 업데이트 하는 로직으로 동작하는 것 같다.

{% highlight python linenos %}
  for dev_id, dev_batch in enumerate(dev_batches):
      dev_key = list(self.device_model_map)[dev_id]
      dev_device = self.device_model_map[dev_key].device
      model_uncond, model_t = self.device_model_map[dev_key].apply_model(
                              x_in_list[dev_id], 
                              t_in_list[dev_id], 
                              c_in_list[dev_id],
                              ).chunk(2)
      model_output = model_uncond + cfg_scale * (model_t - model_uncond)
      e_t = model_output[:, R:]  # eps prediction mode, extract the generation samples starting at n_ref
      e_t_list.append(e_t)

  for dev_id, dev_batch in enumerate(dev_batches):
      all_e_t[gen_batches[dev_batch]] += e_t_list[dev_id].to(mem_device)
{% endhighlight %}

이제 해당 부분이 한꺼번에 update하고 latent를 return하는 부분인 것 같은데 정확한 의미를 알지 못해서 일단 코드 첨부만 한다....

{% highlight python linenos %}
alpha_t = self.ddim_alphas.float()[index]
sqrt_one_minus_alpha_t = self.ddim_sqrt_one_minus_alphas[index]
sigma_t = self.ddim_sigmas[index]
alpha_prev_t = torch.tensor(self.ddim_alphas_prev).float()[index]

alpha_prev_t = alpha_prev_t.double()
sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.double()
alpha_t = alpha_t.double()
alpha_prev_t = alpha_prev_t.double()

e_t_factor = -alpha_prev_t.sqrt() * sqrt_one_minus_alpha_t / alpha_t.sqrt() + (1. - alpha_prev_t - sigma_t**2).sqrt()
x_t_factor = alpha_prev_t.sqrt() / alpha_t.sqrt() 

e_t_factor = e_t_factor.float()
x_t_factor = x_t_factor.float()

all_x_T = all_x_T * x_t_factor + all_e_t * e_t_factor
return all_x_T
{% endhighlight %}