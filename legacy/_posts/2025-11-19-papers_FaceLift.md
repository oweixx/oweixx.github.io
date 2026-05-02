---
layout: post
title: "&#91;Papers&#93; FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads &#40;ICCV 2025&#41;"
date: 2025-11-19
description: Code Review
tags: Paper
categories: Paper
featured: true
---


## FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads
### [[Paper]](https://arxiv.org/pdf/2412.17812)[[Github]](https://github.com/weijielyu/FaceLift)[[Project]](https://www.wlyu.me/FaceLift/)

>**Title:** FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads    
**Journal name & Publication Date:** ICCV 2023-12-23  
**Affiliation:** University of California, Merced, Adobe Research

---
>## 0. Multi-View Diffusion based Generated images

논문에서 제시된 ```View Geneartion``` 부분에 대한 코드 부분으로, Training된 Multi-view Diffusion을 이용하여 Single Image로 부터 각기 다른 Viewing image를 생성하는 부분이다.

```inference.py```에서 main함수에서 model들을 모두 init한 이후에 ```process_single_image```로 넘어가면 본격적으로 진행이 된다.

{% highlight python linenos %}
mv_imgs = unclip_pipeline(
    input_image, 
    None,
    prompt_embeds=color_prompt_embedding,
    guidance_scale=guidance_scale_2D,
    num_images_per_prompt=1, 
    num_inference_steps=step_2D,
    generator=generator,
    eta=1.0,
).images
{% endhighlight %}

타고타고 들어가다 보면 ```mvdiffusion/pipelines/pipeline_mvdiffusion_unclip.py```에 DiffusionPipeline을 상속받는 ```StableUnCLIPImg2ImgPipeline```이 있다. 해당 부분은 "pipeline for text-guided image to image generation using stable uinCLIP"이라고 설명이 적혀 있다.

해당 부분은 논문에서와 같이 text embedding으로 view generation을 하기 때문에 해당 pipeline을 사용하는 것 같다.

결국 pipeline을 다시 재수정한 부분이니 실제로 실행되는 ```__call__```에서의 동작과정에 집중해서 확인해보자.

{% highlight python linenos %}
class StableUnCLIPImg2ImgPipeline(DiffusionPipeline):
{% endhighlight %}

input으로 들어오는 prompt에 대하여 embedding시키고 ```prompte_embeds```형태의 출력으로 받는다.

{% highlight python linenos %}
prompt_embeds = self._encode_prompt(
    prompt=prompt,
    device=device,
    num_images_per_prompt=num_images_per_prompt,
    do_classifier_free_guidance=do_classifier_free_guidance,
    negative_prompt=negative_prompt,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    lora_scale=text_encoder_lora_scale,
)
{% endhighlight %}

input image를 encoder에 넣고 embedding하고 latent형태로 변환시킨다.

{% highlight python linenos %}
image_embeds, image_latents = self._encode_image(
    image_pil=image_pil,
    device=device,
    num_images_per_prompt=num_images_per_prompt,
    do_classifier_free_guidance=do_classifier_free_guidance,
    noise_level=noise_level,
    generator=generator,
)
{% endhighlight %}

위에서 처리된 prompt와 image ebedding 변수들과 latent를 이용하여 Denosing Loop에서 처리한다. diffusion에 대한 지식이 아직은 많이 부족해서 어림짐작해서 일단은 해석해보겠다....

먼저 ```torch.cat([latent_model_input, image_latents], dim=1)``` 부분에서 생성해야 하는 latent와 conditioning으로 들어가는 image latents가 concat되어 input으로 들어간다.

이후에 unet에 직접적으로 들어갈 때는 encoder_hidden_states의 input으로 prompt_embeds가 들어가게 되어 ```predict the noise residual```을 수행하게 된다.

이후에 noisy sample을 step해주어 x_t -> x_t-1 latent를 계산해준다.

{% highlight python linenos %}
# 8. Denoising loop
for i, t in enumerate(self.progress_bar(timesteps)):
    if do_classifier_free_guidance:
        latent_model_input = torch.cat([latents, latents], 0)
    else:
        latent_model_input = latents
    latent_model_input = torch.cat([
            latent_model_input, image_latents
        ], dim=1)
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    # predict the noise residual
    unet_out = self.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        class_labels=image_embeds,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=False)
    
    noise_pred = unet_out
        
    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

    if callback is not None and i % callback_steps == 0:
        callback(i, t, latents)
{% endhighlight %}

---
>## 1. GS-LRM

GSLRM Class는 ```gslrm/model/gslrm.py```에 숨어 있다. 해당 부분도 방대한 코드 양으로 모두 이해하기는 힘들 것 같다. 많은 부분을 생략하고 forward 부분의 코드 구성을 한 번 확인해보자. 사실 forward 부분도 150줄이 넘는다.

args부터 간단하게 살펴보면 batch형태로 들어오는 data는 논문에서와 같이 생성된 Multi-view images와 Camera intrinsics 정보 등이 있고, 출력값으로는 Dictionary형태의 model output이고 여기서 Gaussian정보들을 반환해주고 있으니 해당 정보들로 바로 GS Reconstruction이 가능하다.

{% highlight python linenos %}
def forward(
    self, 
    batch_data: edict, 
    create_visual: bool = False, 
    split_data: bool = True
) -> edict:
    """
    Forward pass of the GSLRM model.
    
    Args:
        batch_data: Input batch containing:
            - image: Multi-view images [batch, views, channels, height, width]
            - fxfycxcy: Camera intrinsics [batch, views, 4]
            - c2w: Camera-to-world matrices [batch, views, 4, 4]
        create_visual: Whether to create visualization outputs
        split_data: Whether to split input/target data
        
    Returns:
        Dictionary containing model outputs including Gaussians, renders, and losses
    """
{% endhighlight %}

눈에 확 띄는 주요한 부분만 모아서 한꺼번에 봐보자. 사실 이 밑의 process들이 다 논문에 나와있는 부분이긴한데 중요한 것 같다.

대략적으로는 ```Patchify & Linear``` 하는 부분이 있고 ```transformer process```를 통과하고 Linear & Unpatchify하여 gaussian_tokens과 image_patch_tokens으로 나뉘고 이를 통해 gaussian parameter를 생성하게 된다. 이후에 pixel-aligned를 하여 gaussian parameter를 예측(?) 하게 되는 일련의 과정인 것 같다. 

{% highlight python linenos %}
# Prepare posed images with Plucker coordinates [batch, views, channels, height, width]
posed_images = self._create_posed_images_with_plucker(input_data)

# Tokenize images into patches
image_patch_tokens = self.patch_embedder(posed_images)  # [batch*views, num_patches, hidden_dim]
_, num_patches, hidden_dim = image_patch_tokens.size()
image_patch_tokens = image_patch_tokens.reshape(
    batch_size, num_views * num_patches, hidden_dim
)  # [batch, views*patches, hidden_dim]

# Prepare Gaussian tokens with positional embeddings
gaussian_tokens = self.gaussian_position_embeddings.expand(batch_size, -1, -1)

# Process through transformer with gradient checkpointing
combined_tokens = self._process_through_transformer(
    gaussian_tokens, image_patch_tokens
)

# Split back into Gaussian and image tokens
num_gaussians = self.config.model.gaussians.n_gaussians
gaussian_tokens, image_patch_tokens = combined_tokens.split(
    [num_gaussians, num_views * num_patches], dim=1
)

# Generate Gaussian parameters from transformer outputs
gaussian_params = self.gaussian_upsampler(gaussian_tokens, image_patch_tokens)

# Generate pixel-aligned Gaussians from image tokens
pixel_aligned_gaussian_params = self.pixel_gaussian_decoder(image_patch_tokens)

# Calculate Gaussian parameter dimensions
sh_degree = self.config.model.gaussians.sh_degree
gaussian_param_dim = 3 + (sh_degree + 1) ** 2 * 3 + 3 + 4 + 1

pixel_aligned_gaussian_params = pixel_aligned_gaussian_params.reshape(
    batch_size, -1, gaussian_param_dim
)  # [batch, views*pixels, gaussian_params]
num_pixel_aligned_gaussians = pixel_aligned_gaussian_params.size(1)

# Combine all Gaussian parameters
all_gaussian_params = torch.cat((gaussian_params, pixel_aligned_gaussian_params), dim=1)

# Convert to final Gaussian format
xyz, features, scaling, rotation, opacity = self.gaussian_upsampler.to_gs(all_gaussian_params)

# Extract pixel-aligned Gaussian positions for processing
pixel_aligned_xyz = xyz[:, -num_pixel_aligned_gaussians:, :]
patch_size = self.config.model.image_tokenizer.patch_size

pixel_aligned_xyz = rearrange(
    pixel_aligned_xyz,
    "batch (views height width patch_h patch_w) coords -> batch views coords (height patch_h) (width patch_w)",
    views=num_views,
    height=height // patch_size,
    width=width // patch_size,
    patch_h=patch_size,
    patch_w=patch_size,
)
{% endhighlight %}