---
layout: post
title: "&#91;Papers&#93; One-Step Image Translation with Text-to-Image Models &#40;Preprint&#41;"
date: 2025-11-20
description: Code Review
tags: Paper
categories: Paper
featured: true
---


## One-Step Image Translation with Text-to-Image Models
### [[Paper]](https://arxiv.org/abs/2403.12036)[[Github]](https://github.com/GaParmar/img2img-turbo)

>**Title:** One-Step Image Translation with Text-to-Image Models    
**Journal name & Publication Date:** Preprint 2024-03-18  
**Affiliation:** Carnegie Mellon University, Adobe Research

---
>## 0. Paired Image Translation (pix2pix-turbo)

Paired Image Translation은 애초에 train과 target 쌍이 있는 task의 translation을 의미한다. 

image의 edge와 prompt를 주면 해당 pormpt에 맞게 egde의 structure를 따라 image가 생성이 되는 형식이다.

sketch to image도 비슷하게 "찰떡같이 알아듣는다."를 모델로 표현한 느낌이다. input으로 들어오는 입력으로 들어오는 sketch image에 대하여 prompt로 직접적으로 바로 표현해준다.

{% highlight python linenos %}
# edge to image
python src/inference_paired.py --model_name "edge_to_image" \
    --input_image "assets/examples/bird.png" \
    --prompt "a blue bird" \
    --output_dir "outputs"

# sketch to image
python src/inference_paired.py --model_name "sketch_to_image_stochastic" \
--input_image "assets/examples/sketch_input.png" --gamma 0.4 \
--prompt "ethereal fantasy concept art of an asteroid. magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy" \
--output_dir "outputs"
{% endhighlight %}

먼저 공통적으로 사용하는 ```Pix2Pix_Turbo```Model을 확인해보자.

init을 살펴보면, 공통적으로 Condition image에 대한 vae encoder, decoder module들이 있는 것이 보이고, skip connection conv와 unet이 초기의 ```sd-turbo``` 깡통으로 초기화 되어있는 것 같다.

이후에 두 task에 맞게 달라지는 부분은 각 task에 맞게 학습이 된 sd-turbo unet구조를 가중치를 불러와서 갖다 붙이는 부분이다. 그 외에 둘의 차이점이 있는 부분은 없다. 공통적으로 task에 맞게 model을 불러온 뒤 ```LoRA Adapter```형태로 vae.add_adapter하는 부분이 보인다.

{% highlight python linenos %}
class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        
        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
		# elif pretrained_name == "sketch_to_image_stochastic" :
        	# ...
{% endhighlight %}

이제 해당 model에 forward형식으로 입력만 넣어주면 vae와 unet구조를 이용해서 image를 생성해 낼 수 있다.

먼저 ```edge_to_image```인 경우를 살펴보자. 이때는 본문의 forward parameter에서도 알 수 있지만, deterministic이 기본 default값 True로 고정되어 있다. 그렇게 ```if deterministic:``` 부분으로 호출이 된다. model 구조는 논문의 figure에서 잘 설명되어 있듯이, ```vae.encoder -> unet_encoder -> unet_decoder -> vae.decoder``` 순서로 image가 생성이 되게 된다.

다음으로 ```sketch_to_image_stochastic``` 같은 경우는 edge to image와 다르게 deterministic parameter가 False로 고정 되어 있어 else 문으로 들어가게 된다.
pipeline 구조의 형태는 동일하지만, ```weight = r``` 형태를 이용해서 각 부분 적으로 weight로 기능을 조정해주는 부분이 존재한다. 해당 부분의 특이점 말고는 위와 동일한 형식이다.

{% highlight python linenos %}     
    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
        if deterministic:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image
{% endhighlight %}

그렇게 위에 형식에 맞게 제작이 된 model을 이용해서 prompt와 preprocessing된 image를 model에 넣어주면 output_image를 생성해낼 수 있다.

```sketch_to_iamge_stochastic```같은 경우는 deterministic하게 생성할 것인지에 대한 diverse 조절이 가능하다는 이야기가 논문의 본문 3.4절 Extension에 등장하게 된다. 해당 부분을 기능으로 구현했다는 부분이 눈에 띄는 부분이다.

{% highlight python linenos %}    
# translate the image
with torch.no_grad():
    if args.model_name == 'edge_to_image':
        canny = canny_from_pil(input_image, args.low_threshold, args.high_threshold)
        canny_viz_inv = Image.fromarray(255 - np.array(canny))
        canny_viz_inv.save(os.path.join(args.output_dir, bname.replace('.png', '_canny.png')))
        c_t = F.to_tensor(canny).unsqueeze(0).cuda()
        if args.use_fp16:
            c_t = c_t.half()
        output_image = model(c_t, args.prompt)

    elif args.model_name == 'sketch_to_image_stochastic':
        image_t = F.to_tensor(input_image) < 0.5
        c_t = image_t.unsqueeze(0).cuda().float()
        torch.manual_seed(args.seed)
        B, C, H, W = c_t.shape
        noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
        if args.use_fp16:
            c_t = c_t.half()
            noise = noise.half()
        output_image = model(c_t, args.prompt, deterministic=False, r=args.gamma, noise_map=noise)
{% endhighlight %}

---
>## 1. Unpaired Image Translation (CycleGAN-Turbo)

위의 paried image translation과는 다른 inference.py를 사용하고 있다. 해당 부분에서 Model의 구조가 다를 것이라는 것을 어느정도 예측해볼 수 있다.

{% highlight python linenos %}
# day to night
python src/inference_unpaired.py --model_name "day_to_night" \
    --input_image "assets/examples/day2night_input.png" --output_dir "outputs"

# night to day
python src/inference_unpaired.py --model_name "night_to_day" \
    --input_image "assets/examples/night2day_input.png" --output_dir "outputs"

# clear to rainy
python src/inference_unpaired.py --model_name "clear_to_rainy" \
    --input_image "assets/examples/clear2rainy_input.png" --output_dir "outputs"

# rainy to clear
python src/inference_unpaired.py --model_name "rainy_to_clear" \
    --input_image "assets/examples/rainy2clear_input.png" --output_dir "outputs"
{% endhighlight %}

사실 논문의 본문에서 알 수 있듯이 unpaired image translation에서는 CycleGAN-Turbo 형태로 사용한다고 나와 있다. 밑의 model의 형태를 한 번 간단하게 살펴보자

이 부분은 위에서 본 pix2pix-turbo의 init과 아예 똑같은 형식이다. vae와 unet을 기본 형태로 정의해놓고 pretrained_name에 맞게 model을 불러온다. 재미 있는건 ```self.direction```이라는 부분이 있는데, 여기서 ```a2b, b2a``` 형식으로 있다. 이 부분은 본문에서도 나와있듯이 GAN의 형태를 위해 ```f(b2a(a2b(x)))``` 형태로 다시 복원할 때 사용하기 위한 부분인 것 같다.

{% highlight python linenos %}
class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae
        if pretrained_name == "day_to_night":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the night"
            self.direction = "a2b"
        elif pretrained_name == "night_to_day":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        elif pretrained_name == "clear_to_rainy":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in heavy rain"
            self.direction = "a2b"
        elif pretrained_name == "rainy_to_clear":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
{% endhighlight %}

이후에는 간단하게 아래와 같이 foward를 진행할 수 있다.

{% highlight python linenos %}
@staticmethod
def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb):
    B = x.shape[0]
    assert direction in ["a2b", "b2a"]
    x_enc = vae_enc(x, direction=direction).to(x.dtype)
    model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
    x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
    x_out_decoded = vae_dec(x_out, direction=direction)
    return x_out_decoded

def forward(self, x_t, direction=None, caption=None, caption_emb=None):
    if direction is None:
        assert self.direction is not None
        direction = self.direction
    if caption is None and caption_emb is None:
        assert self.caption is not None
        caption = self.caption
    if caption_emb is not None:
        caption_enc = caption_emb
    else:
        caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt").input_ids.to(x_t.device)
        caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()
    return self.forward_with_networks(x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc)
{% endhighlight %}