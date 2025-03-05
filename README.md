## Spherical Dense-Text-to-Image Synthesis

# Mask Preprocessing
```python
def preprocess_mask(self, mask_path, h, w, device):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    temp = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], w*2)).to(device)
    temp[:, :, :, 64:192] = mask
    temp[:, :, :, 0:64] = mask[:, :, :, 64:128]
    temp[:, :, :, 192:256] = mask[:, :, :, 0:64]
    return temp
```

# Bootstrapping Backgrounds
```python
def get_random_background(self, n_samples):
    # sample random background with a constant rgb value
    backgrounds = torch.rand(n_samples, 3, device=self.device, dtype=torch.float16)[:, :, None, None].repeat(1, 1, 512, 1024)
    return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])
```

# StitchDiffusion Modification
```python
##########################
## MultiStitchDiffusion ##
##########################

if mask_paths == "":
    fg_masks = torch.cat([torch.ones(1, 1, height // 8, width // 8).to(self.device)])
    bootstrapping = 0
else:
    mask_paths = mask_paths.split(';')
    fg_masks = torch.cat(
        [self.preprocess_mask(mask_path, height // 8, width // 16, self.device) for mask_path in mask_paths])
    bootstrapping_backgrounds = get_random_background(bootstrapping)

bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
bg_mask[bg_mask < 0] = 0
masks = torch.cat([bg_mask, fg_masks])

noise = latents.clone().repeat(len(mask_paths), 1, 1, 1)
views_t = get_views(height, width, stride=stride)
count_t = torch.zeros_like(latents)
value_t = torch.zeros_like(latents)

for i, t in enumerate(tqdm(timesteps)):

    count_t.zero_()
    value_t.zero_()

    # initialize the value of latent_view_t
    temp = latents[:, :, :, 64:192]
    masks_view = masks[:, :, :, 64:192]

    # pre-denoising operations twice on the stitch block
    for ii_md in range(2):

        masks_view[:, :, :, 0:64] = masks[:, :, :, 192:256]
        masks_view[:, :, :, 64:128] = masks[:, :, :, 0:64]
        temp[:, :, :, 0:64] = latents[:, :, :, 192:256]
        temp[:, :, :, 64:128] = latents[:, :, :, 0:64]
        latent_view_t = temp.repeat(len(mask_paths) + 1, 1, 1, 1)

        if i < bootstrapping:
            bg = bootstrapping_backgrounds[torch.randint(0, bootstrapping, (len(mask_paths),))]
            bg = self.scheduler.add_noise(bg, noise[:, :, :, 64:192], t)
            latent_view_t[1:] = latent_view_t[1:] * masks_view[1:] + bg * (1 - masks_view[1:])

        # expand the latents if we are doing classifier free guidance
        latent_model_input = latent_view_t.repeat((num_latent_input, 1, 1, 1))
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latent_view_denoised = self.scheduler.step(noise_pred, t, latent_view_t, **extra_step_kwargs).prev_sample

        value_t[:, :, :, 192:256] += (latent_view_denoised[:, :, :, 0:64] * masks_view[:, :, :, 0:64]).sum(dim=0,
                                                                                                           keepdims=True)
        count_t[:, :, :, 192:256] += masks_view[:, :, :, 0:64].sum(dim=0, keepdims=True)

        value_t[:, :, :, 0:64] += (latent_view_denoised[:, :, :, 64:128] * masks_view[:, :, :, 64:128]).sum(dim=0,
                                                                                                            keepdims=True)
        count_t[:, :, :, 0:64] += masks_view[:, :, :, 64:128].sum(dim=0, keepdims=True)

    # same denoising operations as what MultiDiffusion does
    for h_start, h_end, w_start, w_end in views_t:

        masks_view = masks[:, :, h_start:h_end, w_start:w_end]
        latent_view_t = latents[:, :, h_start:h_end, w_start:w_end].repeat(len(mask_paths) + 1, 1, 1, 1)
        if i < bootstrapping:
            bg = bootstrapping_backgrounds[torch.randint(0, bootstrapping, (len(mask_paths),))]
            bg = self.scheduler.add_noise(bg, noise[:, :, h_start:h_end, w_start:w_end], t)
            latent_view_t[1:] = latent_view_t[1:] * masks_view[1:] + bg * (1 - masks_view[1:])

        # expand the latents if we are doing classifier free guidance
        latent_model_input = latent_view_t.repeat((num_latent_input, 1, 1, 1))
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latent_view_denoised = self.scheduler.step(noise_pred, t, latent_view_t, **extra_step_kwargs).prev_sample

        value_t[:, :, h_start:h_end, w_start:w_end] += (latent_view_denoised * masks_view).sum(dim=0, keepdims=True)
        count_t[:, :, h_start:h_end, w_start:w_end] += masks_view.sum(dim=0, keepdims=True)

    latents = torch.where(count_t > 0, value_t / count_t, value_t)

if return_latents:
    return (latents, False)

latents = 1 / 0.18215 * latents
if vae_batch_size >= batch_size:
    image = self.vae.decode(latents).sample
else:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    images = []
    for i in tqdm(range(0, batch_size, vae_batch_size)):
        images.append(
            self.vae.decode(latents[i: i + vae_batch_size] if vae_batch_size > 1 else latents[i].unsqueeze(0)).sample
        )
    image = torch.cat(images)

image = (image / 2 + 0.5).clamp(0, 1)

# global cropping operation
image = image[:, :, :, 512:1536]

# we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
image = image.cpu().permute(0, 2, 3, 1).float().numpy()
```
