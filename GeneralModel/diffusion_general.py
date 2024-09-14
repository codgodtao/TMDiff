import math
import torch
from torch import nn
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

from core.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from utils.util import res2img


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep):
    if schedule == 'linear':
        scale = 1000 / n_timestep
        beta_start = scale * 1e-6  # 固定初始的beta和结束的beta,减少参数量
        beta_end = scale * 1e-2
        return np.linspace(
            beta_start, beta_end, n_timestep, dtype=np.float64
        )
    elif schedule == "cosine":
        return betas_for_alpha_bar(
            n_timestep,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(schedule)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GeneralDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            loss_type='l1'
    ):
        super(GeneralDiffusion, self).__init__()
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss().to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss().to(device)
        elif self.loss_type == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss().to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',  # 在sr3中，sqrt_alphas_cumprod是传入的noise_level
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod_1',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod_1',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def classifier_free_guidance_sample(self, x, t_input, x_in, guidance, prompt):
        """
        once we have two model:condition and unconditional model,we have two noise to predict
        @param:w the weight to balance conditional and unconditonal
        (w+1)*model(x_t,t,y)-w*model(x_t,t,empty)
        """
        condition = x_in['MS']
        unconditional_conditional = torch.zeros_like(condition)

        noise_condition = self.denoise_fn(x, t_input, condition, x_in['PAN'], prompt)
        noise_uncondition = self.denoise_fn(x, t_input, unconditional_conditional, x_in['PAN'], prompt)
        return (guidance + 1.0) * noise_condition - guidance * noise_uncondition

    @torch.no_grad()
    def p_mean_variance(self, x, t, clip_denoised=True, x_in=None, prompt="QB", guidance=1.0):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)

        time_in = torch.tensor([t + 1] * batch_size, device=noise_level.device).view(batch_size, -1)
        assert time_in.shape == noise_level.shape
        if x_in is not None:
            x_recon = self.predict_start_from_noise(  # 根据预测出来的噪声得到x0
                x, t=t, noise=self.denoise_fn(x, time_in, x_in['PAN'], x_in['MS'], prompt))

        if clip_denoised:
            x_recon = self.dynamic_clip(x_recon, is_static=True)

        model_mean, posterior_log_variance = self.q_posterior(  # 根据x0,xt,t真实分布公式直接计算即可
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_mean_variance_xo(self, x, t, clip_denoised=True, x_in=None, prompt="QB", guidance=1.0):
        # print("p_mean_variance", prompt)
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)

        time_in = torch.tensor([t + 1] * batch_size, device=noise_level.device).view(batch_size, -1)
        assert time_in.shape == noise_level.shape
        if x_in is not None:
            x_recon = self.denoise_fn(x, time_in, x_in['PAN'], x_in['MS'], prompt)

        if clip_denoised:
            x_recon = self.dynamic_clip(x_recon, is_static=True)

        model_mean, posterior_log_variance = self.q_posterior(  # 根据x0,xt,t真实分布公式直接计算即可
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    def dynamic_clip(self, x_recon, is_static=True):
        if is_static:
            x_recon.clamp_(-1., 1.)
        else:
            s = torch.max(torch.abs(x_recon))
            s = s if s > 1 else 1.
            x_recon = x_recon / s
            print(s)
        return x_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, prompt="QB", guidance=1.0):
        # print("p_sample", prompt)
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, x_in=condition_x, prompt=prompt, guidance=guidance)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, prompt="QB", guidance=1.0):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        shape = x_in['Res'].shape
        img = torch.randn(shape, device=device)  # 高斯随机噪声，初始XT
        ret_img = res2img(img, x_in['MS'])
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                      total=self.num_timesteps):
            img = self.p_sample(img, i, condition_x=x_in, prompt=prompt, guidance=guidance, clip_denoised=True)
            if i % sample_inter == 0:  # 用于展示的采样间隔，ret_img最终是一个列表，第一个元素为MS，最后一个结果则是T步去噪的结果
                ret_img = torch.cat([ret_img, res2img(img, x_in['MS'])], dim=0)  # sample出来的res+lms
        if continous:
            return ret_img  # 一连串残差预测结果
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample_by_dpmsolver(self, x_in, prompt):
        device = self.betas.device
        x_T = torch.randn(x_in['Res'].shape, device=device)
        model = self.denoise_fn
        model_kwargs = {"PAN": x_in['PAN'], "MS": x_in['MS'], "prompt": prompt}
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type="x_start",  # or "x_start" or "v" or "score"
            model_kwargs=model_kwargs,
        )

        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                                correcting_x0_fn="dynamic_thresholding")

        x_sample = dpm_solver.sample(
            x_T,
            steps=30,
            order=3,
            skip_type="logSNR",
            method="singlestep",
            denoise_to_zero=True
        )
        ret_img = res2img(x_sample, x_in['MS'])  # RES [-1,1]转变[0,1]

        return ret_img  # 在[0,1]的范围内

    @torch.no_grad()
    def sample_by_dpmsolver_noise(self, x_in, prompt):
        # 高斯分布的图像更容易重建
        device = self.betas.device
        x_T = torch.randn(x_in['Res'].shape, device=device)
        model = self.denoise_fn
        model_kwargs = {"PAN": x_in['PAN'], "MS": x_in['MS'], "wav": x_in['wav'], "prompt": prompt}
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type="noise",  # or "x_start" or "v" or "score"
            model_kwargs=model_kwargs,
        )

        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                                correcting_x0_fn="dynamic_thresholding")

        x_sample = dpm_solver.sample(
            x_T,
            steps=50,
            order=3,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True
        )

        ret_img = res2img(x_sample, x_in['MS'])  # RES [-1,1]转变[0,1]

        return ret_img  # 在[0,1]的范围内

    @torch.no_grad()
    def sample_by_regression(self, x_in, prompt):
        device = self.betas.device
        x_T = torch.randn(x_in['Res'].shape, device=device)
        time_in = torch.tensor([1000 + 1] * 1, device=device).view(1, -1)
        x_recon = self.denoise_fn(x_T, time_in, x_in['PAN'], x_in['MS'], x_in['wav'], prompt)

        ret_img = res2img(x_recon, x_in['MS'])  # RES [-1,1]转变[0,1]

        return ret_img  # 在[0,1]的范围内

    @torch.no_grad()
    def sample_by_dpmsolver_guidance(self, x_in, prompt, guidance):
        device = self.betas.device
        x_T = torch.randn(x_in['Res'].shape, device=device)
        condition = x_in['PAN']
        unconditional_condition = torch.zeros_like(condition)
        model = self.denoise_fn
        model_kwargs = {"MS": torch.cat([torch.zeros_like(x_in["MS"]), x_in["MS"]]), "prompt": prompt}
        guidance_scale = guidance

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type="noise",  # or "x_start" or "v" or "score"
            model_kwargs=model_kwargs,
            guidance_type="classifier-free",
            condition=condition,
            unconditional_condition=unconditional_condition,
            guidance_scale=guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                                correcting_x0_fn="dynamic_thresholding")

        x_sample = dpm_solver.sample(
            x_T,
            steps=50,
            order=2,
            skip_type="logSNR",  # recommended for high-resolution time_uniform
            method="multistep"
        )
        ret_img = res2img(x_sample, x_in['MS'])  # RES [-1,1]转变[0,1]

        return ret_img

    @torch.no_grad()
    def super_resolution(self, x_in, continous, prompt, guidance):
        return self.p_sample_loop(x_in, prompt)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):  # 扩散过程的加噪函数
        noise = default(noise, lambda: torch.randn_like(x_start))
        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def p_losses_dynamic(self, x_in, prompt=None):
        # 训练过程使用的L1损失函数
        x_start = x_in['Res']  # GT-MS [-1,1]但是不服从正态分布的
        [b, c, h, w] = x_start.shape
        time_in = torch.from_numpy(np.random.randint(1, self.num_timesteps + 1, size=b))
        if b > 1:
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                self.sqrt_alphas_cumprod_prev[time_in]).to(x_start.device)
        else:
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[time_in]]).to(x_start.device)
        # continuous_sqrt_alpha_cumprod = torch.gather(self.sqrt_alphas_cumprod_prev, 0, time_in).to(x_start.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        time_in = time_in.to(x_start.device).view(b, -1)
        noise = torch.randn_like(x_start)  # 需要预测的随机噪声
        x_noisy = self.q_sample(  # 对X0进行加噪得到xt，拓展[b,1]维度为[b,1,1,1],才可以和[b,c,h,w]的x_start,noise进行运算
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        x_recon = self.denoise_fn(x_noisy, time_in, x_in['PAN'], x_in['MS'], prompt)
        loss = self.loss_func(x_start, x_recon)
        return loss

    def predict_start_from_noise_for_x_hat(self, x_t, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, noise):
        return sqrt_recip_alphas_cumprod * x_t - \
               sqrt_recipm1_alphas_cumprod * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod_1[t] * x_t - \
               self.sqrt_recipm1_alphas_cumprod_1[t] * noise

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def classifier_free_guidance_train(self, cond, p_uncond):
        """"
        classifier_free guidance ,jointly train a diffusion model with classifier-free guidance
        @param:p_uncond probability of unconditional training
        pan_guidance diffusion model
        """

        rand = torch.rand(1)
        if rand > p_uncond:
            return cond
        else:
            # print("unconditional")
            return torch.zeros_like(cond)

    def forward(self, x, *args, **kwargs):
        return self.p_losses_dynamic(x, *args, **kwargs)
