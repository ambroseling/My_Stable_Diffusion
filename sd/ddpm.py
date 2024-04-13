import torch 
import numpy as np

class DDPMSampler:
    def __init__(self,
                generator: torch.Generator,
                num_training_steps=1000,
                beta_start: float=0.00005,
                beta_end: float=0.0120):
        self.beta = torch.linspace(beta_start**0.5,beta_end**0.5,num_training_steps,dtype=torch.float32)**2
        self.alphas = 1.0 - self.beta 
        self.alpha_bar = torch.cumprod(self.alphas,0) #[alpha0 , alpha0 * alpha1, alpha0 * alpha1 * alpha2, ....]
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())
    def set_inference_timesteps(self,num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        # 999, 998, 997, 996... -> 1000 steps
        # 999, 999-20, 999-40, 999-60, 999-80 -> 50 steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0,num_inference_steps)*step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self,original_samples,timestep):
        alpha_bar = self.alpha_bar.to(original_samples.device,dtype = original_samples.dtype)
        timestep = timestep.to(device = original_samples.device)
        mean = alpha_bar[timestep]**0.5
        mean = mean.flatten()
        while len(mean.shape) < len(original_samples.shape):
            mean = mean.unsqueeze(-1)
        stdev = (1.0 - alpha_bar[timestep])**0.5 

        while len(stdev.shape) < len(original_samples.shape):
            stdev = stdev.unsqueeze(-1)
        
        #according to equation for DDPM
        noise = torch.rand(original_samples.shape,generator=self.generator,device = original_samples.device,dtype=original_samples.dtype)
        noisy_samples = mean*original_samples + stdev*noise
        return noisy_samples
    
    def _prev_timestep(self,t):
        prev_t = t - (self.num_training_steps // self.num_inference_steps)
        return prev_t

    def set_strength(self,strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps*strength)
        self.timesteps = self.timesteps[start_step:]


    def step(self,timestep, latents,predicted_noise):
        t = timestep
        prev_t = self._prev_timestep(t)
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_prev_t = self.alpha_bar[prev_t] if prev_t>=0 else self.one
        beta_bar_t = 1 - alpha_bar_t
        beta_bar_prev_t = 1 - alpha_bar_prev_t
        current_alpha_t = alpha_bar_t/alpha_bar_prev_t
        current_beta_t = 1 - current_alpha_t

        #compute predicted orgiinal signal using formula 15 of DDPM paper
        x_0 = (latents - beta_bar_t**0.5 * predicted_noise)/(alpha_bar_t**0.5)

        #compute the coefficients for x_0 and current sample x_t
        x_0_coeff = ((alpha_bar_prev_t**0.5) * current_beta_t) / beta_bar_t
        x_t_coeff = ((current_alpha_t**0.5) * beta_bar_prev_t) / beta_bar_t
        mean = x_0_coeff * x_0 + x_t_coeff*latents

        variance =  ((1 - alpha_bar_prev_t)/(1-alpha_bar_t))*current_beta_t
        variance = torch.clamp(variance,min=1e-20)

        #variance = 0
        if t > 0:
            device = predicted_noise.device
            noise = torch.randn(predicted_noise.shape,generator=self.generator,device=device,dtype=predicted_noise.dtype)
            variance = (variance**0.5)*noise
        prev_mean = mean + variance
        return prev_mean








