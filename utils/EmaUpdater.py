import torch
from GeneralModel.model import DDPM as diffusion

class EmaUpdater(object):
    """exponential moving average model updater
    when iteration > start_iter, update the ema model
    else load the model params to ema model
    """

    def __init__(
            self,
            model: diffusion,
            ema_model: diffusion,
            decay=0.9999,
            start_iter=0,
    ) -> None:
        self.model = model
        self.ema_model = ema_model
        self.decay = decay
        self.start_iter = start_iter
        self.iteration = start_iter

    @torch.no_grad()
    def update(self, iteration):
        self.iteration = iteration
        if iteration > self.start_iter:
            for p, p_ema in zip(
                    self.model.netG.denoise_fn.parameters(),
                    self.ema_model.netG.denoise_fn.parameters(),
            ):
                p_ema.data = p_ema.data * self.decay + \
                             p.data * (1 - self.decay)
        else:
            for p, p_ema in zip(
                    self.model.netG.denoise_fn.parameters(),
                    self.ema_model.netG.denoise_fn.parameters(),
            ):
                p_ema.data = p.data.clone().detach()

    def load_ema_params(self):
        # load ema params to model
        self.model.netG.denoise_fn.load_state_dict(self.ema_model.netG.denoise_fn.state_dict())

    def load_model_params(self):
        # load model params to ema model
        self.ema_model.netG.denoise_fn.load_state_dict(self.model.netG.denoise_fn.state_dict())

    @property
    def on_fly_model_state_dict(self):

        if hasattr(self.model, "module"):
            model = self.model.module.netG.denoise_fn
        else:
            model = self.model.netG.denoise_fn

        return model.state_dict()

    @property
    def ema_model_state_dict(self):
        if hasattr(self.model, "module"):
            model = self.ema_model.netG.module.denoise_fn
        else:
            model = self.ema_model.netG.denoise_fn

        return model.state_dict()
