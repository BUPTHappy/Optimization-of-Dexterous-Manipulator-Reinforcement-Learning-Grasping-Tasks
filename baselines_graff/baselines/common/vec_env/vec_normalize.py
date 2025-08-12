from . import VecEnvWrapper
import numpy as np

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, norm_keys=None, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
        VecEnvWrapper.__init__(self, venv)
        if use_tf:
            from baselines.common.running_mean_std import TfRunningMeanStd
            self.ob_rms = {key: TfRunningMeanStd(shape=self.observation_space[key].shape, scope='ob_rms') for key in norm_keys} if ob else None
            self.ret_rms = TfRunningMeanStd(shape=(), scope='ret_rms') if ret else None
        else:
            from baselines.common.running_mean_std import RunningMeanStd
            self.ob_rms = {key: RunningMeanStd(shape=self.observation_space[key].shape) for key in norm_keys} if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.norm_keys = norm_keys

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            for key in self.norm_keys:
                self.ob_rms[key].update(obs[key])
                obs[key] = np.clip((obs[key] - self.ob_rms[key].mean) / np.sqrt(self.ob_rms[key].var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self, **kwargs):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset(**kwargs)
        return self._obfilt(obs)
