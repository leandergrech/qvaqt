from itertools import product
from typing import *
import numpy as np
import gym
from qutip import basis
from qutip_qip.device import SCQubits
import scipy.interpolate as interp
from qutip_qip.circuit import QubitCircuit

ObsType = TypeVar("QuantumPulse", gym.spaces.Box, list, np.ndarray)
ActType = TypeVar("QuantumPulseCorrection", list, np.ndarray)


def get_discrete_actions(n_act, act_dim=3):
    all_actions = [list(item) for item in product(*np.repeat([[item for item in range(act_dim)]], n_act, axis=0))]
    if n_act == 1:
        all_actions = [item[0] for item in all_actions]
    return all_actions


def get_reduced_discrete_actions(n_act, act_dim=3):
    all_actions = np.vstack([np.zeros(n_act),
                      np.diag(-np.ones(n_act)),
                      np.diag(np.ones(n_act))]).astype(int) + 1
    return all_actions.tolist()


class QuPulseEnv(gym.Env):
    ABS_REW_IMPROVEMENT = 1e-4  # Added to fidelity then scaled by self.rew_scale
    def __init__(self, qc=None, processor_params=None):
        if qc is None:
            qc = QubitCircuit(N=1)
            qc.add_gate('X', targets=0)
        # self.action_eps = 0.001
        self.action_eps = 0.01
        self.thresh_r = 0.999
        self.rew_scale = 10.
        self.max_steps = 20

        self.max_obs_scale = 10.

        self.it = 0

        self.current_state = None

        if processor_params is None: processor_params = dict(t1=50e3, t2=20e3)

        self.processor = SCQubits(num_qubits=qc.N, **processor_params)
        self.orig_tlist, self.orig_coeff = self.processor.load_circuit(qc)
        # self.processor.t1 = 50.e3
        # self.processor.t2 = 20.e3

        self.n_channels = len(self.orig_coeff)
        self.action_space = gym.spaces.MultiDiscrete(np.repeat(3, self.n_channels * 2))
        # Only amplitudes or duration
        self.observation_space = gym.spaces.Box(np.zeros(self.n_channels, dtype=float),
                                                np.repeat(np.inf, self.n_channels))
        # # Amplitudes and duration
        # self.observation_space = gym.spaces.Box(np.zeros(self.n_channels * 2, dtype=float),
        #                                         np.ones(self.n_channels * 2, dtype=float) * 2)

        self.basis0 = basis(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> ObsType:
        init_state = np.ones(self.observation_space.shape[0])
        if options is not None:
            init_state = options.get('init_state', init_state)

        self.thresh_r = (np.diagonal(np.abs(np.array(self.processor.run_state(self.basis0).states[-1])))[1] + \
                         self.ABS_REW_IMPROVEMENT) * self.rew_scale

        self.current_state = init_state
        self.it = 0

        return self.current_state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # print('action=', action)
        centered_action = np.array(action) - 1
        delta_action = centered_action * self.action_eps

        self.current_state += delta_action
        self.current_state = np.clip(self.current_state, a_min=self.action_eps, a_max=self.max_obs_scale)

        # # Only amplitudes
        # for i in range(self.n_channels):
        #     self.scale_pulse_amp(self.current_state[i], i)
        # Only durations
        for i in range(self.n_channels):
            self.scale_pulse_duration(self.current_state[i], i)

        # # Amplitudes and durations
        # for i in range(self.n_channels * 2):
        #     if i < self.n_channels:
        #         self.scale_pulse_amp(self.current_state[i], i)
        #     else:
        #         self.scale_pulse_duration(self.current_state[i], i % 3)

        final_state = np.array(self.processor.run_state(self.basis0).states[-1])
        final_state = np.abs(final_state)
        prob_ket_one = np.diagonal(final_state)[1]

        r = prob_ket_one * self.rew_scale

        success = r > (self.thresh_r * self.rew_scale)

        self.it += 1
        d = success or self.it >= self.max_steps

        return self.current_state, r, d, {'success': success}

    def scale_pulse_amp(self, scale, chidx):
        ogp = [item for item in self.orig_coeff.values()][chidx]
        self.processor.pulses[chidx].coeff = ogp * scale


    def scale_pulse_duration(self, scale, chidx):
        ogp = [item for item in self.orig_coeff.values()][chidx]
        sz = ogp.size
        nsz = int(scale * sz)
        coeff_interp = interp.interp1d(np.arange(sz), ogp)
        self.processor.pulses[chidx].coeff = coeff_interp(np.linspace(0, sz - 1, nsz))
        self.processor.pulses[chidx].tlist = np.arange(nsz).astype(float)

    def render(self):
        fig, axs = plt.subplots(self.n_channels)
        for ax, pulse in zip(axs, self.processor.pulses):
            ax.plot(pulse.coeff)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title(pulse.label)
            fig.tight_layout()
        plt.show(block=False)

    def __repr__(self):
        return f'QuPulseEnv_{self.observation_space.shape[0]}obsx{self.action_space.shape[0]}act'

def run_an_episode():
    from qu_pulse_env import QuPulseEnv
    from qutip_qip.circuit import QubitCircuit
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from pulses_workspace.utils import grid_on
    mpl.rcParams['font.size'] = 25
    from tqdm import trange

    qc = QubitCircuit(N=1)
    qc.add_gate('X', targets=0)
    # qc.add_gate('X', targets=0)
    # qc.add_gate('X', targets=0)
    # processor_params = dict(zz_crosstalk=True, t1=50.e3, t2=20.e3)
    processor_params = dict(zz_crosstalk=True,t1=141.72e3, t2=31.22e3, wq=[5.26], alpha=[-0.33983]) # nairobi settings

    env = QuPulseEnv(qc=qc, processor_params=processor_params)
    print(repr(env))
    env.action_eps = 3e-1
    env.rew_scale = 1.
    env.max_steps = 80
    env.max_obs_scale = 5.
    ch = 'all'

    # o = env.reset(options=dict(init_state=[1, 0, 0]))
    o = env.reset()
    d = False

    obses = []
    acts = []
    rews = []

    for _ in trange(env.max_steps):
        if env.it < 3:   # Do nothing
            a = [1, 1, 1]
        elif env.it < 25:   # Bump down to min limit
            if ch == 'sx0':
                a = [0, 1, 1]   # sx0
            elif ch == 'sz0':
                a = [1, 0, 1]   # sz0
            elif ch == 'sy0':
                a = [1, 1, 0]   # sy0
            elif ch == 'all':
                a = [0, 0, 0]   # all
        elif env.it >= 40:  # Bump up to max limit
            if ch == 'sx0':
                a = [2, 1, 1]   # sx0
            elif ch == 'sz0':
                a = [1, 2, 1]   # sz0
            elif ch == 'sy0':
                a = [1, 1, 2]   # sy0
            elif ch == 'all':
                a = [2, 2, 2]   # all
        # a = env.action_space.sample()
        otp1, r, d, info = env.step(a)

        acts.append(a)
        obses.append(otp1.copy())
        rews.append(r)
        o = otp1

    rews = np.array(rews)

    fig, axs = plt.subplots(3)
    for ax, oc, nc in zip(axs, env.orig_coeff.items(), env.processor.pulses):
        ax.plot(oc[1], c='k', label=oc[0])
        ax.plot(nc.coeff, c='r', ls='--')
        ax.legend(loc='best')

    fig, axs = plt.subplots(3, figsize=(18, 14), gridspec_kw={'height_ratios':[3, 3, 5]})
    fig.suptitle(processor_params)

    ax = axs[0]
    print([item for item in env.orig_coeff.keys()])
    lss = ('solid', 'dashed', 'dotted')
    keys = [item for item in env.orig_coeff.keys()]
    for i, obs in enumerate(np.array(obses).T):
        label = keys[i % 3]
        label += '_amp' if i / 3 < 1 else '_duration'
        ax.plot(obs, marker='x', ls=lss[i % 3], lw=3, label=label)
    ax.legend(loc='best', ncol=2, prop={'size':10})
    ax.set_ylabel('Pulse scaling')

    ax = axs[1]
    for i, act in enumerate(np.array(acts).T):
        label = keys[i % 3]
        label += '_amp' if i / 3 < 1 else '_duration'
        ax.plot(env.action_eps * (act - 1), marker='o', ls=lss[i%3], lw=3, label=label)
    ax.legend(loc='best', ncol=2, prop={'size':10})
    ax.set_ylim((-env.action_eps*1.2, env.action_eps*1.2))
    ax.set_ylabel('Discrete actions')

    ax = axs[2]
    ax.plot(rews, marker='x')
    # for item in rews*100.: print(f'{item:.4f}%', end='\t')
    # print()
    ax.set_yscale('log')
    ax.set_ylabel('Rewards')
    max_rews = max(rews)
    ax.set_title(f'Max fidelity = {max_rews/env.rew_scale}')
    ax.axhline(np.max(rews), ls='dashed', color='k')
    grid_on(ax=ax, axis='y', major_loc=0.1, minor_loc=1e-2)

    for ax in axs:
        ax.axvline(np.argmax(rews), ls='dashed', color='k')
        grid_on(ax=ax, axis='x', major_loc=5, minor_loc=1)
        ax.set_xlabel('Steps')

    fig.tight_layout()
    plt.savefig(f'../rl_analysis/results/{ch}_eps{env.action_eps}_scaling_duration.png')
    plt.show()

if __name__ == '__main__':
    run_an_episode()
