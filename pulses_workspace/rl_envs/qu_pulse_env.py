from itertools import product
from typing import *
import numpy as np
import gym
from qutip import basis
from qutip_qip.device import SCQubits
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
    def __init__(self, qc=None):
        if qc is None:
            qc = QubitCircuit(N=1)
            qc.add_gate('X', targets=0)
        self.action_eps = 0.001
        self.thresh_r = 0.999
        self.rew_scale = 10.

        self.current_state = None

        self.processor = SCQubits(num_qubits=qc.N)
        self.orig_tlist, self.orig_coeff = self.processor.load_circuit(qc)
        self.processor.t1 = 50.e3
        self.processor.t2 = 20.e3

        self.n_channels = len(self.orig_coeff)
        self.action_space = gym.spaces.MultiDiscrete(np.repeat(3, self.n_channels))
        self.observation_space = gym.spaces.Box(np.zeros(self.n_channels, dtype=float),
                                                np.ones(self.n_channels, dtype=float) * 2)

        self.basis0 = basis(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> ObsType:
        init_state = None
        if options is not None:
            init_state = options.get('init_state', None)

        if init_state is not None:
            self.current_state = init_state
        else:
            self.current_state = np.ones(self.n_channels)

        return self.current_state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # print('action=', action)
        centered_action = np.array(action) - 1
        delta_action = centered_action * self.action_eps

        self.current_state += delta_action

        new_coeffs = np.array([v * c for v, c in zip(self.orig_coeff.values(), self.current_state)])

        for i, coeff in enumerate(new_coeffs):
            self.processor.pulses[i].coeff = coeff

        final_state = np.array(self.processor.run_state(self.basis0).states[-1])
        final_state = np.abs(final_state)
        prob_ket_one = np.diagonal(final_state)[1]

        r = prob_ket_one * self.rew_scale
        # print(r)
        d = r > self.thresh_r

        return self.current_state, r, d, {'success': d}

    def render(self):
        fig, axs = plt.subplots(self.n_channels)
        for ax, pulse in zip(axs, self.processor.pulses):
            ax.plot(pulse.coeff)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title(pulse.label)
            fig.tight_layout()
        plt.show(block=False)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    qc = QubitCircuit(N=1)
    qc.add_gate('X', targets=0)
    env = QuPulseEnv(qc)
    env.action_eps = 0.001
    o = env.reset()
    rews = []
    for _ in range(10):
        # a = env.action_space.sample()
        a = [0, 1, 1]
        otp1, r, d, _ = env.step(a)
        # env.render()
        rews.append(r)
    plt.plot(rews, marker='x')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.yscale('log')
    plt.show()
