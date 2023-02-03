import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, ptrace
from qutip.control.pulsegen import PulseGenGaussian
from qutip.operators import sigmaz
from qutip_qip.pulse import Pulse
from qutip.three_level_atom import three_level_ops
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import OptPulseProcessor, SCQubits, SCQubitsModel, Processor

print(sigmaz())
print(three_level_ops())

qc = QubitCircuit(N=1)
qc.add_gate('X', targets=0)

n_levels = 3

# gauss_gen = PulseGenGaussian()
# coeff = gauss_gen.gen_pulse(0, 1)
# tlist = np.arange(len(coeff) + 1)
# spline_kind = "step_func"
# gauss_pulse = Pulse(sigmaz(), 0, tlist=tlist, coeff=coeff, spline_kind="step_func")

processor = SCQubits(num_qubits=1, spline_kind='step_func')
# processor.add_pulse(gauss_pulse)
tlist, coeffs = processor.load_circuit(qc)
processor.t1 = 50.e3
processor.t2 = 20.e3
# processor.pulses[0].coeff[20:40] = 0.

old_pulse = processor.pulses[0].coeff
# processor.plot_pulses(title="Control pulse of SCQubits", figsize=(8, 4), dpi=100, show_axis=True)
new_pulse = old_pulse * 0.5
processor.pulses[0].coeff = new_pulse
# processor.plot_pulses(title="Control pulse of SCQubits", figsize=(8, 4), dpi=100, show_axis=True)
plt.plot(old_pulse)
plt.plot(new_pulse)

psi0 = basis(n_levels)
result = processor.run_state(init_state=psi0)

prob0 = np.abs(np.array([s[0,0] for s in result.states]))
prob1 = np.abs(np.array([s[1,1] for s in result.states]))
prob2 = np.abs(np.array([s[2,2] for s in result.states]))

fig, ax = plt.subplots()
ax.plot(prob0, label='Prob. in state |0>')
ax.plot(prob1, label='Prob. in state |1>')
ax.plot(prob2, label='Prob. in state |2>')
# ax.plot(np.sum([prob0, prob1, prob2], axis=0))
ax.set_yscale('log')
ax.set_ylim((1e-8, 1.2))
ax.set_xlabel('Time')
ax.set_ylabel('Probability')
ax.legend(loc='best')


plt.show()
