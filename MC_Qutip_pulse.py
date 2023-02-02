import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, ptrace
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import OptPulseProcessor, SCQubits, SCQubitsModel

qc = QubitCircuit(N=2)
qc.add_gate("SNOT", targets=0)
qc.add_gate("CNOT", controls=0, targets=1)

basis00 = basis([3, 3], [0, 0])
psi0 = basis([3, 3], [0, 0])

# processor = SCQubits(num_qubits=2)
# processor.load_circuit(qc)
# processor.t1 = 50.e3
# processor.t2 = 20.e3

# processor.plot_pulses(title="Control pulse of SCQubits", figsize=(8, 4), dpi=100)
# plt.show()
#
# result = processor.run_state(init_state=psi0)
# print("Probability of measuring state 00:")
# print(np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)[0, 0]))

setting_args = {"SNOT": {"num_tslots": 6, "evo_time": 2}, "X": {"num_tslots": 1, "evo_time": 0.5},
				"CNOT": {"num_tslots": 12, "evo_time": 5}}
processor = OptPulseProcessor(num_qubits=2, model=SCQubitsModel(2, dims=[3, 3]), dims=[3, 3])
processor.load_circuit(qc, setting_args=setting_args, merge_gates=False, verbose=True, amp_ubound=5, amp_lbound=0)

processor.plot_pulses(title="Control pulse of OptPulseProcessor", figsize=(8, 4), dpi=100)
plt.show()

result = processor.run_state(init_state=psi0)
print("Probability of measuring state 00:")
print(np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)[0, 0]))
