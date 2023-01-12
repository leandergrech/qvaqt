"""
ADVANCED CIRCUITS
"""
import numpy as np
from qiskit import *

# '''
# OPAQUE GATES
# '''
# from qiskit.circuit import Gate
#
#
# lg_gate = Gate(name='lg_gate', num_qubits=2, params=[])
#
# qr = QuantumRegister(3, 'q')
# circ = QuantumCircuit(qr)
# circ.append(lg_gate, [qr[0], qr[1]])
# circ.append(lg_gate, [qr[1], qr[2]])

# print(circ.draw())

# '''
# COMPOSITE GATES
# '''
# # Build a sub-circuit
# sub_q = QuantumRegister(2)
# sub_circ = QuantumCircuit(sub_q, name='sub_circ')
# sub_circ.h(sub_q[0])
# sub_circ.crz(1, sub_q[0], sub_q[1])
# sub_circ.barrier()
# sub_circ.id(sub_q[1])
# sub_circ.u(1, 2, -2, sub_q[0])
#
# # Convert to a gate and stick it into an arb. place in the bigger circuit
# sub_inst = sub_circ.to_instruction()
#
# qr = QuantumRegister(3, 'q')
# circ = QuantumCircuit(qr)
# circ.h(qr[0])
# circ.cx(qr[0], qr[1])
# circ.cx(qr[1], qr[2])
# circ.append(sub_inst, [qr[1], qr[2]])
#
# # print(circ.draw())
# # Circuits are not immediately decomposed upon conversion to_instruction to allow for
# # higher circuit design abstraction. Decomposition is made with decompose method in
# # circuit:
#
# decomposed_circ = circ.decompose() # Does not modify orig. circuit
# print(decomposed_circ.draw())

'''
PARAMETERIZED CIRCUITS
'''
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter
theta = Parameter('0')
n = 5
qc = QuantumCircuit(5, 1)

qc.h(0)
for i in range(n - 1):
    qc.cx(i, i + 1)
qc.barrier()
qc.rz(theta, range(5))
qc.barrier()

for i in reversed(range(n-1)):
    qc.cx(i, i + 1)
qc.h(0)
qc.measure(0, 0)

# qc.draw('mpl')
# plt.show()

'''
BINDING PARAMETERS TO VALUES
'''
theta_range = np.linspace(0, 2 * np.pi, 128)

circuits = [qc.bind_parameters({theta: theta_val}) for theta_val in theta_range]
print(circuits[-1].draw())

backend = BasicAer.get_backend('qasm_simulator')
job = backend.run(transpile(circuits, backend))
counts = job.result().get_counts()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(theta_range, list(map(lambda c: c.get('0', 0), counts)), '.-', label='0')
ax.plot(theta_range, list(map(lambda c: c.get('1', 0), counts)), '.-', label='1')

ax.set_xticks([i*np.pi / 2 for i in range(5)])
ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
ax.set_xlabel('0', fontsize=14)
ax.set_ylabel('Counts', fontsize=14)
ax.legend(fontsize=14)


plt.show()
