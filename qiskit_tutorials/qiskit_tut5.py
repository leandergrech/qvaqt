import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, pulse, transpile
from qiskit.pulse.library import Gaussian
from qiskit.providers.fake_provider import FakeValencia
from qiskit.circuit import Gate

'''
BUILD CIRCUIT
'''
c1 = QuantumCircuit(2, 2)
c1.h(0)
c1.cx(0, 1)
c1.measure(0, 0)
c1.measure(1, 1)

# c1.draw('mpl')
'''
BUILD CALIBRATIONS
'''
backend = FakeValencia()

with pulse.build(backend, name='hadamard') as h_q0:
    pulse.play(Gaussian(duration=128, amp=0.1, sigma=16), pulse.drive_channel(0))

# h_q0.draw()

'''
LINK CALIBRATIONS TO CIRCUIT
'''
c1.add_calibration('h', [0], h_q0)

c1 = transpile(c1, backend)
# print(backend.configuration().basis_gates)
# c1.draw('mpl')

'''
CUSTOM GATES
'''

c2 = QuantumCircuit(1, 1)
custom_gate = Gate('lg_gate', 1, [3.14, 1])
c2.append(custom_gate, [0])
m2 = c2.measure(0,0)
# c2.draw('mpl')

with pulse.build(backend, name='lg') as my_sched:
    pulse.play(Gaussian(duration=64, amp=0.2, sigma=8), pulse.drive_channel(0))

c2.add_calibration('lg_gate', [0], my_sched, [3.14, 1])

c2 = transpile(c2, backend)
c2.draw('mpl', idle_wires=False)
plt.show()

'''
COMMON ERRORS
'''
c3 = QuantumCircuit(2, 2)
c3.append(custom_gate, [1])

from qiskit import QiskitError
try:
    c3 = transpile(c3, backend)
except QiskitError as e:
    print(e)
    # "Cannot unroll the circuit to the given basis, ['id', 'rz', 'sx', 'x', 'cx']. Instruction lg_gate not found in
    # equivalence library and no rule found to expand."
    # When we do not calibrate qubit, this happens since it doesn't know what lg_gate is


