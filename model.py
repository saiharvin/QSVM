import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance,algorithm_globals
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.model_selection import train_test_split


seed = 10599
algorithm_globals.random_seed = seed


data = '../input/d/saiharvin/whateverv11/ionosphere 4.csv'
df = pd.read_csv(data)

col_names = df.columns

X = df.drop(['tar'], axis=1)
y = df['tar']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


dimension = 2
bc_feature_map = ZZFeatureMap(feature_dimension=dimension, reps=2, entanglement="linear")
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
bc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=bc_feature_map)


bc_svc=QSVC(quantum_kernel=bc_kernel)
bc_svc.fit(X_train.to_numpy(), y_train.to_numpy())
bc_score = bc_svc.score(X_test.to_numpy(), y_test.to_numpy())
print(f"Callable kernel classification test score: {bc_score}")

pickle.dump(bc_svc, open('model.pkl','wb'))