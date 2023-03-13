from classes.AR_process import AutoregressiveProcess
from functions.LS_estimator import fit_AR
from classes.make_experiment import Experiment
import matplotlib.pyplot as plt


ar = AutoregressiveProcess([0.2, 0.5, -0.1], gamma0 = 1)

exp = Experiment(ar, fit_AR)
exp.make_experiment(seeds = 10, n = 100)

fig, ax = plt.subplots(figsize=(12, 8))
exp.plot_confidence_region(ax, col = 'C0', label='confidence regions')
fig.savefig('figures/data1.pdf')