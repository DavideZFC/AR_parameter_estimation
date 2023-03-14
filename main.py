from classes.AR_process import AutoregressiveProcess
from functions.LS_estimator import fit_AR
from classes.make_experiment import Experiment
import matplotlib.pyplot as plt

# Instantiate AR process
ar = AutoregressiveProcess([0.2, 0.5, -0.1], gamma0 = 1)

# Create class for the experiment and make it
exp = Experiment(ar, fit_AR)
exp.make_experiment(seeds = 10, n = 100)

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))
exp.plot_confidence_region(ax, col = 'C0', label='confidence regions')

save_name = 'learning_curve_2'
fig.savefig('figures/'+save_name+'.pdf')