from classes.AR_process import AutoregressiveProcess
from functions.LS_estimator import fit_AR
from classes.make_experiment import Experiment
import matplotlib.pyplot as plt

# Instantiate AR process parameters
parameters_list = [[0.99], [0.2, 0.5, -0.1], [0.1, 0.1, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99]]
gamma0 = [0,1]

# Create plot
fig, ax = plt.subplots(1,2,figsize=(10, 4))

# Colors
colors = ['C0', 'C1', 'C2', 'C3']

# Divide in two cases, homogeneous and non homogeneous
ax[0].set_title('Homogeneous case')
ax[1].set_title('Non homogeneous case')

for i in range(2):
    for j in range(len(parameters_list)):
        ar = AutoregressiveProcess(parameters_list[j], gamma0 = gamma0[i])

        # Create class for the experiment and make it
        exp = Experiment(ar, fit_AR)
        exp.make_experiment(seeds = 20, n = 200)

        # plot on the right axis value
        exp.plot_confidence_region(ax[i], col = colors[j], label='experiment {}'.format(j))
        ax[i].legend()

save_name = 'full_experiment'
fig.savefig('figures/'+save_name+'.pdf')