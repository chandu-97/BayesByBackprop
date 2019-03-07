# This script is to run all variations of BBB(Bayes by backprop)
# 
# Running:-
# python test_variations.py >> script.sh
# chmod +x script.sh
# ./script.sh

import os
import numpy as np

beta_s = ["Blundell", "Soenderby", "Normal"]
num_epoch_s = [15]
fc_s = ["[128,64]", "[256,128]", ]
sigma_prior_s = [float(np.exp(-3)), float(np.exp(-2)), float(np.exp(-1))]
init_s = ["--is-gaussian", "--is-orthogonal"]
init_scale_s = [1,10,100]

# cuda = True in this

template = "python bbb/test.py --beta={} --num-epoch={} \
--fc={} --cuda --sigma-prior={} {} --init-scale={} --is-wandb"

for sigma_prior in sigma_prior_s:
	for num_epoch in num_epoch_s:
		for fc in fc_s:
			for beta in beta_s:
				for init in init_s:
					for init_scale in init_scale_s:
						print(template.format(str(beta), str(num_epoch), str(fc),
								str(sigma_prior), str(init), str(init_scale)) )