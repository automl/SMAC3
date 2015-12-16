'''
Created on Nov 19, 2015

@author: Aaron Klein
'''

import logging
import numpy as np

from robo.task.branin import Branin
from robo.initial_design.init_random_uniform import init_random_uniform

from smac.smbo.smbo import SMBO


logging.basicConfig(level=logging.DEBUG)

task = Branin()
X = init_random_uniform(task.X_lower, task.X_upper, 20)
Y = task.evaluate(X)

instance_features = np.array([[1]])
smac = SMBO("/home/kleinaa/devel/git/smac3/test_files/params_branin.pcs",
            instance_features=instance_features)
new_x = smac.choose_next(X, Y)

print new_x
