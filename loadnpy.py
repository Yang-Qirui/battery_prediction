import numpy as np
M = np.load('./dataset/test/soh.npy')
np.savetxt('soh-test.csv',M,delimiter=",")
