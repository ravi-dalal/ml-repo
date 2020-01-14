import sys
import numpy as np
from processEmail import processEmail

#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# ==================== Part 1: Email Preprocessing ====================

# Extract Features
with open('data/emailSample1.txt', 'r') as file:
	file_contents = file.read()
file.close()
#print(file_contents)
word_indices  = processEmail(file_contents)
#print(word_indices)
