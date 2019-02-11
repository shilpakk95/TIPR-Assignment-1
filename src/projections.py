import math
import numpy as np

def random_projection(data_matrix,c,K,ip_file_path,out):
    f1 = 1 / (math.sqrt(K))
    rand_matrix = np.random.normal(0, 1, (c, K))
    r_rand = len(rand_matrix)
    c_rand = len(rand_matrix[0])
    # print(noOfRows, noOfColumns)
    mul = np.matmul(data_matrix, rand_matrix)
    norm = np.multiply(f1, mul)
    op_file =out+'_' + str(K) + '.csv'
    np.savetxt(ip_file_path + op_file, norm, delimiter=" ")
    return norm