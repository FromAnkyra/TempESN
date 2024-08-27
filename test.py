import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rmatrix
import NymphESN.vis as vis
import TempESN
import numpy as np
import pandas as pd
from TempESN import SleepState as ss


def get_u(T, seed=None):
    # random stream of inputs u in range 0,0.5 in col 0, 1s (bias) in col 1
    if seed:
        np.random.seed(seed)
    return np.random.uniform(0.0, 0.5, T)


def narmafun(y, u, alpha, beta, gamma, delta):
    # y =[y(t-N+1, ..., y(t-1), y(t))], similar for u
    return alpha * y[-1] + beta * y[-1] * sum(y) + gamma * u[0] * u[-1] + delta


def run_narma(NARMA, T, u, debug=False):
    narmaparams = {
        5: (0.3, 0.05, 1.5, 0.2),  
        10: (0.3, 0.05, 1.5, 0.1),
        20: (0.25, 0.05, 1.5, 0.01),
        30: (0.2, 0.04, 1.5, 0.001)
    }

    # initial NARMA values of y
    for t in range(0, NARMA):
        y = [0] * NARMA

    for t in range(NARMA - 1, T - 1):
        
        y_Nt = [y[i] for i in range(t-NARMA+1, t)]
        u_Nt = [u[i] for i in range(t-NARMA+1, t)]
        y_t1 = narmafun(y_Nt, u_Nt, *narmaparams[NARMA])  # y(t+1) = f(y(t), u(t), ...)
        y.append(y_t1)
    if(debug):
        print('u =', u)
        print('y =', y)
        print(f"{len(y) - T=}")
    return y

NARMA=10
TWashout = 100
TTrain = 2000
TTest = 1000
TTot = TWashout+TTrain+TTest
i = 0

input = get_u(T=TTot, seed=i)
vtarget = run_narma(NARMA, TTot, input, debug=False)
vtarget_np = np.array(vtarget)
enc1 = TempESN.TempESN_Encoding.generate_encoding(Wn=ss.SLEEP, Wun=ss.WAKE, Bout=ss.SLEEP, Bin=ss.WAKE, f=ss.SLEEP)
# should be 01010
enc2 = TempESN.TempESN_Encoding.generate_encoding(Wn=ss.WAKE, Wun=ss.WAKE, Bout=ss.WAKE, Bin=ss.WAKE, f=ss.WAKE)

#should be 11111

encodings = [enc1, enc2]
print(f"{encodings=}")

temp = TempESN.Temporal_ESN(K=1,
                            N=20,
                            L=1,
                            n_subreservoirs=2,
                            encodings=encodings,
                            measure_length=3)

print(f"{temp.rhythms=}")
# should be a 2 by 2 binary matrix

W = rmatrix.create_restricted_esn_weights(20, 10, 2, 0.8, 0.2)
temp.set_weights(W=W)

# print(f"{temp.W=}")
#should be a 20-by-20 restricted weight matrix

temp.set_data_lengths(TWashout, TTrain, TTest)
temp.set_input_stream(input)
temp.run_full()
temp.train_reservoir(vtarget_np[TWashout:-TTest])
# print(f"{temp.Wv=}")
temp.get_output()
# print(temp.vall)
# print(expected_output)
print(temp.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse))