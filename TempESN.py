import numpy as np
import NymphESN.nymphesn as nymph
import NymphESN.restrictedmatrix as rm
import functools
from enum import Enum

debug = False

class SleepState(Enum):
    SLEEP = 0
    WAKE = 1

rng = np.random.default_rng()

class Temporal_ESN(nymph.NymphESN):
    def __init__(self, 
                K, 
                N, 
                L, 
                n_subreservoirs: int, 
                encodings: str, 
                measure_length=0, 
                seed=1, 
                f=np.tanh, 
                rho=2, 
                density=0.1, 
                svd_dv=1) -> None:
        super().__init__(K, N, L, seed, f, rho, density, svd_dv)
        self.n_subreservoirs = n_subreservoirs
        self.size_subreservoirs = self.N//self.n_subreservoirs #only deals with same-sized subreservoirs
        self.measure_length = measure_length
        self.set_rhythms()
        self.set_f(f)
        self.encodings = encodings #currently is a list of 5-bit strings, there's probably a better pythonic way to do this 
            # Wn ---- most significant bit
            # Wun
            # Bn_
            # B_n 
            # f ---- least significant bit

        #list[tuple(lambda function, string)]
        return
    
    def set_input_weights(self, Wu=None):
        super().set_input_weights(Wu)
        if debug:
            print(f"{self.Wu.shape=}")
        self.WU_BASE = np.array(self.Wu)
        return

    def set_weights(self, W=None, density=0.1):
        super().set_weights(W, density)
        self.W_BASE = np.array(self.W)
        return

    def set_f(self, f=np.tanh):
        self.f = np.asarray([f]*self.N)
        self.F_BASE = np.array(self.f)
        return

    def set_rhythms(self, rhythms=None):
        if rhythms is not None:
            self.rhythms = rhythms
        else:
            self.rhythms = rng.integers(0, 2**self.measure_length, (self.n_subreservoirs, 1), dtype=np.uint8)
            self.rhythms = np.unpackbits(self.rhythms, axis=1)[:, -self.measure_length:] # unpack bits and remove any leading 0s
        if debug:
            print(f"{self.rhythms=}")
        return

    def generate_encodings(self, t: int) -> np.array:
        wake_states = [rhythm[t%len(rhythm)] for rhythm in self.rhythms]
        # 0 if subreservoir is asleep, 1 otherwise
        return list(map(lambda a, b: "11111" if a==1 else b, wake_states, self.encodings))

    def generate_single_subinput(self, encoding, index):
        binary = np.ones((self.N, 1))
        Wun = np.ones((self.K, self.size_subreservoirs)) * int(encoding[1])
        indices_Wn = np.asarray(range(self.size_subreservoirs)) +  (self.size_subreservoirs*index)
        indices_K = np.asarray(range(self.K))
        coords_Wun = np.meshgrid(indices_Wn, indices_K)
        if debug:
            print(f"{binary.shape=}")
            print(f"{Wun.shape=}")
            print(f"{coords_Wun=}")
        binary[tuple(coords_Wun)] = Wun
        return binary

    def generate_Wu(self, encodings: list) -> None:
        binary_inputs_all = map(self.generate_single_subinput, encodings, range(len(encodings)))
        binary_inputs = functools.reduce(np.multiply, binary_inputs_all)
        if debug:
            print(f"{binary_inputs.shape=}")
            print(f"{self.WU_BASE.shape=}")
            print(f"{self.Wu.shape=}")
        self.Wu = np.multiply(self.WU_BASE, binary_inputs)
        return
    
    def generate_single_subreservoir(self, encoding: str, index: int) -> np.array:
        """
        takes the binary encoding of a single reservoir at time t
        and returns the binary edge matrix that corresponds to it
        """
        binary = np.ones((self.N, self.N))
        B_n = np.ones((self.N, self.size_subreservoirs)) * int(encoding[2])
        Bn_ = np.ones((self.size_subreservoirs, self.N)) * int(encoding[3])
        Wn = np.ones((self.size_subreservoirs, self.size_subreservoirs)) * int(encoding[0])

        indices_N = np.asarray(range(self.N))
        indices_Wn = np.asarray(range(self.size_subreservoirs)) +  (self.size_subreservoirs*index)
        coords_B_n = np.meshgrid(indices_Wn, indices_N)
        coords_Bn_ = np.meshgrid(indices_N, indices_Wn)
        coords_Wn = np.meshgrid(indices_Wn, indices_Wn)
        
        binary[tuple(coords_B_n)] = B_n
        binary[tuple(coords_Bn_)] = Bn_
        binary[tuple(coords_Wn)] = Wn # must take place _after_ the first two so it does not get overwritten!!

        return binary
    
    def generate_W(self, encodings: list) -> None:
        binary_edges_all = map(self.generate_single_subreservoir, encodings, range(len(encodings)))
        #fold matrix multiplication over the list of binary edges
        binary_edges = functools.reduce(np.multiply, binary_edges_all)
        self.W = np.multiply(self.W_BASE, binary_edges)
        s = np.linalg.svd(self.W, compute_uv=False)
        self.W = self.W / (s[0]/self.svd_dv)
        return

    def generate_fs(self, encodings: list) -> None:
        indices = [i for i in range(len(encodings)) if encodings[i][4] == "0"]
        self.f = np.asarray(self.F_BASE)
        self.f[indices] = lambda x: x
        return

    def run_timestep(self, t: int):
        encodings = self.generate_encodings(t)
        self.generate_fs(encodings)
        self.generate_W(encodings)
        self.generate_Wu(encodings)
        # print(f"{self.W=}")
        u_t = self.uall[:, t]
        x_t = self.xall[:,t]
        x_t.shape = (self.N, 1)
        if debug:
            print(f"{self.Wu.shape=}")
            print(f"{u_t.shape=}")
        Wu_x_u = self.Wu.dot(u_t)
        Wu_x_u.shape = (self.N, 1)
        x_t1 = list(map(lambda f,x: f(x), self.f, self.rho * x_t.T.dot(self.W) + Wu_x_u.T))
        x_t1 = np.array(x_t1)
        self.xall = np.hstack((self.xall, x_t1.T))
        return

class TempESN_Encoding:
    def __init__(self):
        return

    @staticmethod
    def generate_encoding(Wn: SleepState, Wun: SleepState, Bout: SleepState, Bin: SleepState, f: SleepState):
        '''Generate the encoding for the sleep state of a single reservoir.
        if an argument is set to SLEEP, then that will be set to the default sleep value for the data structure during the subreservoir's sleep states.
        The default sleep value are the following:
        Wn: I
        Wun: 0
        Bout: 0
        Bin: 0
        f: identity function f(x)=x
        '''
        return f"{Wn.value}{Wun.value}{Bout.value}{Bin.value}{f.value}"

    @staticmethod
    def generate_circuit_encoding():
        return "01110"

    @staticmethod
    def generate_swarm_encoding():
        return "01000"

    @staticmethod
    def generate_gondor_encoding():
        return "00100"

# temp = Temporal_ESN(1, 12, 1, 3)
# temp.set_weights(np.ones((12, 12)))
# print(temp.W)
# print(temp.generate_single_subreservoir("1111", 1))
# # should be all 1s
# print(temp.generate_single_subreservoir("0111", 1))
# # should be ((1, 1), (1, 0))
# print(temp.generate_single_subreservoir("0100", 1))
# #should be ((1, 0), (0, 0))
# print(temp.generate_single_subreservoir("0101", 1))
# #should be ((1, 1), (0, 0))
# temp.generate_W(["1111", "0100", "0111"])
# print(temp.W)