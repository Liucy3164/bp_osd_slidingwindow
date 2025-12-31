import numpy as np
from tqdm import tqdm
import json
import time
import datetime
from ldpc import BpOsdDecoder
from bposd.css import css_code
import scipy


class css_decode_sim:
    """
    Repeated-round BP+OSD lifetime simulation for CSS codes (channel_update is assumed None).

    Key conventions (paper-consistent):
      - Store prev_synd_*_window as ABSOLUTE (noisy) syndromes for the current window.
      - Each EC cycle: build window syndromes = kept(old) + new(measured), decode, commit ξ,
        then UPDATE the stored absolute syndromes by adding H ξ ONCE (post-commit).
      - Window decoder uses differential syndromes Δσ inside the window.
      - Measurement error is i.i.d Bernoulli per syndrome bit per round with rate meas_error_rate.
      - H_win = [I_W ⊗ H | B ⊗ I_m], B lower bidiagonal with ones on diag and subdiag.

    Notes:
      - We keep a cumulative committed correction (prev_c_*_prev2) for logical testing / residual.
      - prev_corrections_* stores ONLY the LAST committed ξ (used nowhere else by default).
    """

    def __init__(self, hx=None, hz=None, **input_dict):
        default_input = {
            "error_rate": None,
            "xyz_error_bias": [1, 1, 1],
            "target_runs": 100,
            "seed": 0,
            "bp_method": "minimum_sum",
            "ms_scaling_factor": 0.625,
            "max_iter": 0,
            "osd_method": "osd_cs",
            "osd_order": 2,
            "save_interval": 2,
            "output_file": None,
            "check_code": 1,
            "tqdm_disable": 0,
            "run_sim": 1,
            "channel_update": None,
            "max_rounds": 100000,
            "store_lifetimes": True,
            "failure_decoder": "osdw",
            "Offset_F": 1,
            "Window_W": 1,
            "meas_error_rate": None,  # if None, defaults to error_rate
        }

        for k, v in input_dict.items():
            self.__dict__[k] = v
        for k, v in default_input.items():
            if k not in self.__dict__:
                self.__dict__[k] = v

        if self.error_rate is None:
            raise ValueError("error_rate must be provided (float in [0,1]).")
        if self.meas_error_rate is None:
            self.meas_error_rate = self.error_rate

        self.K = None
        self.N = None
        self.start_date = None
        self.runtime = 0.0
        self.runtime_readable = None
        self.run_count = 0

        self.lifetimes = []
        self.mean_lifetime = 0.0
        self.median_lifetime = 0.0
        self.std_lifetime = 0.0
        self.min_lifetime = 0
        self.max_lifetime_observed = 0
        self.fail_count = 0
        self.censored_count = 0

        if self.seed == 0 or self.run_count != 0:
            self.seed = np.random.randint(low=1, high=2**32 - 1)
        np.random.seed(self.seed)

        self.hx = scipy.sparse.csr_matrix(hx).astype(np.uint8)
        self.hz = scipy.sparse.csr_matrix(hz).astype(np.uint8)
        self.N = self.hz.shape[1]

        self._construct_code()
        self._error_channel_setup()
        self._decoder_setup()

        if self.channel_update is not None:
            raise ValueError("This version assumes channel_update=None.")

        if self.run_sim:
            self.run_lifetime_sim()

    def _construct_code(self):
        if isinstance(self.hx, (np.ndarray, scipy.sparse.spmatrix)) and isinstance(
            self.hz, (np.ndarray, scipy.sparse.spmatrix)
        ):
            qcode = css_code(self.hx, self.hz)
            self.lx = qcode.lx
            self.lz = qcode.lz
            self.K = qcode.K
            self.N = qcode.N
            if self.check_code and not qcode.test():
                raise Exception("Error: invalid CSS code. Check hx/hz!")
        else:
            raise Exception("Invalid object type for hx/hz")

    def _error_channel_setup(self):
        xyz_error_bias = np.array(self.xyz_error_bias, dtype=float)

        if xyz_error_bias[0] == np.inf:
            self.px, self.py, self.pz = self.error_rate, 0.0, 0.0
        elif xyz_error_bias[1] == np.inf:
            self.px, self.py, self.pz = 0.0, self.error_rate, 0.0
        elif xyz_error_bias[2] == np.inf:
            self.px, self.py, self.pz = 0.0, 0.0, self.error_rate
        else:
            s = np.sum(xyz_error_bias)
            if s == 0:
                raise ValueError("xyz_error_bias sums to 0.")
            self.px, self.py, self.pz = self.error_rate * xyz_error_bias / s

        hadamard_rotate = getattr(self, "hadamard_rotate", 0)
        hadamard_rotate_sector1_length = getattr(self, "hadamard_rotate_sector1_length", 0)

        if hadamard_rotate == 0:
            self.channel_probs_x = np.ones(self.N) * self.px
            self.channel_probs_y = np.ones(self.N) * self.py
            self.channel_probs_z = np.ones(self.N) * self.pz
        elif hadamard_rotate == 1:
            n1 = hadamard_rotate_sector1_length
            self.channel_probs_x = np.hstack([np.ones(n1) * self.px, np.ones(self.N - n1) * self.pz])
            self.channel_probs_z = np.hstack([np.ones(n1) * self.pz, np.ones(self.N - n1) * self.px])
            self.channel_probs_y = np.ones(self.N) * self.py
        else:
            raise ValueError(f"hadamard_rotate must be 0 or 1, got {hadamard_rotate}")

        self.channel_probs_x.setflags(write=False)
        self.channel_probs_y.setflags(write=False)
        self.channel_probs_z.setflags(write=False)

    def _decoder_setup(self):
        self.ms_scaling_factor = float(self.ms_scaling_factor)

        m_z = self.hx.shape[0]
        m_x = self.hz.shape[0]
        W = int(self.Window_W)

        p_meas = float(self.meas_error_rate)
        meas_probs_z = np.ones(W * m_z, dtype=float) * p_meas
        meas_probs_x = np.ones(W * m_x, dtype=float) * p_meas

        # B lower bidiagonal: ones on diagonal and subdiagonal
        B = scipy.sparse.diags(
            [np.ones(W, dtype=np.uint8), np.ones(W - 1, dtype=np.uint8)],
            offsets=[0, -1],
            shape=(W, W),
            format="csr",
            dtype=np.uint8,
        )
        I_W = scipy.sparse.identity(W, dtype=np.uint8)

        # Z-window decoder
        I_m_z = scipy.sparse.identity(m_z, dtype=np.uint8)
        left_part_z = scipy.sparse.kron(I_W, self.hx, format="csr")
        right_part_z = scipy.sparse.kron(B, I_m_z, format="csr")
        H_win_z = scipy.sparse.hstack([left_part_z, right_part_z], format="csr").astype(np.uint8)

        data_probs_z = np.tile(self.channel_probs_z + self.channel_probs_y, W)
        self.bpd_z = BpOsdDecoder(
            H_win_z,
            channel_probs=np.hstack([data_probs_z, meas_probs_z]),
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

        # X-window decoder
        I_m_x = scipy.sparse.identity(m_x, dtype=np.uint8)
        left_part_x = scipy.sparse.kron(I_W, self.hz, format="csr")
        right_part_x = scipy.sparse.kron(B, I_m_x, format="csr")
        H_win_x = scipy.sparse.hstack([left_part_x, right_part_x], format="csr").astype(np.uint8)

        data_probs_x = np.tile(self.channel_probs_x + self.channel_probs_y, W)
        self.bpd_x = BpOsdDecoder(
            H_win_x,
            channel_probs=np.hstack([data_probs_x, meas_probs_x]),
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

        # Ideal decoders
        self.bpd_z_ideal = BpOsdDecoder(
            self.hx,
            channel_probs=self.channel_probs_z + self.channel_probs_y,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )
        self.bpd_x_ideal = BpOsdDecoder(
            self.hz,
            channel_probs=self.channel_probs_x + self.channel_probs_y,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

    def _generate_error(self):
        ex = np.zeros(self.N, dtype=np.uint8)
        ez = np.zeros(self.N, dtype=np.uint8)

        px = self.channel_probs_x
        py = self.channel_probs_y
        pz = self.channel_probs_z

        r = np.random.random(self.N)

        z_only = r < pz
        ez[z_only] = 1

        x_only = (r >= pz) & (r < (pz + px))
        ex[x_only] = 1

        y_region = (r >= (pz + px)) & (r < (pz + px + py))
        ex[y_region] = 1
        ez[y_region] = 1

        return ex, ez

    def _one_round_update_residual(self, residual_x, residual_z):
        # STEP 1: generate F fresh errors and measure F new (absolute) syndromes
        if self.prev_E_x_list is None or self.prev_E_z_list is None:
            raise RuntimeError("prev_E_*_list is None. Did you set it in lifetime_trial()?")

        E_x_list = [e.copy() for e in self.prev_E_x_list]
        E_z_list = [e.copy() for e in self.prev_E_z_list]

        error_x0 = residual_x.copy()
        error_z0 = residual_z.copy()

        E_x_cumul = np.zeros(self.N, dtype=np.uint8)
        E_z_cumul = np.zeros(self.N, dtype=np.uint8)

        new_sigma_x_list = []
        new_sigma_z_list = []
        e_x_list = []
        e_z_list = []

        m_x = self.hz.shape[0]
        m_z = self.hx.shape[0]
        n_x = self.hz.shape[1]
        n_z = self.hx.shape[1]

        p_meas = float(self.meas_error_rate)

        for _ in range(self.Offset_F):
            e_x_new, e_z_new = self._generate_error()

            E_x_list.append(e_x_new.copy())
            E_z_list.append(e_z_new.copy())

            e_x_list.append(e_x_new.copy())
            e_z_list.append(e_z_new.copy())

            E_x_cumul ^= e_x_new
            E_z_cumul ^= e_z_new

            eta_x = np.random.binomial(1, p_meas, size=m_x).astype(np.uint8)
            sigma_x = (self.hz @ (error_x0 ^ E_x_cumul)) % 2
            sigma_x = (np.asarray(sigma_x, dtype=np.uint8).ravel() ^ eta_x)
            new_sigma_x_list.append(sigma_x)

            eta_z = np.random.binomial(1, p_meas, size=m_z).astype(np.uint8)
            sigma_z = (self.hx @ (error_z0 ^ E_z_cumul)) % 2
            sigma_z = (np.asarray(sigma_z, dtype=np.uint8).ravel() ^ eta_z)
            new_sigma_z_list.append(sigma_z)

        # STEP 2: form the new window of ABSOLUTE syndromes (NO pre-update here)
        if self.prev_synd_x_window is not None:
            kept_sigma_x = self.prev_synd_x_window[self.Offset_F:]
            kept_sigma_z = self.prev_synd_z_window[self.Offset_F:]
        else:
            kept_sigma_x = []
            kept_sigma_z = []

        sigma_x_windowed_pre = kept_sigma_x + new_sigma_x_list
        sigma_z_windowed_pre = kept_sigma_z + new_sigma_z_list

        # STEP 3: differential syndromes within the window
        Δsigma_x_list = [sigma_x_windowed_pre[0]]
        Δsigma_z_list = [sigma_z_windowed_pre[0]]
        for w in range(1, len(sigma_x_windowed_pre)):
            Δsigma_x_list.append(sigma_x_windowed_pre[w] ^ sigma_x_windowed_pre[w - 1])
            Δsigma_z_list.append(sigma_z_windowed_pre[w] ^ sigma_z_windowed_pre[w - 1])

        Δsigma_x_stacked = np.concatenate(Δsigma_x_list, axis=0)
        Δsigma_z_stacked = np.concatenate(Δsigma_z_list, axis=0)

        # STEP 4: decode with window decoder
        self.bpd_x.decode(Δsigma_x_stacked)
        self.bpd_z.decode(Δsigma_z_stacked)

        c_x_full = self.bpd_x.osdw_decoding
        c_z_full = self.bpd_z.osdw_decoding

        # STEP 5: sum F fresh errors
        E_fresh_x_1_to_F = np.zeros(self.N, dtype=np.uint8)
        E_fresh_z_1_to_F = np.zeros(self.N, dtype=np.uint8)
        for i in range(self.Offset_F):
            E_fresh_x_1_to_F ^= e_x_list[i]
            E_fresh_z_1_to_F ^= e_z_list[i]

        # STEP 6: sum first F physical correction blocks (data part only)
        c_new_x_1_to_F = np.zeros(n_x, dtype=np.uint8)
        c_new_z_1_to_F = np.zeros(n_z, dtype=np.uint8)
        for i in range(self.Offset_F):
            c_new_x_1_to_F ^= c_x_full[i * n_x : (i + 1) * n_x]
            c_new_z_1_to_F ^= c_z_full[i * n_z : (i + 1) * n_z]

        # cumulative committed correction (for logical test / residual tracking)
        if self.prev_c_x_prev2 is None or self.prev_c_z_prev2 is None:
            raise RuntimeError("prev_c_*_prev2 is None. It must be initialized in lifetime_trial().")

        c_x_prev2 = self.prev_c_x_prev2 ^ c_new_x_1_to_F
        c_z_prev2 = self.prev_c_z_prev2 ^ c_new_z_1_to_F

        # STEP 7: POST-COMMIT syndrome-frame update ONCE by H*ξ (paper-consistent)
        H_xi_x = np.asarray((self.hz @ c_new_x_1_to_F) % 2, dtype=np.uint8).ravel()
        H_xi_z = np.asarray((self.hx @ c_new_z_1_to_F) % 2, dtype=np.uint8).ravel()

        sigma_x_windowed_post = [(s ^ H_xi_x) for s in sigma_x_windowed_pre]
        sigma_z_windowed_post = [(s ^ H_xi_z) for s in sigma_z_windowed_pre]

        # store for next round
        self.prev_synd_x_window = sigma_x_windowed_post
        self.prev_synd_z_window = sigma_z_windowed_post
        self.prev_E_x_list = [e.copy() for e in E_x_list]
        self.prev_E_z_list = [e.copy() for e in E_z_list]
        self.prev_c_x_prev2 = c_x_prev2.copy()
        self.prev_c_z_prev2 = c_z_prev2.copy()

        # store last committed only (optional)
        self.prev_corrections_x = c_new_x_1_to_F.copy()
        self.prev_corrections_z = c_new_z_1_to_F.copy()

        # STEP 8: test residual (your original structure)
        W = int(self.Window_W)
        F = int(self.Offset_F)
        cut = max(0, len(E_x_list) - (W - F))

        E_x_sum = np.zeros(self.N, dtype=np.uint8)
        E_z_sum = np.zeros(self.N, dtype=np.uint8)
        for e in E_x_list[:cut]:
            E_x_sum ^= e
        for e in E_z_list[:cut]:
            E_z_sum ^= e

        R_x_test = (E_x_sum ^ c_x_prev2)
        R_z_test = (E_z_sum ^ c_z_prev2)

        # STEP 9: ideal decoder logical test
        sigma_z_ideal = np.asarray((self.hx @ R_z_test) % 2, dtype=np.uint8).ravel()
        self.bpd_z_ideal.decode(sigma_z_ideal)
        c_z_ideal = self.bpd_z_ideal.osdw_decoding
        R_z_ideal = R_z_test ^ c_z_ideal

        sigma_x_ideal = np.asarray((self.hz @ R_x_test) % 2, dtype=np.uint8).ravel()
        self.bpd_x_ideal.decode(sigma_x_ideal)
        c_x_ideal = self.bpd_x_ideal.osdw_decoding
        R_x_ideal = R_x_test ^ c_x_ideal

        fail_x = (self.lz @ (R_x_ideal % 2) % 2).any()
        fail_z = (self.lx @ (R_z_ideal % 2) % 2).any()
        failed = bool(fail_x or fail_z)

        # Returned residual: keep your “sum-all + cumulative committed” convention
        E_x_sumc = np.zeros(self.N, dtype=np.uint8)
        E_z_sumc = np.zeros(self.N, dtype=np.uint8)
        for e in E_x_list:
            E_x_sumc ^= e
        for e in E_z_list:
            E_z_sumc ^= e

        residual_x = (E_x_sumc ^ c_x_prev2)
        residual_z = (E_z_sumc ^ c_z_prev2)

        return residual_x, residual_z, failed

    def lifetime_trial(self):
        # Reset per-trial persistent state
        self.prev_synd_x_window = None
        self.prev_synd_z_window = None
        self.prev_corrections_x = None
        self.prev_corrections_z = None
        self.prev_error_x = None
        self.prev_error_z = None
        self.prev_E_x_list = None
        self.prev_E_z_list = None
        self.prev_R_x_F = None
        self.prev_R_z_F = None
        self.prev_c_x_prev2 = None
        self.prev_c_z_prev2 = None

        p_meas = float(self.meas_error_rate)

        # STEP 1: generate initial W syndromes
        E_x = np.zeros(self.N, dtype=np.uint8)
        E_z = np.zeros(self.N, dtype=np.uint8)

        E_x_list = []
        E_z_list = []
        sigma_x_list = []
        sigma_z_list = []

        for _ in range(int(self.Window_W)):
            e_x_new, e_z_new = self._generate_error()
            E_x ^= e_x_new
            E_z ^= e_z_new

            E_x_list.append(e_x_new.copy())
            E_z_list.append(e_z_new.copy())

            eta_x = np.random.binomial(1, p_meas, size=self.hz.shape[0]).astype(np.uint8)
            sigma_x = np.asarray((self.hz @ E_x) % 2, dtype=np.uint8).ravel() ^ eta_x
            sigma_x_list.append(sigma_x)

            eta_z = np.random.binomial(1, p_meas, size=self.hx.shape[0]).astype(np.uint8)
            sigma_z = np.asarray((self.hx @ E_z) % 2, dtype=np.uint8).ravel() ^ eta_z
            sigma_z_list.append(sigma_z)

        # STEP 2: differential syndromes
        Δsigma_x_list = [sigma_x_list[0]]
        Δsigma_z_list = [sigma_z_list[0]]
        for w in range(1, int(self.Window_W)):
            Δsigma_x_list.append(sigma_x_list[w] ^ sigma_x_list[w - 1])
            Δsigma_z_list.append(sigma_z_list[w] ^ sigma_z_list[w - 1])

        Δsigma_x_stacked = np.concatenate(Δsigma_x_list, axis=0)
        Δsigma_z_stacked = np.concatenate(Δsigma_z_list, axis=0)

        # STEP 3: decode initial window
        self.bpd_x.decode(Δsigma_x_stacked)
        self.bpd_z.decode(Δsigma_z_stacked)

        c_x_full = self.bpd_x.osdw_decoding
        c_z_full = self.bpd_z.osdw_decoding

        n_x = self.hz.shape[1]
        n_z = self.hx.shape[1]

        # STEP 4: commit first F corrections
        F = int(self.Offset_F)

        E_x_1_to_F = np.zeros(self.N, dtype=np.uint8)
        E_z_1_to_F = np.zeros(self.N, dtype=np.uint8)
        for i in range(F):
            E_x_1_to_F ^= E_x_list[i]
            E_z_1_to_F ^= E_z_list[i]

        c_x_1_to_F = np.zeros(n_x, dtype=np.uint8)
        c_z_1_to_F = np.zeros(n_z, dtype=np.uint8)
        for i in range(F):
            c_x_1_to_F ^= c_x_full[i * n_x : (i + 1) * n_x]
            c_z_1_to_F ^= c_z_full[i * n_z : (i + 1) * n_z]

        # STEP 5: logical test at position F
        R_x_F = E_x_1_to_F ^ c_x_1_to_F
        R_z_F = E_z_1_to_F ^ c_z_1_to_F

        sigma_z_ideal = np.asarray((self.hx @ R_z_F) % 2, dtype=np.uint8).ravel()
        self.bpd_z_ideal.decode(sigma_z_ideal)
        c_z_ideal = self.bpd_z_ideal.osdw_decoding
        R_z_ideal = R_z_F ^ c_z_ideal

        sigma_x_ideal = np.asarray((self.hz @ R_x_F) % 2, dtype=np.uint8).ravel()
        self.bpd_x_ideal.decode(sigma_x_ideal)
        c_x_ideal = self.bpd_x_ideal.osdw_decoding
        R_x_ideal = R_x_F ^ c_x_ideal

        fail_x = (self.lz @ (R_x_ideal % 2) % 2).any()
        fail_z = (self.lx @ (R_z_ideal % 2) % 2).any()
        failed = bool(fail_x or fail_z)
        if failed:
            return 0, False

        # STEP 6: prepare residual at window edge
        residual_x = E_x ^ c_x_1_to_F
        residual_z = E_z ^ c_z_1_to_F

        # POST-COMMIT syndrome-frame update for initial stored window (once)
        H_xi_x = np.asarray((self.hz @ c_x_1_to_F) % 2, dtype=np.uint8).ravel()
        H_xi_z = np.asarray((self.hx @ c_z_1_to_F) % 2, dtype=np.uint8).ravel()
        sigma_x_list_post = [(s ^ H_xi_x) for s in sigma_x_list]
        sigma_z_list_post = [(s ^ H_xi_z) for s in sigma_z_list]

        self.prev_synd_x_window = [s.copy() for s in sigma_x_list_post]
        self.prev_synd_z_window = [s.copy() for s in sigma_z_list_post]

        self.prev_corrections_x = c_x_1_to_F.copy()  # last committed
        self.prev_corrections_z = c_z_1_to_F.copy()  # last committed

        self.prev_E_x_list = [e.copy() for e in E_x_list]
        self.prev_E_z_list = [e.copy() for e in E_z_list]

        self.prev_R_x_F = R_x_F
        self.prev_R_z_F = R_z_F

        # cumulative committed (for test/residual)
        self.prev_c_x_prev2 = c_x_1_to_F.copy()
        self.prev_c_z_prev2 = c_z_1_to_F.copy()

        passed = 1
        for _round_num in range(2, int(self.max_rounds) + 1):
            residual_x, residual_z, failed = self._one_round_update_residual(residual_x, residual_z)
            if failed:
                return passed, False
            passed += 1

        return passed, True

    def run_lifetime_sim(self):
        self.start_date = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%A, %B %d, %Y %H:%M:%S"
        )

        pbar = tqdm(
            range(self.run_count + 1, self.target_runs + 1),
            disable=self.tqdm_disable,
            ncols=0,
        )

        start_time = time.time()
        save_time = start_time

        sum_lt = 0.0
        sumsq_lt = 0.0
        min_lt = None
        max_lt = 0

        for self.run_count in pbar:
            lt, censored = self.lifetime_trial()

            if self.store_lifetimes:
                self.lifetimes.append(int(lt))

            if censored:
                self.censored_count += 1
            else:
                self.fail_count += 1

            sum_lt += lt
            sumsq_lt += lt * lt
            if min_lt is None or lt < min_lt:
                min_lt = lt
            if lt > max_lt:
                max_lt = lt

            n = self.run_count
            mean = sum_lt / n
            var = max(0.0, (sumsq_lt / n) - mean * mean)
            std = np.sqrt(var)

            self.mean_lifetime = float(mean)
            self.std_lifetime = float(std)
            self.min_lifetime = int(min_lt if min_lt is not None else 0)
            self.max_lifetime_observed = int(max_lt)

            if self.store_lifetimes and len(self.lifetimes) > 0:
                self.median_lifetime = float(np.median(np.array(self.lifetimes, dtype=float)))
            else:
                self.median_lifetime = 0.0

            current_time = time.time()
            save_loop = current_time - save_time

            if int(save_loop) > self.save_interval or self.run_count == self.target_runs:
                save_time = time.time()
                self.runtime = (current_time - start_time)
                self.runtime_readable = time.strftime("%H:%M:%S", time.gmtime(self.runtime))

                if self.output_file is not None:
                    with open(self.output_file, "w+") as f:
                        f.write(json.dumps(self.output_dict(), sort_keys=True, indent=4))

        print(self.mean_lifetime)
        return json.dumps(self.output_dict(), sort_keys=True, indent=4)

    def output_dict(self):
        out = {
            "start_date": self.start_date,
            "runtime": self.runtime,
            "runtime_readable": self.runtime_readable,
            "seed": int(self.seed),
            "N": int(self.N),
            "K": int(self.K),
            "error_rate": float(self.error_rate),
            "meas_error_rate": float(self.meas_error_rate),
            "xyz_error_bias": list(self.xyz_error_bias),
            "bp_method": self.bp_method,
            "ms_scaling_factor": float(self.ms_scaling_factor),
            "max_iter": int(self.max_iter),
            "osd_method": self.osd_method,
            "osd_order": int(self.osd_order),
            "failure_decoder": self.failure_decoder,
            "target_runs": int(self.target_runs),
            "run_count": int(self.run_count),
            "max_rounds": int(self.max_rounds),
            "fail_count": int(self.fail_count),
            "censored_count": int(self.censored_count),
            "mean_lifetime": float(self.mean_lifetime),
            "median_lifetime": float(self.median_lifetime),
            "std_lifetime": float(self.std_lifetime),
            "min_lifetime": int(self.min_lifetime),
            "max_lifetime_observed": int(self.max_lifetime_observed),
        }
        if self.store_lifetimes:
            out["lifetimes"] = list(map(int, self.lifetimes))
        return out
