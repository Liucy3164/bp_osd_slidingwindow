import numpy as np
from tqdm import tqdm
import json
import time
import datetime
from ldpc import BpOsdDecoder
from bposd.css import css_code
import scipy


class css_decode_sim:

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
            "meas_error_rate": None,
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

        # Outputs
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

        # RNG
        if self.seed == 0:
            self.seed = np.random.randint(low=1, high=2**32 - 1)
        np.random.seed(self.seed)

        # Matrices
        self.hx = scipy.sparse.csr_matrix(hx).astype(np.uint8)
        self.hz = scipy.sparse.csr_matrix(hz).astype(np.uint8)
        self.N = int(self.hz.shape[1])

        # Build code + decoders
        self._construct_code()
        self._error_channel_setup()
        self._decoder_setup()

        if self.channel_update is not None:
            raise ValueError("This version assumes channel_update=None.")

        if self.run_sim:
            self.run_lifetime_sim()

    # ------------------------------------------------------------
    # Helper: GF(2) sum of per-round errors with optional tail drop
    # ------------------------------------------------------------
    def _sum_error_list_gf2(self, E_list, drop_last=0):
        """
        Returns sum(E_list[:-drop_last]) mod 2 if drop_last>0, else sum(E_list) mod 2.
        """
        out = np.zeros(self.N, dtype=np.uint8)
        if E_list is None:
            return out

        if drop_last is None:
            drop_last = 0
        drop_last = int(drop_last)
        if drop_last < 0:
            drop_last = 0

        if drop_last == 0:
            iterable = E_list
        else:
            if drop_last >= len(E_list):
                iterable = []
            else:
                iterable = E_list[:-drop_last]

        for e in iterable:
            out = (out + e) % 2
        return out

    # ------------------------------------------------------------
    # Code construction
    # ------------------------------------------------------------
    def _construct_code(self):
        qcode = css_code(self.hx, self.hz)
        self.lx = qcode.lx
        self.lz = qcode.lz
        self.K = int(qcode.K)
        self.N = int(qcode.N)
        if self.check_code and not qcode.test():
            raise RuntimeError("Invalid CSS code: check hx/hz!")

    # ------------------------------------------------------------
    # Noise model
    # ------------------------------------------------------------
    def _error_channel_setup(self):
        xyz_error_bias = np.array(self.xyz_error_bias, dtype=float)
        if np.isinf(xyz_error_bias[0]):
            self.px, self.py, self.pz = self.error_rate, 0.0, 0.0
        elif np.isinf(xyz_error_bias[1]):
            self.px, self.py, self.pz = 0.0, self.error_rate, 0.0
        elif np.isinf(xyz_error_bias[2]):
            self.px, self.py, self.pz = 0.0, 0.0, self.error_rate
        else:
            s = float(np.sum(xyz_error_bias))
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
            n1 = int(hadamard_rotate_sector1_length)
            self.channel_probs_x = np.hstack([np.ones(n1) * self.px, np.ones(self.N - n1) * self.pz])
            self.channel_probs_z = np.hstack([np.ones(n1) * self.pz, np.ones(self.N - n1) * self.px])
            self.channel_probs_y = np.ones(self.N) * self.py
        else:
            raise ValueError(f"hadamard_rotate must be 0 or 1, got {hadamard_rotate}")

        self.channel_probs_x.setflags(write=False)
        self.channel_probs_y.setflags(write=False)
        self.channel_probs_z.setflags(write=False)

    def _generate_error(self):
        """
        Returns fresh (ex, ez) for ONE round (Pauli i.i.d per qubit).
        Y corresponds to ex=1, ez=1.
        """
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

    # ------------------------------------------------------------
    # Decoder setup
    # ------------------------------------------------------------
    def _decoder_setup(self):
        self.ms_scaling_factor = float(self.ms_scaling_factor)

        m_z = int(self.hx.shape[0])
        m_x = int(self.hz.shape[0])
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

        # -------- Z-window decoder --------
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

        # -------- X-window decoder --------
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

        # -------- Ideal decoders --------
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

    # ------------------------------------------------------------
    # One-round update
    # ------------------------------------------------------------
    def _one_round_update_residual(self, residual_x, residual_z):
        """
        One EC cycle update:
          - Form new window (kept abs syndromes + newly measured abs syndromes)
          - Decode window on Δσ
          - Commit ξ (first F blocks)
          - Post-commit update stored abs syndromes by adding H ξ (once)
          - Update cumulative committed correction (prev_c_*_accu2)
          - Perform ideal-decoder logical test (R_test changed per request + early pass)
          - Return updated residual (UNCHANGED convention)
        """
        if self.prev_E_x_list is None or self.prev_E_z_list is None:
            raise RuntimeError("prev_E_*_list is None. Did you set it in lifetime_trial()?")

        E_x_list = [e.copy() for e in self.prev_E_x_list]
        E_z_list = [e.copy() for e in self.prev_E_z_list]

        # --- generate F new errors and measure F new ABSOLUTE syndromes ---
        error_x0 = residual_x.copy()
        error_z0 = residual_z.copy()

        new_sigma_x_list = []
        new_sigma_z_list = []

        m_x = int(self.hz.shape[0])
        m_z = int(self.hx.shape[0])
        p_meas = float(self.meas_error_rate)

        for _ in range(int(self.Offset_F)):
            e_x_new, e_z_new = self._generate_error()
            E_x_list.append(e_x_new.copy())
            E_z_list.append(e_z_new.copy())

            # update the "current physical error at this measurement time"
            error_x0 = (error_x0 + e_x_new) % 2
            error_z0 = (error_z0 + e_z_new) % 2

            eta_x = np.random.binomial(1, p_meas, size=m_x).astype(np.uint8)
            eta_z = np.random.binomial(1, p_meas, size=m_z).astype(np.uint8)

            sigma_x = (np.asarray((self.hz @ error_x0) % 2, dtype=np.uint8).ravel() + eta_x) % 2
            sigma_z = (np.asarray((self.hx @ error_z0) % 2, dtype=np.uint8).ravel() + eta_z) % 2

            new_sigma_x_list.append(sigma_x)
            new_sigma_z_list.append(sigma_z)

        # --- form new ABSOLUTE syndrome window (NO pre-update here) ---
        if self.prev_synd_x_window is not None:
            kept_sigma_x = self.prev_synd_x_window[int(self.Offset_F):]
            kept_sigma_z = self.prev_synd_z_window[int(self.Offset_F):]
        else:
            kept_sigma_x = []
            kept_sigma_z = []

        sigma_x_windowed_pre = kept_sigma_x + new_sigma_x_list
        sigma_z_windowed_pre = kept_sigma_z + new_sigma_z_list

        # --- build Δσ in-window ---
        d_x = [sigma_x_windowed_pre[0]]
        d_z = [sigma_z_windowed_pre[0]]
        for w in range(1, len(sigma_x_windowed_pre)):
            d_x.append((sigma_x_windowed_pre[w] + sigma_x_windowed_pre[w - 1]) % 2)
            d_z.append((sigma_z_windowed_pre[w] + sigma_z_windowed_pre[w - 1]) % 2)

        d_x_stacked = np.concatenate(d_x, axis=0)
        d_z_stacked = np.concatenate(d_z, axis=0)

        # --- decode window ---
        self.bpd_x.decode(d_x_stacked)
        self.bpd_z.decode(d_z_stacked)

        c_x_full = np.asarray(self.bpd_x.osdw_decoding, dtype=np.uint8) % 2
        c_z_full = np.asarray(self.bpd_z.osdw_decoding, dtype=np.uint8) % 2

        n_x = int(self.hz.shape[1])
        n_z = int(self.hx.shape[1])
        F = int(self.Offset_F)

        # --- commit ξ = first F data blocks ---
        c_new_x_1_to_F = np.zeros(n_x, dtype=np.uint8)
        c_new_z_1_to_F = np.zeros(n_z, dtype=np.uint8)
        for i in range(F):
            c_new_x_1_to_F = (c_new_x_1_to_F + c_x_full[i * n_x:(i + 1) * n_x]) % 2
            c_new_z_1_to_F = (c_new_z_1_to_F + c_z_full[i * n_z:(i + 1) * n_z]) % 2

        # --- update cumulative committed correction ---
        if self.prev_c_x_accu2 is None or self.prev_c_z_accu2 is None:
            raise RuntimeError("prev_c_*_accu2 is None. Must be initialized in lifetime_trial().")

        c_x_accu2 = (self.prev_c_x_accu2 + c_new_x_1_to_F) % 2
        c_z_accu2 = (self.prev_c_z_accu2 + c_new_z_1_to_F) % 2
        
        # --- post-commit syndrome frame update ONCE by H ξ ---
        H_xi_x = np.asarray((self.hz @ c_new_x_1_to_F) % 2, dtype=np.uint8).ravel()
        H_xi_z = np.asarray((self.hx @ c_new_z_1_to_F) % 2, dtype=np.uint8).ravel()

        sigma_x_windowed_post = [(s + H_xi_x) % 2 for s in sigma_x_windowed_pre]
        sigma_z_windowed_post = [(s + H_xi_z) % 2 for s in sigma_z_windowed_pre]

        # --- store for next round ---
        self.prev_synd_x_window = sigma_x_windowed_post
        self.prev_synd_z_window = sigma_z_windowed_post
        self.prev_E_x_list = [e.copy() for e in E_x_list]
        self.prev_E_z_list = [e.copy() for e in E_z_list]
        self.prev_c_x_accu2 = c_x_accu2.copy()
        self.prev_c_z_accu2 = c_z_accu2.copy()

        # also store last committed only (optional)
        self.prev_corrections_x = c_new_x_1_to_F.copy()
        self.prev_corrections_z = c_new_z_1_to_F.copy()

        # ------------------------------------------------------------
        # --- logical test (ideal decoder) ---
        # R_test uses errors up to committed frontier (drop newest W-F errors),
        # but still adds all committed corrections.
        # Early pass if both R_x_test and R_z_test are zero.
        # ------------------------------------------------------------
        W = int(self.Window_W)
        tail_len = max(0, W - F)

        E_x_sumc_test = self._sum_error_list_gf2(E_x_list, drop_last=tail_len)
        E_z_sumc_test = self._sum_error_list_gf2(E_z_list, drop_last=tail_len)

        R_x_test = (E_x_sumc_test + c_x_accu2) % 2
        R_z_test = (E_z_sumc_test + c_z_accu2) % 2

        # --- EARLY PASS: zero test residual ---
        if (not R_x_test.any()) and (not R_z_test.any()):
            # Returned residual stays full-history residual
            E_x_sumc_full = self._sum_error_list_gf2(E_x_list, drop_last=0)
            E_z_sumc_full = self._sum_error_list_gf2(E_z_list, drop_last=0)
            residual_x = (E_x_sumc_full + c_x_accu2) % 2
            residual_z = (E_z_sumc_full + c_z_accu2) % 2
            return residual_x, residual_z, False

        sigma_x_ideal = np.asarray((self.hz @ R_x_test) % 2, dtype=np.uint8).ravel()
        sigma_z_ideal = np.asarray((self.hx @ R_z_test) % 2, dtype=np.uint8).ravel()

        self.bpd_x_ideal.decode(sigma_x_ideal)
        self.bpd_z_ideal.decode(sigma_z_ideal)

        c_x_ideal = np.asarray(self.bpd_x_ideal.osdw_decoding, dtype=np.uint8) % 2
        c_z_ideal = np.asarray(self.bpd_z_ideal.osdw_decoding, dtype=np.uint8) % 2

        R_x_ideal = (R_x_test + c_x_ideal) % 2
        R_z_ideal = (R_z_test + c_z_ideal) % 2

        fail_x = ((self.lz @ R_x_ideal) % 2).any()
        fail_z = ((self.lx @ R_z_ideal) % 2).any()
        failed = bool(fail_x or fail_z)

        # ------------------------------------------------------------
        # Returned residual stays full-history residual
        # ------------------------------------------------------------
        E_x_sumc_full = self._sum_error_list_gf2(E_x_list, drop_last=0)
        E_z_sumc_full = self._sum_error_list_gf2(E_z_list, drop_last=0)
        residual_x = (E_x_sumc_full + c_x_accu2) % 2
        residual_z = (E_z_sumc_full + c_z_accu2) % 2

        return residual_x, residual_z, failed

    # ------------------------------------------------------------
    # One trial 
    # ------------------------------------------------------------
    def lifetime_trial(self):
        """
        Runs one lifetime trial.
        Returns (lifetime, censored_bool).
        """

        # Reset per-trial state
        self.prev_synd_x_window = None
        self.prev_synd_z_window = None
        self.prev_corrections_x = None
        self.prev_corrections_z = None
        self.prev_E_x_list = None
        self.prev_E_z_list = None
        self.prev_c_x_accu2 = None
        self.prev_c_z_accu2 = None

        p_meas = float(self.meas_error_rate)

        # ---- initial W rounds ----
        E_x = np.zeros(self.N, dtype=np.uint8)
        E_z = np.zeros(self.N, dtype=np.uint8)

        E_x_list = []
        E_z_list = []
        sigma_x_list = []
        sigma_z_list = []

        for _ in range(int(self.Window_W)):
            e_x_new, e_z_new = self._generate_error()
            E_x = (E_x + e_x_new) % 2
            E_z = (E_z + e_z_new) % 2

            E_x_list.append(e_x_new.copy())
            E_z_list.append(e_z_new.copy())

            eta_x = np.random.binomial(1, p_meas, size=self.hz.shape[0]).astype(np.uint8)
            eta_z = np.random.binomial(1, p_meas, size=self.hx.shape[0]).astype(np.uint8)

            sigma_x = (np.asarray((self.hz @ E_x) % 2, dtype=np.uint8).ravel() + eta_x) % 2
            sigma_z = (np.asarray((self.hx @ E_z) % 2, dtype=np.uint8).ravel() + eta_z) % 2

            sigma_x_list.append(sigma_x)
            sigma_z_list.append(sigma_z)

        # ---- build initial Δσ ----
        d_x = [sigma_x_list[0]]
        d_z = [sigma_z_list[0]]
        for w in range(1, int(self.Window_W)):
            d_x.append((sigma_x_list[w] + sigma_x_list[w - 1]) % 2)
            d_z.append((sigma_z_list[w] + sigma_z_list[w - 1]) % 2)

        d_x_stacked = np.concatenate(d_x, axis=0)
        d_z_stacked = np.concatenate(d_z, axis=0)

        # ---- decode initial window ----
        self.bpd_x.decode(d_x_stacked)
        self.bpd_z.decode(d_z_stacked)

        c_x_full = np.asarray(self.bpd_x.osdw_decoding, dtype=np.uint8) % 2
        c_z_full = np.asarray(self.bpd_z.osdw_decoding, dtype=np.uint8) % 2

        n_x = int(self.hz.shape[1])
        n_z = int(self.hx.shape[1])
        F = int(self.Offset_F)

        # ---- commit initial ξ (first F blocks) ----
        c_x_1_to_F = np.zeros(n_x, dtype=np.uint8)
        c_z_1_to_F = np.zeros(n_z, dtype=np.uint8)
        for i in range(F):
            c_x_1_to_F = (c_x_1_to_F + c_x_full[i * n_x:(i + 1) * n_x]) % 2
            c_z_1_to_F = (c_z_1_to_F + c_z_full[i * n_z:(i + 1) * n_z]) % 2

        # ------------------------------------------------------------
        # ---- initial logical test (R_test changed per your request + early pass) ----
        # ------------------------------------------------------------
        W = int(self.Window_W)
        tail_len = max(0, W - F)

        E_x_sumc_test = self._sum_error_list_gf2(E_x_list, drop_last=tail_len)
        E_z_sumc_test = self._sum_error_list_gf2(E_z_list, drop_last=tail_len)

        R_x_test = (E_x_sumc_test + c_x_1_to_F) % 2
        R_z_test = (E_z_sumc_test + c_z_1_to_F) % 2

        # --- EARLY PASS: zero test residual ---
        if not R_x_test.any() and not R_z_test.any():
            failed = False
        else:
            sigma_x_ideal = np.asarray((self.hz @ R_x_test) % 2, dtype=np.uint8).ravel()
            sigma_z_ideal = np.asarray((self.hx @ R_z_test) % 2, dtype=np.uint8).ravel()

            self.bpd_x_ideal.decode(sigma_x_ideal)
            self.bpd_z_ideal.decode(sigma_z_ideal)

            c_x_ideal = np.asarray(self.bpd_x_ideal.osdw_decoding, dtype=np.uint8) % 2
            c_z_ideal = np.asarray(self.bpd_z_ideal.osdw_decoding, dtype=np.uint8) % 2

            R_x_ideal = (R_x_test + c_x_ideal) % 2
            R_z_ideal = (R_z_test + c_z_ideal) % 2

            fail_x = ((self.lz @ R_x_ideal) % 2).any()
            fail_z = ((self.lx @ R_z_ideal) % 2).any()
            failed = bool(fail_x or fail_z)

        if failed:
            return 0, False

        # ---- prepare stored window syndromes (post-commit frame update ONCE) ----
        H_xi_x = np.asarray((self.hz @ c_x_1_to_F) % 2, dtype=np.uint8).ravel()
        H_xi_z = np.asarray((self.hx @ c_z_1_to_F) % 2, dtype=np.uint8).ravel()

        sigma_x_list_post = [(s + H_xi_x) % 2 for s in sigma_x_list]
        sigma_z_list_post = [(s + H_xi_z) % 2 for s in sigma_z_list]

        self.prev_synd_x_window = [s.copy() for s in sigma_x_list_post]
        self.prev_synd_z_window = [s.copy() for s in sigma_z_list_post]

        self.prev_E_x_list = [e.copy() for e in E_x_list]
        self.prev_E_z_list = [e.copy() for e in E_z_list]

        # cumulative committed
        self.prev_c_x_accu2 = c_x_1_to_F.copy()
        self.prev_c_z_accu2 = c_z_1_to_F.copy()

        # ------------------------------------------------------------
        # residual = SAME convention as before (full history)
        # ------------------------------------------------------------
        E_x_sumc_full = self._sum_error_list_gf2(E_x_list, drop_last=0)
        E_z_sumc_full = self._sum_error_list_gf2(E_z_list, drop_last=0)

        residual_x = (E_x_sumc_full + c_x_1_to_F) % 2
        residual_z = (E_z_sumc_full + c_z_1_to_F) % 2

        passed = 1
        for _round_num in range(2, int(self.max_rounds) + 1):
            residual_x, residual_z, failed = self._one_round_update_residual(residual_x, residual_z)
            if failed:
                return passed, False
            passed += 1

        return passed, True

    # ------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------
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
