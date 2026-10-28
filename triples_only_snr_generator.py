import numpy as np
import os
from scipy import stats
import cupy as cp
import pandas as pd
from gbgpu.gbgpu import GBGPU
from gbgpu.thirdbody import GBGPUThirdBody

from gbgpu.utils.constants import *
from gbgpu.utils.utility import *
from gbgpu.thirdbody import third_body_factors

from typing import Callable, Union, List, Tuple

from lisatools.sensitivity import get_sensitivity


dir_real_info = "/expanse/lustre/projects/umn131/vgupta1/GBGPU/population_paper/Realizations_all/"
dir_snr_info  = "/expanse/lustre/projects/umn131/vgupta1/GBGPU/Triples_on_Erebor/injection_with_snr/eight_year/injection_with_snr_triples/"
real_map = {
    2: "s442",
    3: "s352",
    4: "s444",
    5: "s555",
    6: "s633"
}
scenarios = [f"{l}{n}" for l in "ABCD" for n in (1, 2)]

keys = [
        'id',
        'P',
        'l_deg',
        'b_deg',
        'd_kpc',
        'wd1_mass',
        'wd1_rad',
        'wd2_mass',
        'wd2_rad',
        'age_yr',
        'wd1_cool_time',
        'wd2_cool_time',
        'cos_i',
        'SeBa_id',
        'phi',
        'theta',
        'l_ecl',
        'b_ecl',
        'M3',
        'a3',
        'Phi3',
        'iota3', 
        'e3'
    ]
class TriplesSetupGuide():
    def __init__(self, *args, N=16384, gpu=0, batch_size=1000, waveform_kwargs: dict):
        self.batch_size = batch_size
        self.N = N
        if gpu is not None:
            self.use_gpu = True
            cp.cuda.runtime.setDevice(gpu)
        else:
            self.use_gpu = False

        self.base_info  = GBGPU(use_gpu=self.use_gpu)
        self.third_info = GBGPUThirdBody(use_gpu=self.use_gpu)
        self.waveform_kwargs = waveform_kwargs
        self.waveform_kwargs["use_c_implementation"] = False
        self.df = 1 / self.waveform_kwargs["T"]
        self.ll_cut = 0.0
        self.verbose = True

    def build_sampling_parameter_set(self, data_input):
        
        m1 = data_input['wd1_mass']
        m2 = data_input['wd2_mass']
        period = data_input['P'] * 24.0 * 3600.0 # days to seconds
        d_kpc = data_input['d_kpc']
        index_out = np.arange(len(m1))

        f0 = 2 / period
        Amp = get_amplitude(m1, m2, f0, d_kpc)
        fdot = get_fdot(f0, m1=m1, m2=m2)
        fddot = 11/3 * fdot ** 2 / f0
        phi0 = np.random.uniform(0.0, 2* np.pi, size=len(f0))
        iota = np.arccos(data_input['cos_i'])
        psi = np.random.uniform(0, np.pi, size=len(f0))
        lam = data_input['l_ecl']
        beta =data_input['b_ecl']

        # third body comps
        m3 = data_input['M3']
        a2 = data_input['a3'] * AU
        phi2 = data_input['Phi3']
        i2 = data_input['iota3']
        e2 = data_input['e3']
        M_tot_kg = (m1 + m2) * MSUN + m3 * Mjup
        P2 = (2 * np.pi * np.sqrt(a2 ** 3 / (G * M_tot_kg))) / YEAR
        # breakpoint()
        Omega2 = np.random.rand(len(f0)) * 2 * np.pi
        omega2 = np.random.rand(len(f0)) * 2 * np.pi

        A2, varpi, T2 = third_body_factors(
            m1 + m2,
            m3.copy(),
            P2.copy(),
            e2.copy(),
            i2.copy(),
            Omega2,
            omega2, 
            phi2,
            lam,
            beta,
            third_mass_unit="Mjup",
            third_period_unit="yrs",
        )

        out = np.array([
            Amp, 
            f0, 
            fdot, 
            fddot,
            phi0, 
            iota, 
            psi,
            lam,
            beta,
            A2,
            varpi,
            e2,
            P2,
            T2,
            i2,
            phi2,
            Omega2,
            omega2,
        ]).T

        return out
    
    def build_waveforms(self, gb_gen: Callable, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        waveform_kwargs = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs:
            waveform_kwargs.pop("N")

        if "oversample" in waveform_kwargs:
            waveform_kwargs.pop("oversample")

        gb_gen.run_wave(*params.T, N=self.N, **waveform_kwargs)
        return gb_gen.A, gb_gen.E, gb_gen.freqs
    
    def get_snr_information(self, params: np.ndarray):

        A_third, E_third, freqs_third = self.build_waveforms(self.third_info, params[:, :14])
        A_base, E_base, freqs_base = self.build_waveforms(self.base_info, params[:, :9])
        assert cp.all(freqs_base == freqs_third)

        freqs_third_in_psd = freqs_third if not self.use_gpu else freqs_third.get()
        psd = cp.asarray(get_sensitivity(freqs_third_in_psd.flatten(), sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.waveform_kwargs["T"] / YEAR).reshape(freqs_third.shape))

        third_third = 4 * self.df * cp.sum((A_third.conj() * A_third + E_third.conj() * E_third) / psd, axis=-1)
        third_base = 4 * self.df * cp.sum((A_third.conj() * A_base + E_third.conj() * E_base) / psd, axis=-1)
        base_base = 4 * self.df * cp.sum((A_base.conj() * A_base + E_base.conj() * E_base) / psd, axis=-1)
        
        phase_change = cp.angle(third_base)
        opt_snr = cp.sqrt(third_third.real)
        ll_diff = -1/2 * (third_third + base_base - 2 * cp.abs(third_base))
        
        try:
            # get back from gpu if needed
            phase_change = phase_change.get()
            opt_snr = opt_snr.get()
            ll_diff = ll_diff.get()

        except AttributeError:
            pass
        return (ll_diff, opt_snr, phase_change)
    
    def get_input_data(self, path_to_input_file):
        data = np.genfromtxt(path_to_input_file, dtype=None)
        dtype = np.dtype([(key, '<f8') for key in keys])
        data_triples = np.asarray([tuple(data[i]) for i in range(len(data))], dtype=dtype)

        return data_triples
        
    def run_single_model(self, directory: str, indicator: str, fp: str):
        
        print("Start third setup:", directory, indicator, fp)

        data_input = self.get_input_data(directory + fp)
        
        params_input = self.build_sampling_parameter_set(data_input)
        # breakpoint()
        num_sources = params_input.shape[0]
        inds_tmp = np.arange(0, num_sources, self.batch_size)

        if inds_tmp[-1] != num_sources - 1:
            inds_tmp = np.concatenate([inds_tmp, np.array([num_sources])])

        ll_info, snr_info, phase_change_info = np.zeros(num_sources), np.zeros(num_sources), np.zeros(num_sources)
        for iteration, (start, end) in enumerate(zip(inds_tmp[:-1], inds_tmp[1:])):
            ll_tmp, snr_tmp, phase_change_tmp = self.get_snr_information(params_input[start:end]) # used to be start:end, but it was missing a source.

            ll_info[start:end] = ll_tmp
            snr_info[start:end] = snr_tmp
            phase_change_info[start:end] = phase_change_tmp

            if self.verbose:
                print(f"{iteration + 1} of {len(inds_tmp[:-1])}")

        original_id = data_input["id"]
        self.readout(indicator, original_id, params_input, ll_info, snr_info, phase_change_info)

        if self.verbose:
            print("End triples setup:", directory, indicator, fp)

    def get_out_fp(self, indicator: str) -> str:    
        return dir_snr_info + "SNR_RUN" + f"_pop_for_search_{indicator}.dat"
    
    def readout(self, indicator: str, original_id: np.ndarray, params: np.ndarray, ll_info: np.ndarray, opt_snr: np.ndarray, phase_change: np.ndarray):

        out_fp = self.get_out_fp(indicator)
        # print(np.max(ll_info))
        # breakpoint()
        inds = np.where(ll_info < self.ll_cut)[0]
        ll_good = ll_info[inds]
        snr_good = opt_snr[inds]
        phase_change_good = phase_change[inds]

        tmp = np.array([ll_good, snr_good, phase_change_good]).T
        params_in = params[inds]

        readout = np.concatenate([params_in, tmp, inds[:, np.newaxis], original_id[inds][:, np.newaxis]], axis=1)

        header = ""

        for key in [
            "Amp", 
            'f0', 
            'fdot', 
            'fddot', 
            'phi0', 
            'iota', 
            'psi',
            'lam',
            'beta',
            'A2',
            'omegabar',
            'e2',
            'P2',
            'T2',
            'i2',
            'phi2',
            'Omega2',
            'omega2',
            'lldiff_base_marg',
            'snr',
            'phase_shift_base',
            'index',
            "orig_id"
            ]:
            header += key + ','

        header = header[:-1]

        if self.verbose:
            print("Saving", out_fp)

        np.savetxt(out_fp, readout, header=header, delimiter=',')

        # test it immediately to make sure it reads in
        check = np.genfromtxt(out_fp, names=True, delimiter=',', dtype=None)


dt = 15.0
Tobs = 8.0 * YEAR

N_data = int(Tobs / dt)
Tobs = N_data * dt
df = 1.0 / Tobs
oversample = 4

waveform_kwargs = dict(
    dt=dt,
    T=Tobs,
    N=None,
    oversample=oversample,
    use_c_implementation=False,
)


for realization in range(5, 7):
    directory = dir_real_info + f"Realization_{realization}/"
    for scen in scenarios:
        indicator = f"{scen}_{real_map[realization]}"
        fp = f"data_{indicator}.dat"
        start = TriplesSetupGuide(waveform_kwargs=waveform_kwargs)
        start.run_single_model(directory, indicator, fp)