import numpy as np
import os
from scipy import stats
import cupy as cp
import pandas as pd
from gbgpu.gbgpu import GBGPU

from gbgpu.utils.constants import *
from gbgpu.utils.utility import *
from typing import Callable, Union, List, Tuple

from lisatools.sensitivity import get_sensitivity



dir_real_info = "/expanse/lustre/projects/umn131/vgupta1/GBGPU/Triples_on_Erebor/Realization_dwds_only/"
dir_snr_info  = "/expanse/lustre/projects/umn131/vgupta1/GBGPU/Triples_on_Erebor/injection_with_snr/eight_year/injection_with_snr_dwd/"
real_map = {
    2: "s442",
    3: "s352",
    4: "s444",
    5: "s555",
    6: "s633"
}
scenarios = [f"{l}{n}" for l in "ABCD" for n in (1, 2)]

class BaseSetupGuide():
    def __init__(self, *args, N=16384, gpu=0, batch_size=1000, waveform_kwargs: dict):
        self.batch_size = batch_size
        self.N = N
        if gpu is not None:
            self.use_gpu = True
            cp.cuda.runtime.setDevice(gpu)
        else:
            self.use_gpu = False
        self.base_info = GBGPU(use_gpu=self.use_gpu)
        self.waveform_kwargs = waveform_kwargs
        self.waveform_kwargs["use_c_implementation"] = False
        self.df = 1 / self.waveform_kwargs["T"]

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

        A_base, E_base, freqs_base = self.build_waveforms(self.base_info, params[:, :9])

        freqs_base_in_psd = freqs_base if not self.use_gpu else freqs_base.get()
        psd = cp.asarray(get_sensitivity(freqs_base_in_psd.flatten(), sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.waveform_kwargs["T"] / YEAR).reshape(freqs_base.shape))

        base_base = 4 * self.df * cp.sum((A_base.conj() * A_base + E_base.conj() * E_base) / psd, axis=-1)
        
        opt_snr = cp.sqrt(base_base.real)
        
        try:
            # get back from gpu if needed
            opt_snr = opt_snr.get()

        except AttributeError:
            pass
        return opt_snr
    
    def get_input_data(self, path_to_input_file):
        data_base = pd.read_csv(path_to_input_file)
        data_base = data_base.drop(columns = ["Unnamed: 0"])
        return data_base
        
    def run_single_model(self, directory: str, indicator: str, fp: str):
        
        print("Start base setup:", directory, indicator, fp)

        data_input = self.get_input_data(directory + fp)
        
        params_input = self.build_sampling_parameter_set(data_input)
        # breakpoint()
        num_sources = params_input.shape[0]
        inds_tmp = np.arange(0, num_sources, self.batch_size)

        if inds_tmp[-1] != num_sources - 1:
            inds_tmp = np.concatenate([inds_tmp, np.array([num_sources])])

        snr_info = np.zeros(num_sources)
        for iteration, (start, end) in enumerate(zip(inds_tmp[:-1], inds_tmp[1:])):
            snr_tmp = self.get_snr_information(params_input[start:end]) # used to be start:end, but it was missing a source.

            snr_info[start:end] = snr_tmp

            
            print(f"{iteration + 1} of {len(inds_tmp[:-1])}")

        original_id = data_input["id"]
        # breakpoint()
        self.readout(indicator, original_id, params_input, snr_info)

        # self.add_to_status_file(indicator)
        print("End base setup:", directory, indicator, fp)

    def get_out_fp(self, indicator: str) -> str:    
        return dir_snr_info + "SNR_RUN" + f"_pop_for_search_{indicator}.dat"
    
    def readout(self, indicator: str, original_id: np.ndarray, params: np.ndarray, opt_snr: np.ndarray):

        out_fp = self.get_out_fp(indicator)
        # print(np.max(ll_info))
        # breakpoint()
        snr_good = opt_snr

        tmp = np.array([snr_good]).T
        params_in = params
        # breakpoint()
        original_id = np.array(original_id)
        readout = np.concatenate([params_in, tmp, original_id[:, np.newaxis]], axis=1)

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
            'snr',
            "orig_id"
            ]:
            header += key + ','

        header = header[:-1]

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
        fp = f"data_{indicator}.csv"
        start = BaseSetupGuide(waveform_kwargs=waveform_kwargs)
        start.run_single_model(directory, indicator, fp)