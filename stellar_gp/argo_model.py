
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal
from .stellar_scalings import get_stellar_hypers
from .gp_kernels import SHO_kernel_full, qp, sqexp, m32, m52, m52pd, SHO_latent, per


# ===== Abstract Kernel Interface =====
class Kernel(ABC):
    @abstractmethod
    def psd(self, omega, nyquist_freq=None):
        pass

    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        return self.psd(np.abs(delta_t))  # fallback behavior



# ===== Specific Kernel Implementations =====
class OscillationKernel(Kernel):
    def __init__(self, amplitude, frequency, quality):
        self.amplitude = amplitude  # S0
        self.frequency = frequency  # omega0
        self.quality = quality      # Q

    def psd(self, omega, nyquist_freq=None):
        omega0 = self.frequency
        Q = self.quality
        S0 = self.amplitude
        return 4 * S0 * omega0**4 / ((omega**2 - omega0**2)**2 + omega0**2 * omega**2/Q**2)
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        if exp_time_accounting:
            return SHO_kernel_full(delta_t,dur1,dur2,self.frequency,self.quality,self.amplitude)
        else:
            return SHO_latent(delta_t,self.frequency,self.quality,self.amplitude)


class GranulationKernel(Kernel):
    def __init__(self, amplitude, frequency):
        self.amplitude = amplitude  # S0
        self.frequency = frequency  # omega0

    def psd(self, omega, nyquist_freq=None):
        return 4 * self.amplitude / ((omega / self.frequency)**4 + 1)
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        if exp_time_accounting:
            return SHO_kernel_full(delta_t,dur1,dur2,self.frequency,1/np.sqrt(2),self.amplitude)
        else:
            return SHO_latent(delta_t,self.frequency,1/np.sqrt(2),self.amplitude)


class QPKernel(Kernel):
    def __init__(self, amplitude, periodicity, coherence_timescale, harmonic_complexity):
        self.amplitude = amplitude
        self.periodicity = periodicity
        self.coherence_timescale = coherence_timescale
        self.harmonic_complexity = harmonic_complexity

    def psd(self, omega, nyquist_freq=None):
        raise NotImplementedError("PSD for QPKernel is not defined")
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        return qp(delta_t,self.amplitude,self.harmonic_complexity,self.coherence_timescale,self.periodicity)

class SEKernel(Kernel):
    def __init__(self, amplitude, lengthscale):
        self.amplitude = amplitude
        self.lengthscale = lengthscale

    def psd(self, omega, nyquist_freq=None):
        raise NotImplementedError("PSD for SEKernel is not defined")
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        return sqexp(delta_t,self.amplitude,self.lengthscale)


class PerKernel(Kernel):
    def __init__(self, amplitude, period, lengthscale):
        self.amplitude = amplitude
        self.period = period
        self.lengthscale = lengthscale

    def psd(self, omega, nyquist_freq=None):
        raise NotImplementedError("PSD for PerKernel is not defined")
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        return per(delta_t,self.amplitude,self.lengthscale,self.period)


class M32Kernel(Kernel):
    def __init__(self, amplitude, lengthscale):
        self.amplitude = amplitude
        self.lengthscale = lengthscale

    def psd(self, omega, nyquist_freq=None):
        raise NotImplementedError("PSD for M32Kernel is not defined")
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        return m32(delta_t,self.amplitude,self.lengthscale)


class M52Kernel(Kernel):
    def __init__(self, amplitude, lengthscale):
        self.amplitude = amplitude
        self.lengthscale = lengthscale

    def psd(self, omega, nyquist_freq=None):
        raise NotImplementedError("PSD for M52Kernel is not defined")
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        return m52(delta_t,self.amplitude,self.lengthscale)
    

class M52PDKernel(Kernel):
    def __init__(self, amplitude1, amplitude2, lengthscale):
        self.amplitude1 = amplitude1
        self.amplitude2 = amplitude2
        self.lengthscale = lengthscale

    def psd(self, omega, nyquist_freq=None):
        raise NotImplementedError("PSD for M52PDKernel is not defined")
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        return m52pd(delta_t,self.amplitude1,self.amplitude2,self.lengthscale)


class WNKernel(Kernel):
    def __init__(self, amplitude):
        self.amplitude = amplitude

    def psd(self, omega, nyquist_freq):
        return np.full_like(omega, self.amplitude**2 / nyquist_freq)
    
    def covariance(self, delta_t, dur1, dur2, exp_time_accounting=True):
        return sqexp(delta_t,self.amplitude)



# ===== GP Data Container =====
class GPData:
    def __init__(self, time, duration, rv, rv_err):
        self.time = np.array(time)
        self.duration = np.array(duration)
        self.rv = np.array(rv)
        self.rv_err = np.array(rv_err)

    def __len__(self):
        return len(self.time)


# ===== GP Model Container =====
class GPModel:
    def __init__(self, kernels=None, parameters=None, param_labels=None):
        self.kernels = kernels if kernels is not None else []
        self.parameters = np.array(parameters) if parameters is not None else np.array([])
        self.param_labels = param_labels if param_labels is not None else []

    def add_kernel(self, kernel):
        self.kernels.append(kernel)

    def update_kernels(self):
        raise NotImplementedError("update_kernels method needs implementation")


# ===== Covariance Matrix Construction =====
def covariance_matrix(model, data, ignore_errs=False, exp_time_accounting=True, pdmat=True):
    n = len(data)
    total_cov = np.zeros((n, n))

    delta_t = np.abs(data.time[:, None] - data.time[None, :])
    durs1 = np.tile(data.duration[:, None], (1, n))
    durs2 = np.tile(data.duration[None, :], (n, 1))

    for kernel in model.kernels:
        cov_func = np.vectorize(
            lambda dt, d1, d2: kernel.covariance(dt, d1, d2, exp_time_accounting=exp_time_accounting)
        )
        total_cov += cov_func(delta_t, durs1, durs2)

    if not ignore_errs:
        total_cov += np.diag(data.rv_err**2)

    return total_cov


def get_stellar_kernels(logg=4.43,Teff=5777):
    wosc, Sosc, Q, w1, S1, w2, S2 = get_stellar_hypers(logg=logg,Teff=Teff)
    k_osc = OscillationKernel(Sosc,wosc,Q)
    k_gran1 = GranulationKernel(S1,w1)
    k_gran2 = GranulationKernel(S2,w2)

    return k_osc, k_gran1, k_gran2

# ===== GP Sample Generation =====
def generate_ts(model, data, num_draws=1, ignore_errs=True):
    cov = covariance_matrix(model, data, ignore_errs=ignore_errs)
    return np.random.multivariate_normal(np.zeros(len(data)), cov, size=num_draws).T


def generate_ts_components(model, data, num_draws=1, ignore_errs=True):
    labels = label_kernels(model)
    ts_dict = {}
    ts_total = np.zeros((len(data), num_draws))

    for i, kernel in enumerate(model.kernels):
        temp_model = GPModel(kernels=[kernel])
        cov = covariance_matrix(temp_model, data, ignore_errs=True)
        sample = np.random.multivariate_normal(np.zeros(len(data)), cov, size=num_draws).T
        label = labels[i]
        ts_dict[label] = sample
        ts_total += sample

    if not ignore_errs:
        err_cov = np.diag(data.rv_err ** 2)
        ts_dict["err"] = np.random.multivariate_normal(np.zeros(len(data)), err_cov, size=num_draws).T

    ts_dict["tot"] = ts_total
    return ts_dict


# ===== Kernel Labeling Utilities =====
def label_kernels(model):
    type_counts = {}
    labels = []
    prefix_dict = {
        "OscillationKernel": "osc",
        "GranulationKernel": "gran",
        "QPKernel": "qp",
        "SEKernel": "se",
        "WNKernel": "wn",
        "M32Kernel": "m32",
        "M52Kernel": "m52",
    }

    for kernel in model.kernels:
        kernel_type = type(kernel).__name__
        type_counts[kernel_type] = type_counts.get(kernel_type, 0) + 1
        count = type_counts[kernel_type]
        label_prefix = prefix_dict.get(kernel_type, "k")
        labels.append(f"{label_prefix}{count}")

    return labels


def label_hypers(kernel):
    name = type(kernel).__name__
    if name == "OscillationKernel":
        return {
            "osc_amplitude": kernel.amplitude,
            "osc_frequency": kernel.frequency,
            "osc_strength": kernel.quality,
        }
    elif name == "GranulationKernel":
        return {
            "gran_amplitude": kernel.amplitude,
            "gran_frequency": kernel.frequency,
        }
    elif name == "QPKernel":
        return {
            "qp_amplitude": kernel.amplitude,
            "qp_periodicity": kernel.periodicity,
            "qp_coherence": kernel.coherence_timescale,
            "qp_harmonic_complexity": kernel.harmonic_complexity,
        }
    elif name == "PerKernel":
        return {
            "per_amplitude": kernel.amplitude,
            "per_period": kernel.period,
            "per_lengthscale": kernel.lengthscale,
        }
    elif name == "SEKernel":
        return {
            "se_amplitude": kernel.amplitude,
            "se_lengthscale": kernel.lengthscale,
        }
    elif name == "M32Kernel":
        return {
            "m32_amplitude": kernel.amplitude,
            "m32_lengthscale": kernel.lengthscale,
        }
    elif name == "M52Kernel":
        return {
            "m52_amplitude": kernel.amplitude,
            "m52_lengthscale": kernel.lengthscale,
        }
    elif name == "WNKernel":
        return {
            "se_amplitude": kernel.amplitude,
        }
    else:
        raise ValueError(f"Unknown kernel type: {name}")


def format_value(value):
    return f"{value:.4g}"


def model_summary(model):
    lines = ["Gaussian Process Model Summary:", "---------------------------------"]
    lines.append(f"Number of Kernels: {len(model.kernels)}\n")

    for i, kernel in enumerate(model.kernels):
        lines.append(f"Kernel {i+1}: {type(kernel).__name__}")
        for name, value in label_hypers(kernel).items():
            lines.append(f"    {name} = {format_value(value)}")
        lines.append("")

    lines.append("---------------------------------")
    return "\n".join(lines)


# Optionally override string representation
GPModel.__str__ = lambda self: model_summary(self)
GPModel.__repr__ = lambda self: model_summary(self)