# from pathlib import Path
# import numpy as np
# import emcee
# import multiprocess as mp
# import os
# import logging
# from lysubsite.mixture_model.util_2 import create_models

# logger = logging.getLogger(__name__)
# global MODEL


# def log_prob_fn(theta: np.ndarray | list) -> float:
#     global MODEL
#     for t in theta:
#         if t < 0 or 1 < t:
#             return -10000
#     llh = MODEL.likelihood(given_param_args=theta, log=True)
#     if np.isnan(llh):
#         llh = -10000
#     if np.isinf(llh):
#         llh = -10000
#     return llh


# def emcee_sampling_ext(
#     llh_function,
#     n_params=None,
#     sample_name=None,
#     n_burnin=None,
#     nwalkers=None,
#     n_step=None,
#     start_with=None,
#     skip_burnin=False,
#     llh_args=None,
# ):
#     nwalkers = 20 * n_params if nwalkers is None else nwalkers
#     burnin = 1000 if n_burnin is None else n_burnin
#     nstep = 1000 if n_step is None else n_step
#     thin_by = 1
#     print(f"Dimension: {n_params} with n walkers: {nwalkers}")
#     output_name = sample_name

#     if False:
#         samples = np.load("samples/" + output_name + ".npy")
#     else:
#         created_pool = mp.Pool(os.cpu_count())
#         with created_pool as pool:
#             if start_with is None:
#                 starting_points = np.random.uniform(size=(nwalkers, n_params))
#             else:
#                 if np.shape(start_with) != np.shape(
#                     np.random.uniform(size=(nwalkers, n_params))
#                 ):
#                     starting_points = np.tile(start_with, (nwalkers, 1))
#                 else:
#                     starting_points = start_with
#             print(
#                 f"Start Burning (steps = {burnin}) with {created_pool._processes} cores"
#             )
#             burnin_sampler = emcee.EnsembleSampler(
#                 nwalkers,
#                 n_params,
#                 llh_function,
#                 args=llh_args,
#                 pool=pool,
#             )
#             burnin_results = burnin_sampler.run_mcmc(
#                 initial_state=starting_points, nsteps=burnin, progress=True
#             )

#             ar = np.mean(burnin_sampler.acceptance_fraction)
#             print(
#                 f"the HMM sampler for model 01 accepted {ar * 100 :.2f} % of samples."
#             )
#             starting_points = burnin_sampler.get_last_sample()[0]
#             # print(f"The shape of the last sample is {starting_points.shape}")
#             original_sampler_mp = emcee.EnsembleSampler(
#                 nwalkers,
#                 n_params,
#                 llh_function,
#                 args=llh_args,
#                 backend=None,
#                 pool=pool,
#             )
#             sampling_results = original_sampler_mp.run_mcmc(
#                 initial_state=starting_points,
#                 nsteps=nstep,
#                 progress=True,
#                 thin_by=thin_by,
#             )

#         ar = np.mean(original_sampler_mp.acceptance_fraction)
#         print(f"the HMM sampler for model accepted {ar * 100 :.2f} % of samples.")
#         samples = original_sampler_mp.get_chain(flat=True)
#         log_probs = original_sampler_mp.get_log_prob(flat=True)
#         end_point = original_sampler_mp.get_last_sample()[0]
#         if output_name is not None:
#             np.save(f"./samples/" + output_name, samples)
#         # plots["acor_times"].append(burnin_info["acor_times"][-1])
#         # plots["accept_rates"].append(burnin_info["accept_rates"][-1])
#     return samples, end_point, log_probs


# def sample_from_model(model, params_sampling, sample_path: Path, sample_name: str):
#     """Initiate sampling from a given model (loaded with patients), given some parameters."""

#     MODEL = model
#     ndim = len(model.get_params())
#     nwalkers = ndim * params_sampling["walkers_per_dim"]
#     thin_by = params_sampling["thin_by"]
#     logger.info(
#         f"Set up sampling for {type(MODEL)} model with {ndim} parameters and "
#         f"{len(model.patient_data)} patients"
#     )
#     # make sure path to output file exists
#     sample_path.parent.mkdir(parents=True, exist_ok=True)

#     # prepare backend
#     hdf5_backend = emcee.backends.HDFBackend(sample_path / sample_name, name="mcmc")

#     logger.info(f"Prepared sampling params & backend at {sample_path}")

#     sampling_chain = emcee_sampling_ext(
#     log_prob_fn,
#     n_params=ndim,
#     sample_name=sample_name,
#     n_burnin=params_sampling["nburnin"],
#     nwalkers=nwalkers
#     n_step=params_sampling["nsteps"],
#     start_with=None,
#     skip_burnin=False,
#     llh_args=None,
# ):

#     return sampling_chain
