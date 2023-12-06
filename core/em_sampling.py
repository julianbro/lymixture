from pathlib import Path
import emcee
import lymph
from typing import Optional, TypedDict, List, Union
import logging
import numpy as np
import multiprocess as mp
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from lyscripts.plot.utils import save_figure
from util_2 import set_size
from lyscripts.sample import sample_from_global_model_and_configs

global MODELS, N_CLUSTERS

logger = logging.getLogger(__name__)


def assign_global_params(models, n_clusters):
    global MODELS, N_CLUSTERS
    MODELS = models
    N_CLUSTERS = n_clusters


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def set_size(width="full", fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX."""
    # Width presets in inches
    width_dict = {"full": 6.202, "half": 3.031}
    width = width_dict.get(width, width)
    # Figure width in inches
    fig_width = width * fraction
    # Golden ratio to set aesthetic figure height
    fig_height = fig_width / 1.618
    return (fig_width, fig_height)


def plot_history(history, labels_w, models, n_clusters, save_dir=None):
    weights_hist = history["z_samples"]
    model_params_hist = history["thetas"]
    llh_hist = history["log_probs"]
    c_r = history["convergence_criteria_values"]

    fig, axs = plt.subplots(2, 2, figsize=set_size(width="full"))
    plt.rcParams.update({"font.size": 8})  # Adjust font size

    # Likelihood Plot
    axs[0][0].plot(range(len(llh_hist)), llh_hist, label="Likelihood")
    axs[0][0].set_xlabel("Steps")
    axs[0][0].set_ylabel("Log Likelihood")
    axs[0][0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Weights Plot
    ax = axs[0][1]
    for i, l in enumerate(labels_w):
        ax.plot(
            range(len(weights_hist)), [w[i] for w in weights_hist], label=f"π_{l},0"
        )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Weights")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize="small")

    # Model Parameters Plot
    ax = axs[1, 0]
    label_ts = [
        t.replace("primary", "T").replace("_spread", "")
        for t in models[0].get_params(as_dict=True).keys()
        if "primary" in t
    ]
    label_ts = [item for item in label_ts for _ in range(n_clusters)]

    for i in range(0, len(label_ts), n_clusters):
        for j in range(n_clusters):
            ax.plot(
                range(len(model_params_hist)),
                [w[i + j] for w in model_params_hist],
                label=f"{label_ts[i]}^{j}",
            )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Parameter Values")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize="small")

    fig.tight_layout(pad=3.0)

    if save_dir is not None:
        save_figure(
            save_dir / f"history_em", fig, formats=["png", "svg"], logger=logger
        )


def emcee_simple_sampler(
    log_prob_fn: callable,
    ndim: int,
    sampling_params: dict,
    hdf5_backend: Optional[emcee.backends.HDFBackend] = None,
    starting_point: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    llh_args: Optional[List] = None,
    models: Optional[List] = None,
    show_progress=True,
):
    global MODELS
    if models is not None:
        MODELS = models

    logger.info(
        f"Set up sampling for {len(MODELS)}x {type(MODELS[0])} model with {ndim} parameters"
    )

    nwalkers = sampling_params["walkers_per_dim"] * ndim
    burnin = sampling_params["nburnin"]
    nstep = sampling_params["nsteps"]
    # thin_by = 1

    created_pool = mp.Pool(os.cpu_count())
    with created_pool as pool:
        if starting_point is None:
            starting_point = np.random.uniform(size=(nwalkers, ndim))
        else:
            if np.shape(starting_point) != np.shape(
                np.random.uniform(size=(nwalkers, ndim))
            ):
                starting_point = np.tile(starting_point, (nwalkers, 1))

        original_sampler_mp = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn,
            args=llh_args,
            backend=hdf5_backend,
            pool=pool,
        )
        sampling_results = original_sampler_mp.run_mcmc(
            initial_state=starting_point,
            nsteps=nstep + burnin,
            progress=show_progress,
        )

    ar = np.mean(original_sampler_mp.acceptance_fraction)
    print(f"Accepted {ar * 100 :.2f} % of samples.")
    samples = original_sampler_mp.get_chain(flat=True, discard=burnin)
    log_probs = original_sampler_mp.get_log_prob(flat=True)
    end_point = original_sampler_mp.get_last_sample()[0]
    # if save_dir is not None:
    #     np.save(save_dir, samples)
    return samples, end_point, log_probs


def exceed_param_bound(p: np.ndarray) -> bool:
    """Checks if any element in p exceeds the range [0,1]"""
    for ps in p:
        if ps < 0 or 1 < ps:
            return True
    return False


def get_n_thetas() -> int:
    """Returns the number of model parameters of the models"""
    return len(MODELS[0].get_params()) * N_CLUSTERS


def get_n_mixing_params() -> int:
    """Returns the number of mixture parameters to sample. Note that this number is smaller than the total mixing parameters, since we do not need to sample all of them."""
    return (N_CLUSTERS - 1) * len(MODELS)


def group_model_params(theta: Union[np.ndarray, List[float]]):
    """
    Regroup the model params in a structure which is understandable by the model.
    """
    K = MODELS[0].n_mixture_components
    return [list(theta[i : i + K]) for i in range(0, len(theta), K)]


def group_mixing_params(z: np.ndarray):
    """
    Regroup the mixing parameters to a structure which is understandable by the model.
    N is number of models, K is number of clusters, z are mixing parameters
    """
    z_grouped = [
        z[i : i + (N_CLUSTERS - 1)] for i in range(0, len(MODELS), N_CLUSTERS - 1)
    ]
    return [[*zs, 1 - np.sum(zs)] for zs in z_grouped]


def assign_model_params(theta: Union[np.ndarray, List[float]]):
    theta_grouped = group_model_params(theta)
    for model in MODELS:
        model.assign_params(*theta_grouped)


def assign_mixing_parameters(z: Union[np.ndarray, List[float]]):
    """
    Assigns mixing parameters (latent variables) to models.

    Structure [z_11, z_12, .. , z1(K-1), z21, .., z2(K-1), .. , zS(K-1)]
    z_s,k, where s is subsite and k is cluster
    """
    z_grouped = group_mixing_params(z)
    for i, model in enumerate(MODELS):
        model.mixture_components = z_grouped[i]


def llh_z_given_theta(z, inverted=False):
    """Return the log likelihood of observing p(z | D, theta)"""
    # global MODELS, N_CLUSTERS

    if exceed_param_bound(z):
        return np.inf if inverted else -np.inf
    grouped_mixing_params = group_mixing_params(z)
    llh = 0
    for i, model in enumerate(MODELS):
        model.mixture_components = grouped_mixing_params[i]
        llh += model._hmm_likelihood(log=True)
    if np.isinf(llh):
        return np.inf if inverted else -np.inf
    return -llh if inverted else llh


def llh_theta_given_z(theta, inverted=False):
    """Return the log likelihood of observing p(theta | D, z)"""
    global MODELS, N_CLUSTERS

    if exceed_param_bound(theta):
        return np.inf if inverted else -np.inf

    theta_grouped = group_model_params(theta)

    llh = 0
    for i, model in enumerate(MODELS):
        model.assign_params(*theta_grouped)
        llh += model._hmm_likelihood(log=True)
    if np.isinf(llh):
        return np.inf if inverted else -np.inf
    return -llh if inverted else llh


def perform_expectation(
    given_approx: Union[np.ndarray, List[float]],
    method: str,
    save_dir: Optional[Path] = None,
    sampling_params: Optional[dict] = None,
    emcee_starting_point: Optional[np.ndarray] = None,
    MODELS=None,
):
    """
    Implements the Expectation step. Create posterior samples using the function defined by the method.
    In 'default' method, the expectation step takes an approximation of thetas and samples for mixing parameters
    In 'inverted' method, the expectation takes an approximation of the mixing parameters and samples for theta
    """
    # global MODELS, N_CLUSTERS
    if method != "inverted":
        assign_model_params(given_approx)
        n_sample_params = get_n_mixing_params()
        log_prob_fn = llh_z_given_theta
    else:
        MODELS = assign_mixing_parameters(given_approx)
        n_sample_params = get_n_thetas()
        log_prob_fn = llh_theta_given_z

    if sampling_params is None:
        sampling_params = {"walkers_per_dim": 20, "nsteps": 150, "nburnin": 300}

    sampler = sampling_params.get("sampler", "SIMPLE")
    show_progress = sampling_params.get("show_sampling_progress", True)

    if sampler == "SIMPLE":
        print("Simple Sampler")
        sample_chain, end_point, log_probs = emcee_simple_sampler(
            log_prob_fn,
            ndim=n_sample_params,
            sampling_params=sampling_params,
            starting_point=emcee_starting_point,
            save_dir=save_dir,
            show_progress=show_progress,
        )
    else:
        print("Pro Sampler")
        sample_chain, end_point, log_probs = sample_from_global_model_and_configs(
            log_prob_fn,
            n_sample_params,
            sampling_params,
            emcee_starting_point,
            save_dir,
        )
    return sample_chain, end_point, log_probs


# def draw_weights(theta: np.ndarray, method: str, last_weights: np.ndarray, draw_weights_args: dict, save_name: Optional[Path]):
#     """
#     Implements the Expectation step. Create posterior samples using the function defined by the method.
#     """
#     # global MODELS, N_CLUSTERS

#     # Sampling definitions
#     nburnin = nburnin
#     n_step = n_step
#     # assign theta to the models (actually, the model can do this already by its own)
#     assign_model_params(theta)

#     n_params = (N_CLUSTERS -1)*N
#     if method == 'emcee':
#         z_posterior, last_sample, log_probs = emcee_sampling_ext_em(llh_z_given_theta, n_params, sample_name = f"exp_{step}", nburnin=nburnin, n_step=n_step, start_with = last_sample)
#     else:

#     return z_posterior, last_sample, log_probs


def draw_m_imputations(posterior: np.ndarray, m: int):
    """A function which handles the logic of sampling from the posterior"""
    if m == 1:
        return [posterior.mean(axis=0)]
    else:
        return posterior[np.random.choice(posterior.shape[0], m, replace=False)]


def q_function(thetas, mixing_proposals):
    """Returns 1/m * sum (log(p(theta| z_j, D)))"""
    llh = 0
    for zj in mixing_proposals:
        MODELS = assign_mixing_parameters(zj)
        llh += llh_theta_given_z(thetas, inverted=True)

    return llh / len(mixing_proposals)


def q_function_inverted(z, mixing_proposals):
    """If method is inverted, returns 1/m * sum (log(p(z| theta_j, D)))"""
    llh = 0
    for thetaj in mixing_proposals:
        assign_model_params(thetaj)
        llh += llh_z_given_theta(z, inverted=True)

    return llh / len(mixing_proposals)


def perform_maximization(
    proposal_params: list[list[float]],
    method: str,
    save_dir: Optional[str] = None,
    minimize_method: str = "SLSQP",
    starting_point: Optional[np.ndarray] = None,
):
    "Performs the maximation step: Returns the parameters which maximize the approximated Q function."
    # global MODELS
    # Sampling definitions
    nburnin = 300
    n_step = 100

    if method != "inverted":
        # Find thetas which maximize the q function
        # If we have only one proposal (equals to the number if imputations beeing 1) then we can assign the parameters already here. This makes it more performant.
        # if len(proposal_params) == 1:
        #     assign_mixing_parameters(proposal_params[0])
        n_params = get_n_thetas()
        n_log_prob_fn = q_function

    else:
        n_params = get_n_mixing_params()
        n_log_prob_fn = q_function_inverted

    initial_guess = starting_point
    if starting_point is None:
        initial_guess = np.random.random(n_params)

    param_bounds = [(0, 1)] * n_params
    res = minimize(
        n_log_prob_fn,
        initial_guess,
        args=(proposal_params),
        method=minimize_method,
        bounds=param_bounds,
    )

    params_hat = res.x
    log_probs_max = -res.fun
    return params_hat, log_probs_max


def find_mixture_components(
    initial_weights: Optional[np.ndarray] = None,
    initial_theta: Optional[np.ndarray] = None,
    method: str = "Default",
    base_dir: Path = None,
    sampling_params: Optional[dict] = None,
    m_imputations_fun: callable = lambda x: 1,
    max_steps=20,
    convergence_ths=0.015,
    minimize_method="SLSQP",
    # interactive_plotting=False,
):
    """This function implements the em-algorithm. TODO restructure it as class object!"""

    # make sure path to output file exists
    base_dir.parent.mkdir(parents=True, exist_ok=True)

    n_z = get_n_mixing_params()
    n_theta = get_n_thetas()
    # Initialize the weights to 0.5 if not given
    if initial_weights is not None and len(initial_weights) != n_z:
        raise ValueError(
            "Number of initial weight do not match with number of sampling weights."
        )

    # Initialize the set of thetas. If None, the algorithm will start with the Maximizing step
    if initial_theta is not None and len(initial_theta) != n_theta:
        raise ValueError(
            "Number of initial theta values do not match with the expected number."
        )

    # First proposals for the expcetatoin and maximations steps.
    if method != "inverted":
        proposal_params_maximation = (
            (np.zeros(n_z) + 0.5) if initial_weights is None else initial_weights
        )
        proposal_params_expectation = initial_theta
    else:
        proposal_params_maximation = (
            np.random.random(n_theta) if initial_theta is None else initial_theta
        )
        proposal_params_expectation = initial_weights

    emcee_end_point = None

    # Prepare the history object
    history = {
        "z_samples": [],
        "thetas": [],
        "log_probs": [],
        "convergence_criteria_values": [],
    }

    # If needed, fill the history object with the initial data:
    if proposal_params_expectation is None:
        if method != "inverted":
            history["z_samples"].append(proposal_params_maximation)
        else:
            history["thetas"].append(proposal_params_maximation)

    for step in range(max_steps):
        print(f"Step {step} " + "- " * 20)

        # Run the expectation step.
        if proposal_params_expectation is not None:
            print(f"Perform expectation step with method: {method}")
            llhs_e = []
            weights_e = []
            save_dir = base_dir / f"E_r{step}_{method}"

            exp_posterior, emcee_end_point, exp_log_probs = perform_expectation(
                given_approx=proposal_params_expectation,
                method=method,
                save_dir=save_dir,
                sampling_params=sampling_params["params_for_expectation"],
                emcee_starting_point=emcee_end_point,
            )

            # Dram m proposals from the posterior
            proposal_params_maximation = draw_m_imputations(
                exp_posterior, m_imputations_fun(step)
            )

            if method != "inverted":
                history["z_samples"].append(exp_posterior.mean(axis=0))
            else:
                history["thetas"].append(exp_posterior.mean(axis=0))

            print(
                f"    Proposal Params Exp: {np.round(np.mean(exp_posterior, axis = 0), 2)}"
            )
            # np.save(f"./samples/expl_oc_hp/em/" + f"{name}_z_samples_{step}", z_samples)
        else:
            proposal_params_maximation = [proposal_params_maximation]

        # Maximation
        print("Performing maximization: ")
        save_dir = base_dir / f"M_r{step}_{method}"
        proposal_params_expectation, max_log_probs = perform_maximization(
            proposal_params_maximation,
            method,
            save_dir,
            minimize_method,
            starting_point=None,
        )
        print(f"    Maximation Suggests: {np.round(proposal_params_expectation, 2)}")

        if method != "inverted":
            history["thetas"].append(proposal_params_expectation)
        else:
            history["z_samples"].append(proposal_params_expectation)
        history["log_probs"].append(max_log_probs)

        # Store the histor

        # convergence criteria
        converged = False
        try:
            max_steps_for_convergence = 3
            last_thetas_array = np.array(
                history["z_samples"][-max_steps_for_convergence:]
            )
            reference_theta_line = (
                last_thetas_array.sum(axis=0) / max_steps_for_convergence
            )
            squared_differences = (last_thetas_array - reference_theta_line) ** 2
            history["convergence_criteria_values"].append(
                np.sqrt(squared_differences.max())
            )
            converged = (squared_differences < convergence_ths**2).all()
        except:
            history["convergence_criteria_values"].append(0)
            pass

        # Store the history object every round:
        try:
            np.save(base_dir / f"history_{method}", history)
        except:
            pass
        if converged:
            print(f"Condition Fullfilled at step {step}")
            break

    if method != "inverted":
        z_final = np.mean(proposal_params_maximation, axis=0)
        theta_final = proposal_params_expectation
    else:
        theta_final = np.mean(proposal_params_maximation, axis=0)
        z = proposal_params_expectation

    return z_final, theta_final, history


def em_sampler(
    models: List[lymph.models.Unilateral],
    n_clusters: int,
    save_dir: Path,
    save_name: str,
    em_params: dict = None,
):
    """Handles the calling of the em sampling algorithm."""
    global MODELS, N_CLUSTERS

    MODELS = models
    N_CLUSTERS = n_clusters

    base_dir = save_dir / save_name
    # Make sure the path exists
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    # Configs
    if em_params is None:
        em_params = {}
    m_imputations_fun = em_params.get(
        "imputation_func", lambda x: int(5 / (20) * x + 1)
    )
    max_steps = em_params.get("max_steps", 10)

    method = em_params.get("method", "Default")

    PARAMS_DEFAULT = {
        "params_for_expectation": {
            "walkers_per_dim": 20,
            "nsteps": 10,
            "nburnin": 5,
            "sampler": "PRO",
        },
        "params_for_maximation": {"minimize_method": "SLSQP"},
    }
    sampling_params = em_params.get("sampling_params", PARAMS_DEFAULT)

    convergence_ths = em_params.get("convergence_ths", 0.01)

    z_final, theta_final, history = find_mixture_components(
        initial_theta=None,
        initial_weights=None,
        method=method,
        m_imputations_fun=m_imputations_fun,
        max_steps=max_steps,
        base_dir=base_dir,
        sampling_params=sampling_params,
        convergence_ths=convergence_ths,
    )

    return z_final, theta_final, history
