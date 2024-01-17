from locale import currency
from pathlib import Path
import emcee
from tqdm import tqdm

# from mixture_model import LymphMixtureModel
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
from costum_types import EMConfigType
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
    logger=None,
):
    # global MODELS
    # if models is not None:
    #     MODELS = models

    # logger.info(
    #     f"Set up sampling for {len(MODELS)}x {type(MODELS[0])} model with {ndim} parameters"
    # )

    nwalkers = sampling_params["walkers_per_dim"] * ndim
    burnin = sampling_params["nburnin"]
    nstep = sampling_params["nsteps"]
    # thin_by = 1

    created_pool = mp.Pool(os.cpu_count())
    with created_pool as pool:
        if starting_point is None:
            starting_point = np.random.uniform(size=(nwalkers, ndim)) / 10
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
            pool=None,
        )
        sampling_results = original_sampler_mp.run_mcmc(
            initial_state=starting_point,
            nsteps=nstep + burnin,
            progress=show_progress,
        )

    ar = np.mean(original_sampler_mp.acceptance_fraction)
    if logger:
        logger.debug(f"Accepted {ar * 100 :.2f} % of samples.")
    else:
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
        if ps <= 0 or 1 <= ps:
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


def draw_m_imputations(posterior: np.ndarray, m: int) -> List[np.ndarray]:
    """
    A function which handles the logic of sampling from the posterior.
    When m is 1, simply take the mode of the posterior. Whem m is larger than 1, sample from the posterior.
    """
    if m == 1:
        return [np.mean(posterior, axis=0)]
    else:
        return [
            posterior[idx]
            for idx in np.random.choice(posterior.shape[0], m, replace=False)
        ]


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


def boundary_condition_cluster_assignments(cluster_assignments):
    n_k = LMM.n_clusters
    for s in range(LMM.n_subpopulation):
        # print(cluster_assignments[s * (n_k - 1) : s + 1 * (n_k - 1)])
        # print(sum(cluster_assignments[s * (n_k - 1) : s + 1 * (n_k - 1)]))
        if sum(cluster_assignments[s * (n_k - 1) : (s + 1) * (n_k - 1)]) >= 1:
            return True

    return False


def log_ll_cl_assignments(cluster_assignments):
    if exceed_param_bound(
        cluster_assignments
    ) or boundary_condition_cluster_assignments(cluster_assignments):
        return -np.inf
    # print(cluster_assignments)
    llh = LMM.mm_hmm_likelihood(cluster_assignments=cluster_assignments)
    if np.isinf(llh):
        return -np.inf
    if np.isnan(llh):
        llh = -np.inf
    return llh


def log_ll_cl_parameters(cluster_parameters):
    if exceed_param_bound(cluster_parameters):
        return -np.inf
    llh = LMM.mm_hmm_likelihood(cluster_parameters=cluster_parameters)
    if np.isinf(llh):
        return -10000
    if np.isnan(llh):
        llh = -10000
    return llh


def log_ll_cl_parameters_multiple_assignments(cluster_parameters, cluster_assignments):
    """Returns 1/m * sum (log(p(theta| z_j, D)))"""
    if exceed_param_bound(cluster_parameters):
        return -np.inf
    llh = 0
    for cluster_assignment_j in cluster_assignments:
        # Sets the cluster assignments, and triggers recomputation of the corresponsing matrices.
        LMM.cluster_assignments = cluster_assignment_j

        llh += log_ll_cl_parameters(cluster_parameters)

    return llh / len(cluster_assignments)


def log_ll_cl_assignments_multiple_parameters(cluster_assignments, cluster_parameters):
    """Returns 1/m * sum (log(p(z_j| theta, D)))"""
    if exceed_param_bound(cluster_assignments):
        return -np.inf
    llh = 0
    for cluster_parameter_j in cluster_parameters:
        # Sets the cluster parameters, and triggers recomputation of the corresponsing matrices.
        LMM.cluster_parameters = cluster_parameter_j

        llh += log_ll_cl_assignments(cluster_assignments)

    return llh / len(cluster_parameters)


class ExpectationMaximization:
    def __init__(
        self,
        lmm,
        em_config: EMConfigType = None,
    ):
        """Class which implements the EM Algorithm.

        Note: We need an instance of the Mixture Model, since the implementations of the likelihood function are in the Mixture Model.
        """
        # For debugger
        from mixture_model import LymphMixtureModel

        lmm: LymphMixtureModel = lmm

        # Initialize logger
        self._setup_logger()

        # Get an instance of the Lymph Mixture Model and set it as a global variable
        global LMM
        self.lmm = lmm
        LMM = lmm

        # Set configs.
        self.em_config = self.default_em_config()
        if em_config is not None:
            self.set_em_config(em_config)

        # Parameters used by the ``:py:class:run_em`` method.
        self.current_step = 0
        self.max_steps = self.em_config["max_steps"]
        # Current best approximation of the cluster assignments
        self.current_cluster_assignments = np.array(
            [1 / self.lmm.n_clusters] * lmm.n_cluster_assignments
        )
        # Current Posterior of cluster assignments
        self.current_cluster_assignments_posterior = np.array(
            [[1 / self.lmm.n_clusters] * lmm.n_cluster_assignments]
        )

        # Current best approximation of cluster parameter
        self.current_cluster_parameters = np.array([0.2] * lmm.n_cluster_parameters)
        # Current Posterior of cluster parameters
        self.current_cluster_parameters_posterior = np.array(
            [[0.2] * lmm.n_cluster_parameters]
        )
        # Current best likelihood.
        self.current_likelihood = self.likelihood(
            self.current_cluster_parameters, self.current_cluster_assignments
        )

        self.em_dir = lmm.base_dir / "EM"
        self.em_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self):
        """
        Sets up the logger for the class.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def mcmc_sampling(self, log_prob_fn, ndim, llh_args=None, initial_guess=None):
        """Uses MCMC sampling to estimate the density of the given function. Returns the posterior chain and the log likelihoods."""
        if self.em_config["e_step"]["sampler"] == "SIMPLE":
            sample_chain, end_point, log_probs = emcee_simple_sampler(
                log_prob_fn,
                ndim=ndim,
                sampling_params=self.em_config["e_step"],
                starting_point=initial_guess,
                save_dir=self.em_dir,
                show_progress=self.em_config["e_step"]["show_progress"]
                & self.em_config["verbose"],
                logger=self.logger,
                llh_args=llh_args,
            )
        else:
            sample_chain, end_point, log_probs = sample_from_global_model_and_configs(
                log_prob_fn,
                ndim,
                sampling_params=self.em_config["e_step"],
                starting_point=initial_guess,
                save_dir=self.em_dir,
                models=LMM,
                llh_args=llh_args,
                verbose=self.em_config["e_step"]["show_progress"]
                & self.em_config["verbose"],
            )
        self.emcee_end_point = end_point
        return sample_chain, log_probs

    def maximize_estimation(self, maximize_fn, ndim, llh_args=None, initial_guess=None):
        """Returns the maximizer of the function given."""

        if initial_guess is None:
            initial_guess = np.random.random(ndim)

        param_bounds = [(0, 1)] * ndim
        res = minimize(
            lambda x, params: -1 * maximize_fn(x, *params),
            initial_guess,
            args=llh_args,
            method=self.em_config["m_step"]["minimize_method"],
            bounds=param_bounds,
        )

        maximizer = res.x
        self.maximizer_end_point = maximizer
        max_llh = -res.fun
        return maximizer, max_llh

    def init_run_em(self):
        """Initializes objects needed for runnning the EM algortihm."""
        # Define starting parameters for the samplers ()
        ndim_e = (
            self.lmm.n_cluster_assignments
            if self.em_config["method"] == "DEFAULT"
            else self.lmm.n_cluster_parameters
        )
        ndim_m = (
            self.lmm.n_cluster_parameters
            if self.em_config["method"] == "DEFAULT"
            else self.lmm.n_cluster_assignments
        )
        self.emcee_end_point = (
            np.random.uniform(
                size=(self.em_config["e_step"]["walkers_per_dim"] * ndim_e, ndim_e)
            )
            / 10
        )
        self.maximizer_end_point = np.random.random(ndim_m) / 10

        # Initialize the convergence checker
        self.convergence = Convergence(
            criterion=self.em_config["convergence"]["criterion"],
            config=self.em_config["convergence"],
        )

        # Initialize a history object
        self.history = History()
        self.history.add_entry(
            self.current_cluster_assignments,
            self.current_cluster_parameters,
            self.current_likelihood,
            self.convergence.get_initial_values(),
        )

    def run_em(self):
        logger.info(
            f"Run EM algorithm with method {self.em_config['method']} for max {self.max_steps} steps."
        )
        self.init_run_em()

        # Show additional information if verbose is set to true.
        if self.em_config["verbose"]:
            self.logger.setLevel(logging.DEBUG)
        # Run the EM algorithm.
        for iteration in tqdm(range(self.max_steps), desc="EM Algorithm Progress"):
            if self.em_config["method"] == "DEFAULT":
                self.e_step()
                self.m_step()
            else:
                self.e_step_sampling_cluster_parameters()
                self.m_step_sampling_cluster_assignments()

            # Update convergence..
            self.convergence.update(self)
            # .. and check convergence.
            converged = self.convergence.check()

            # Add entries to the history object.
            self.history.add_entry(
                self.current_cluster_assignments,
                self.current_cluster_parameters,
                self.current_likelihood,
                self.convergence.current_convergence_values,
            )

            # Store the history object in each round
            self.history.save(save_dir=self.em_dir)

            if converged:
                logger.info(f"Step {self.current_step} / {self.max_steps}: Converged!")
                break

            self.current_step = iteration

        if not converged:
            logger.warning(
                f"Max steps reached without convergence, return current approximation."
            )

        return self.current_cluster_assignments, self.history

    def e_step(self):
        self.logger.debug(f"Step {self.current_step}: Perform expectation.")
        # Set the cluster parameters to the current estimates.
        self.lmm.cluster_parameters = self.current_cluster_parameters
        # self.lmm.diagnose_matrices

        # show_progress = self.em_config["e_step"]["show_progress"]

        log_prob_fn = log_ll_cl_assignments

        # Use mcmc sampling to estimate the current cluster assignments
        cluster_assignments_posterior, log_probs = self.mcmc_sampling(
            log_prob_fn,
            self.lmm.n_cluster_assignments,
            initial_guess=self.emcee_end_point,
        )
        # print(cluster_assignments_posterior)
        # raise
        self.current_cluster_assignments = cluster_assignments_posterior.mean(axis=0)
        self.current_cluster_assignments_posterior = cluster_assignments_posterior
        logger.debug(f"Expectation yields: {self.current_cluster_assignments}")

    def m_step(self):
        """
        This step essentially maximizes the Q-Function.
        This means it takes imputations from the cluster assignment posterior,
        and returns the cluster parameters which maximize the likelihood over all imputations.
        """
        self.logger.debug(f"Step {self.current_step}: Perform maximation.")

        log_prob_fn = log_ll_cl_parameters_multiple_assignments

        # Draw m imputations from the estimated posterior of the cluster assignments
        cluster_assignment_imputations = draw_m_imputations(
            self.current_cluster_assignments_posterior,
            self.em_config["m_step"]["imputation_function"](self.current_step),
        )

        self.logger.debug(
            f"Number of imputations: {len(cluster_assignment_imputations)}"
        )
        cluster_parameter_proposal, max_llh = self.maximize_estimation(
            log_prob_fn,
            self.lmm.n_cluster_parameters,
            llh_args=[cluster_assignment_imputations],
            initial_guess=self.maximizer_end_point,
        )

        self.current_cluster_parameters = cluster_parameter_proposal
        self.current_likelihood = max_llh
        logger.debug(f"Maximation yields: {self.current_cluster_parameters}")

    def e_step_sampling_cluster_parameters(self):
        """This implements the E-Step of the EM algorithm in the 'inverted' method, where we sample for the cluster parameters given the current cluster assignments."""
        self.logger.debug(f"Step {self.current_step}: Perform expectation.")
        # Set the cluster parameters to the current estimates. This triggers recomputation of the matrices.
        self.lmm.cluster_assignments = self.current_cluster_assignments
        # self.lmm.diagnose_matrices

        # show_progress = self.em_config["e_step"]["show_progress"]

        log_prob_fn = log_ll_cl_parameters

        # Use mcmc sampling to estimate the current cluster parameters
        cluster_parameters_posterior, log_probs = self.mcmc_sampling(
            log_prob_fn,
            self.lmm.n_cluster_parameters,
            initial_guess=self.emcee_end_point,
        )

        self.current_cluster_parameters = cluster_parameters_posterior.mean(axis=0)
        self.current_cluster_parameters_posterior = cluster_parameters_posterior
        logger.info(f"Expectation yields: {self.current_cluster_parameters}")

    def m_step_sampling_cluster_assignments(self):
        """
        This implements the M-Step of the EM algorithm in the 'inverted' method, where we sample m imputations from the cl parameter posterior and find the cluster assignments which maximize the Q-function.
        """
        self.logger.info(f"Step {self.current_step}: Perform maximation.")

        log_prob_fn = log_ll_cl_assignments_multiple_parameters

        # Draw m imputations from the estimated posterior of the cluster parameters
        cluster_parameter_imputations = draw_m_imputations(
            self.current_cluster_parameters_posterior,
            self.em_config["m_step"]["imputation_function"](self.current_step),
        )

        cluster_assignment_proposal, max_llh = self.maximize_estimation(
            log_prob_fn,
            self.lmm.n_cluster_assignments,
            llh_args=[cluster_parameter_imputations],
            initial_guess=self.maximizer_end_point,
        )
        print(max_llh)
        self.current_cluster_assignments = cluster_assignment_proposal
        self.current_likelihood = max_llh
        logger.info(f"Maximation yields: {self.current_cluster_assignments}")

    @staticmethod
    def default_em_config() -> EMConfigType:
        default_config: EMConfigType = (
            {
                "max_steps": 15,
                "method": "DEFAULT",
                "e_step": {
                    "walkers_per_dim": 20,
                    "nsteps": 50,
                    "nburnin": 20,
                    "sampler": "SIMPLE",
                    "show_progress": True,
                },
                "m_step": {
                    "minimize_method": "SLSQP",
                    "imputation_function": lambda x: int(5 / (10) * x + 1),
                },
                "convergence": {
                    "criterion": "default",
                    "default": {"lookback_period": 3, "threshold": 0.010},
                },
            },
        )

        return default_config[0]

    def set_em_config(self, new_config):
        for k, v in new_config.items():
            self.em_config[k] = v

    def likelihood(self, cluster_parameters=None, cluster_assignment=None):
        # Access the likelihood function from LMM
        return self.lmm.mm_hmm_likelihood(cluster_parameters, cluster_assignment)


def reduce_to_1_dim(v: np.ndarray):
    """Reduce a numpy array to one dimension or compute the mean if it's not one-dimensional."""
    if v.ndim == 1:
        # If it's already one-dimensional, return it as is
        return v
    else:
        # Compute the mean along all dimensions and return a one-dimensional array with shape (2)
        return np.mean(v, axis=0)


class History:
    """
    Class which holds information about the history along the convergence process of the EM-algorithm.
    The class holds 4 different values: likelihood, cluster assignments, cluster parameters, and convergence values.
    """

    def __init__(self):
        self.cluster_assignments = []
        self.cluster_parameters = []
        self.likelihood = []
        self.convergence_value = []

    def add_entry(
        self, cluster_assignments, cluster_parameters, likelihood, conv_values
    ):
        """Adds an entry to the history object."""
        self.cluster_assignments.append(cluster_assignments)
        self.cluster_parameters.append(cluster_parameters),
        self.likelihood.append(likelihood)
        self.convergence_value.append(conv_values)

    def save(self, save_dir):
        """Build a history dict and save the dictionary."""
        obj = {
            "cluster_assignments": self.cluster_assignments,
            "cluster_parameters": self.cluster_parameters,
            "likelihood": self.likelihood,
            "convergence_value": self.convergence_value,
        }
        np.save(save_dir / "history", obj)

    def get_entries(self):
        return self.entries

    def plot_history(
        self, labels_subpopulation, parameter_labels, n_clusters, save_dir
    ):
        fig, axs = plt.subplots(2, 2, figsize=set_size(width="full"))
        plt.rcParams.update({"font.size": 8})  # Adjust font size

        # Likelihood Plot
        axs[0][0].plot(range(len(self.likelihood)), self.likelihood, label="Likelihood")
        axs[0][0].set_xlabel("Steps")
        axs[0][0].set_ylabel("Log Likelihood")
        axs[0][0].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Weights Plot
        ax = axs[0][1]
        for i, l in enumerate(labels_subpopulation):
            ax.plot(
                range(len(self.cluster_assignments)),
                [w[i] for w in self.cluster_assignments],
                label=f"π_{l},0",
            )
        ax.set_xlabel("Steps")
        ax.set_ylabel("Weights")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize="small"
        )

        # Model Parameters Plot
        ax = axs[1, 0]
        label_ts = [
            t.replace("primary", "T").replace("_spread", "")
            for t in parameter_labels
            if "primary" in t
        ]
        label_ts = [item for item in label_ts for _ in range(n_clusters)]

        for i in range(0, len(label_ts), n_clusters):
            for j in range(n_clusters):
                ax.plot(
                    range(len(self.cluster_parameters)),
                    [w[i + j] for w in self.cluster_parameters],
                    label=f"{label_ts[i]}^{j}",
                )
        ax.set_xlabel("Steps")
        ax.set_ylabel("Parameter Values")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize="small"
        )

        # Model Parameters Plot
        ax = axs[1, 1]
        label_ts = ["theta", "pi", "llh"]

        for i in range(0, len(label_ts)):
            ax.plot(
                range(len(self.convergence_value)),
                [v[i] for v in self.convergence_value],
                label=label_ts[i],
            )
        ax.set_xlabel("Steps")
        ax.set_ylabel("Convergence Values")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize="small"
        )

        fig.tight_layout()

        if save_dir is not None:
            save_figure(
                save_dir / f"history_em", fig, formats=["png", "svg"], logger=logger
            )


class Convergence:
    """Class which handles the convergence of the EM algorithm. The idea is one can define different 'convergence checker' using the criterion keyword."""

    def __init__(self, config, criterion="default"):
        self.criterion = criterion
        self.config = config

        if self.criterion == "default":
            # Define the parameters which are used by the 'default' convergence method.
            self.params = {
                "cluster_parameters": [],
                "cluster_assignments": [],
                "likelihood": [],
            }

        elif self.criterion == "posterior_quantiles":
            self.params = {"distribution": []}
        self.current_convergence_values = None

    def update(self, instance):
        # Update the convergence-related data
        if self.criterion == "default":
            self.params["cluster_parameters"].append(
                instance.current_cluster_parameters
            )
            self.params["cluster_assignments"].append(
                instance.current_cluster_assignments
            )
            self.params["likelihood"].append(instance.current_likelihood)

            #

    def get_initial_values(self):
        """Returns initial values for the given criterion."""
        if self.criterion == "default":
            return [1, 1, 1]

    def get_current_values(self):
        """Simply returns the current convergence values."""
        return self.current_convergence_values

    def check(self):
        # Implement the logic to check for convergence

        if self.criterion == "default":
            return self.default_convergence_check()

        return False

    def default_convergence_check(self):
        """checks that neither of the cluster assignment, cluster parameters, and likelihoods change more than a threshold value over the last n steps."""
        force_fail = False

        convergence_values = []
        lookback = self.config["default"]["lookback_period"]
        threshold = self.config["default"]["threshold"]

        # Check that cluster parameters are not changing more than a ths over the last steps
        lookback_cl_parameters = np.array(self.params["cluster_parameters"][-lookback:])
        cluster_parameters_change = np.abs(
            (lookback_cl_parameters - lookback_cl_parameters.mean(axis=0))
        )
        # print(lookback_cl_parameters)
        # print(f"mean: {lookback_cl_parameters.mean(axis=0)}")
        # print(f"change: {cluster_parameters_change}")
        # print(f"check: {cluster_parameters_change < threshold}")
        # print(cluster_parameters_change.max())
        convergence_check_cl_parameters = np.all(cluster_parameters_change < threshold)

        try:
            convergence_values.append(cluster_parameters_change.max())
        except:
            convergence_values.append(1)

        # Check that cluster assignments are not changing more than a ths over the last steps
        lookback_cl_assignments = np.array(
            self.params["cluster_assignments"][-lookback:]
        )
        cluster_assignments_change = np.abs(
            (lookback_cl_assignments - np.mean(lookback_cl_assignments, axis=0))
        )
        convergence_check_cl_assignments = np.all(
            cluster_assignments_change < threshold
        )

        try:
            convergence_values.append(cluster_assignments_change.max())
        except:
            convergence_values.append(1)

        # Check that likelihood is not changing more than a ths over the last steps
        lookback_llhs = np.array(self.params["likelihood"][-lookback:])
        likelihood_change = np.abs(np.diff(lookback_llhs) / lookback_llhs[1:])
        convergence_check_llh = np.all(likelihood_change < threshold)
        try:
            convergence_values.append(max(likelihood_change))
        except:
            convergence_values.append(1)
        self.current_convergence_values = convergence_values

        # If we have less rounds than the lookback period, we return False anyways.
        if len(self.params["cluster_parameters"]) <= lookback:
            return False

        return (
            convergence_check_cl_parameters
            & convergence_check_cl_assignments
            & convergence_check_llh
        )
