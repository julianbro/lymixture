import emcee
from tqdm import tqdm

from typing import Optional, List
import logging
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lyscripts.plot.utils import save_figure, get_size
from lymixture.types import EMConfigType
from lymixture.utils import sample_from_global_model_and_configs
# from lyscripts.sample import sample_from_global_model_and_configs

global MODELS, N_CLUSTERS

logger = logging.getLogger(__name__)


def emcee_simple_sampler(
    log_prob_fn: callable,
    ndim: int,
    sampling_params: dict,
    hdf5_backend: Optional[emcee.backends.HDFBackend] = None,
    starting_point: Optional[np.ndarray] = None,
    llh_args: Optional[List] = None,
    show_progress=True,
):
    """A simple sampler."""
    nwalkers = sampling_params["walkers_per_dim"] * ndim
    burnin = sampling_params["nburnin"]
    nstep = sampling_params["nsteps"]

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
    )
    sampling_results = original_sampler_mp.run_mcmc(
        initial_state=starting_point,
        nsteps=nstep + burnin,
        progress=show_progress,
    )

    ar = np.mean(original_sampler_mp.acceptance_fraction)
    logger.info(f"Accepted {ar * 100 :.2f} % of samples.")
    samples = original_sampler_mp.get_chain(flat=True, discard=burnin)
    log_probs = original_sampler_mp.get_log_prob(flat=True)
    end_point = original_sampler_mp.get_last_sample()[0]

    return samples, end_point, log_probs


def exceed_param_bound(p: np.ndarray) -> bool:
    """Checks if any element in p exceeds the range [0,1]"""
    for ps in p:
        if ps <= 0 or 1 <= ps:
            return True
    return False


def draw_m_imputations(posterior: np.ndarray, m: int) -> List[np.ndarray]:
    """
    A function which handles the logic of sampling from the posterior.
    When m is 1, simply take the mode of the posterior. Whem m is larger than 1, sample
    from the posterior.
    """
    if m == 1:
        return [np.mean(posterior, axis=0)]

    return [
        posterior[idx]
        for idx in np.random.choice(posterior.shape[0], m, replace=False)
    ]


def boundary_condition_cluster_assignments(cluster_assignments):
    n_k = LMM.n_clusters
    for s in range(LMM.n_subpopulation):
        if sum(cluster_assignments[s * (n_k - 1) : (s + 1) * (n_k - 1)]) >= 1:
            return True

    return False


def log_ll_cl_assignments(cluster_assignments):
    if exceed_param_bound(
        cluster_assignments
    ) or boundary_condition_cluster_assignments(cluster_assignments):
        return -np.inf

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

        Note: We need an instance of the Mixture Model, since the implementations of
        the likelihood function are in the Mixture Model.
        """
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


    def mcmc_sampling(self, log_prob_fn, ndim, llh_args=None, initial_guess=None):
        """
        Uses MCMC sampling to estimate the density of the given function. Returns
        the posterior chain and the log likelihoods.
        """
        if self.em_config["e_step"]["sampler"] == "SIMPLE":
            sample_chain, end_point, log_probs = emcee_simple_sampler(
                log_prob_fn,
                ndim=ndim,
                sampling_params=self.em_config["e_step"],
                starting_point=initial_guess,
                show_progress=self.em_config["e_step"]["show_progress"]
                & self.em_config["verbose"],
                llh_args=llh_args,
            )
        else:
            sample_chain, end_point, log_probs = sample_from_global_model_and_configs(
                log_prob_fn,
                ndim,
                sampling_params=self.em_config["e_step"],
                starting_point=initial_guess,
                models=LMM,
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
        logger.info(f"Run EM with method {self.em_config['method']} for max {self.max_steps} steps.")
        self.init_run_em()

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

            if converged:
                logger.info(f"Step {self.current_step} / {self.max_steps}: Converged!")
                break

            self.current_step = iteration

        if not converged:
            logger.info("Max steps reached, no convergence, return current approximation.")

        return self.current_cluster_assignments, self.history


    def e_step(self):
        logger.info(f"Step {self.current_step}: Perform expectation.")
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
        # logger.info(cluster_assignments_posterior)
        # raise
        self.current_cluster_assignments = cluster_assignments_posterior.mean(axis=0)
        self.current_cluster_assignments_posterior = cluster_assignments_posterior
        logger.info(f"Expectation yields: {self.current_cluster_assignments}")


    def m_step(self):
        """
        This step essentially maximizes the Q-Function.
        This means it takes imputations from the cluster assignment posterior,
        and returns the cluster parameters which maximize the likelihood over all imputations.
        """
        logger.info(f"Step {self.current_step}: Perform maximation.")
        log_prob_fn = log_ll_cl_parameters_multiple_assignments

        # Draw m imputations from the estimated posterior of the cluster assignments
        cluster_assignment_imputations = draw_m_imputations(
            self.current_cluster_assignments_posterior,
            self.em_config["m_step"]["imputation_function"](self.current_step),
        )

        logger.info(f"Number of imputations: {len(cluster_assignment_imputations)}")
        cluster_parameter_proposal, max_llh = self.maximize_estimation(
            log_prob_fn,
            self.lmm.n_cluster_parameters,
            llh_args=[cluster_assignment_imputations],
            initial_guess=self.maximizer_end_point,
        )

        self.current_cluster_parameters = cluster_parameter_proposal
        self.current_likelihood = max_llh
        logger.info(f"Maximation yields: {self.current_cluster_parameters}")


    def e_step_sampling_cluster_parameters(self):
        """
        This implements the E-Step of the EM algorithm in the 'inverted' method, where
        we sample for the cluster parameters given the current cluster assignments.
        """
        logger.info(f"Step {self.current_step}: Perform expectation.")
        # Set the cluster parameters to the current estimates. This triggers recomputation of the matrices.
        self.lmm.cluster_assignments = self.current_cluster_assignments

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
        This implements the M-Step of the EM algorithm in the 'inverted' method, where
        we sample m imputations from the cl parameter posterior and find the cluster
        assignments which maximize the Q-function.
        """
        logger.info(f"Step {self.current_step}: Perform maximation.")

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
        self.current_cluster_assignments = cluster_assignment_proposal
        self.current_likelihood = max_llh
        logger.info(f"Maximation yields: {self.current_cluster_assignments}")


    @staticmethod
    def default_em_config() -> EMConfigType:
        return {
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
        }


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

    # Compute the mean along all dimensions and return a one-dimensional array with shape (2)
    return np.mean(v, axis=0)


class History:
    """
    Class which holds information about the history along the convergence process of
    the EM-algorithm. The class holds 4 different values: likelihood, cluster
    assignments, cluster parameters, and convergence values.
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
        fig, axs = plt.subplots(2, 2, figsize=get_size(width="full"))
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

        for i, label in enumerate(label_ts):
            ax.plot(
                range(len(self.convergence_value)),
                [v[i] for v in self.convergence_value],
                label=label,
            )
        ax.set_xlabel("Steps")
        ax.set_ylabel("Convergence Values")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize="small"
        )
        fig.tight_layout()

        if save_dir is not None:
            save_figure(save_dir/f"history_em", fig, formats=["png", "svg"])


class Convergence:
    """
    Class which handles the convergence of the EM algorithm. The idea is one can define
    different 'convergence checker' using the criterion keyword.
    """
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
            raise NotImplementedError()

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
        """
        checks that neither of the cluster assignment, cluster parameters, and
        likelihoods change more than a threshold value over the last n steps.
        """
        convergence_values = []
        lookback = self.config["default"]["lookback_period"]
        threshold = self.config["default"]["threshold"]

        # Check that cluster parameters are not changing too much
        lookback_cl_parameters = np.array(self.params["cluster_parameters"][-lookback:])
        cluster_parameters_change = np.abs(
            (lookback_cl_parameters - lookback_cl_parameters.mean(axis=0))
        )
        convergence_check_cl_parameters = np.all(cluster_parameters_change < threshold)

        # TODO: Ask for expected exception
        try:
            convergence_values.append(cluster_parameters_change.max())
        except:
            convergence_values.append(1)

        # Check that cluster assignments are not changing too much
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

        # Check that likelihood is not changing too much
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
