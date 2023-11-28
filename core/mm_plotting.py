import seaborn as sns
import matplotlib.pyplot as plt
import corner
from util_2 import set_size
from lyscripts.plot.utils import save_figure


def plot_cluster_assignments(
    assignment_matrix, labels=None, save_dir=None, logger=None
):
    # Assuming assignment_matrix is a 2D array with shape (n_models, n_clusters)
    fig, ax = plt.subplots(1, figsize=set_size(width="full"))
    sns.heatmap(assignment_matrix.T, annot=True, cmap="viridis", ax=ax)
    ax.set_xlabel("Models")
    ax.set_ylabel("Clusters")

    ax.set_yticklabels(labels)

    if save_dir is not None:
        save_figure(
            save_dir / f"cluster_assignments",
            fig,
            formats=["png", "svg"],
            logger=logger,
        )
    fig.show()


def plot_cluster_parameters(
    cluster_parameters, n_clusters, labels, figures_dir, logger
):
    """
    Creates corner plots for cluster parameters.

    """
    cluster_params_grouped = [
        cluster_parameters[:, i::n_clusters] for i in range(n_clusters)
    ]

    for i, chain in enumerate(cluster_params_grouped):
        if len(labels) != chain.shape[1]:
            raise RuntimeError(
                f"Length labels: {len(labels)}, shape chain: {chain.shape}"
            )

        fig = corner.corner(chain, labels=labels, show_titles=True)
        fig.suptitle(f"Cluster {i}", fontsize=16)
        save_figure(
            figures_dir / f"corner_cluster_{i}",
            fig,
            formats=["png", "svg"],
            logger=logger,
        )
        fig.show()
