import seaborn as sns
import matplotlib.pyplot as plt
import corner
from lyscripts.plot.utils import save_figure, get_size


def plot_cluster_assignments(
    assignment_matrix, labels=None, save_dir=None, logger=None
):
    # Assuming assignment_matrix is a 2D array with shape (n_models, n_clusters)
    fig, ax = plt.subplots(1, figsize=get_size(width="full"))
    sns.heatmap(assignment_matrix, annot=True, cmap="viridis", ax=ax)
    ax.set_ylabel("Models")
    ax.set_xlabel("Clusters")
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
    Creates individual corner plots for each cluster using the cluster parameters.
    """
    # Separate the parameters for each cluster
    cluster_params_grouped = [
        cluster_parameters[:, i * len(labels) : (i + 1) * len(labels)]
        for i in range(n_clusters)
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
