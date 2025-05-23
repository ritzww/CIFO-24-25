from scipy.stats import kruskal
import scikit_posthocs as sp
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import numpy as np



import seaborn as sns

#--------------------------------------------------------------------------------
# NON-PARAMETRIC 3+ GROUPS
#--------------------------------------------------------------------------------

# KRUSKALL-WALLIS TEST
def kruskal_wallis_test(groups, alpha=0.01):
    kruskal_wallis_stats, p_value = kruskal(*groups)
    
    # Total sample size and number of groups
    N = sum(len(g) for g in groups)
    k = len(groups)

    # Epsilon-squared effect size
    eta_squared = (kruskal_wallis_stats - k + 1) / (N - k)
    
    print(f"Kruskal-Wallis H-test statistic: {kruskal_wallis_stats:.3f}")
    print(f"P-value: {p_value:.3e}")
    print(f"Eta-squared effect size (η²): {eta_squared:.3f}")
    print(100*'-')
    
    if p_value < alpha:
        print("REJECT H0: At least one configuration has significantly different fitness scores.")
    else:
        print("FAIL TO REJECT H0: No significant differences.")



# POST HOC DUNN'S TEST
def run_dunn_posthoc(df, value_col, group_col, p_adjust_method="holm"):
    dunn_results = sp.posthoc_dunn(
        df,
        val_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust_method
    )
    return dunn_results


#--------------------------------------------------------------------------------
# NON-PARAMETRIC 2 GROUPS
#--------------------------------------------------------------------------------
def mann_whitney_u_test(group1, group2, alpha=0.01):
    n1 = len(group1)
    n2 = len(group2)

    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')

    print(f"Mann-Whitney U statistic: {stat:.3f}")
    print(f"P-value: {p:.3e}")
    print(100*'-')

    if p < alpha:
        print("REJECT H0: Groups differ significantly.")
        # Rank-biserial correlation: r_rb = (2U) / (n1*n2) - 1
        r_rb = (2 * stat) / (n1 * n2) - 1
        print(f"Rank-biserial correlation: {r_rb:.3f}")
    else:
        print("FAIL TO REJECT H0: No significant difference.")



#--------------------------------------------------------------------------------
# PLOTS
#--------------------------------------------------------------------------------


# PLOT DUNN'S TEST
def plot_dunn_results(dunn_results,n_decimals=4,shrink=0.7,figsize=(20, 20)):

    # Create mask for upper triangle to hide it in heatmap
    mask = np.triu(np.ones_like(dunn_results, dtype=bool))

    # Plot the heatmap with only lower triangle shown
    plt.figure(figsize=figsize)
    sns.heatmap(dunn_results, 
                mask=mask,
                cmap="GnBu", 
                annot=True, 
                fmt=f".{n_decimals}f", 
                cbar_kws={'label': 'p-value', 'shrink': shrink},
                linewidths=0.5, 
                linecolor='gray',
                square=True)

    plt.title("Dunn's Post-Hoc Test P-values (Holm-adjusted)", fontsize=16)
    plt.xlabel("Configuration")
    plt.ylabel("Configuration")

    plt.xticks(rotation=30, ha='right')  # Rotate bottom x-axis labels by 45 degrees
    plt.tight_layout()
    plt.show()

# PLOT FITNESS DISTRIBUTIONS
def plot_fitness_distribution(df,column,palette,title="Fitness Distribution",xlabel="Fitness",ylabel="Density",figsize=(12, 6),x='fitness'
    ):
    """
    df : pandas.DataFrame
        Data containing a numeric 'fitness' column.
    column : str
        Column whose categories define each curve.
    palette : dict
        Mapping {category_name: color_hex}.
    title : str, optional
        Figure title.

    """
    plt.figure(figsize=figsize)

    for name, colour in palette.items():
        sns.kdeplot(
            data=df[df[column] == name],
            x=x,
            alpha=0.5,
            linewidth=1.5,
            fill=True,
            label=name.replace("_", " "),
            color=colour
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=column.replace("_", " ").title())
    plt.tight_layout()
    plt.show()


def plot_fitness_distribution(df, column, palette, title="Fitness Distribution",
                              xlabel="Fitness", ylabel="Density",
                              figsize=(12, 6), x='fitness'):
    plt.figure(figsize=figsize)

    for name, colour in palette.items():
        subset = df[df[column] == name]

        sns.kdeplot(
            data=subset,
            x=x,
            alpha=0.5,
            linewidth=1.5,
            fill=True,
            label=str(name).replace("_", " "),
            color=colour,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=column.replace("_", " ").title())
    plt.tight_layout()
    plt.show()
