import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
import pandas as pd

# --- 1. SIMULATION AND STRATEGY PARAMETERS ---

# Set a seed for reproducibility of the random results
np.random.seed(42)

# Baseline fund characteristics
FUND_SIZE = 50_000_000
N_SIMULATIONS = 10_000
COMPANIES_PER_PORTFOLIO = 50

# Probabilistic model for company progression
GRADUATION_RATES = {
    'seed_to_a': 0.15,
    'a_to_b': 0.25,
    'b_to_exit': 0.30,
}

# Valuation step-ups for successful companies
VALUATION_STEPS = {
    'seed_to_a': 2.2,
    'a_to_b': 2,
}

# Parameters for the log-normal distribution of exit multiples
EXIT_DISTRIBUTION_PARAMS = {
    'mu': 1.6,
    'sigma': 1.5,
}

# --- Investment Strategy Definitions ---
STRATEGY_1 = {
    'name': "Big Initial Tickets",
    'initial_investment': 1_000_000,
    'follow_on_a': 0,
    'follow_on_b': 0,
}

STRATEGY_2 = {
    'name': "Small Initial Tickets, Big Follow-ons",
    'initial_investment': 400_000,
    'follow_on_a': 2_000_000,
    'follow_on_b': 2_000_000,
}


# --- 2. SIMULATION LOGIC ---

def simulate_one_portfolio(strategy: dict) -> tuple[float, float]:
    """
    Simulates a single venture capital portfolio from seed to exit.

    Returns:
        A tuple containing: (portfolio_return_multiple, total_capital_invested)
    """
    capital_remaining = FUND_SIZE
    total_invested = 0
    total_final_value = 0

    for _ in range(COMPANIES_PER_PORTFOLIO):
        invested_in_company = 0
        current_stake_value = 0

        # Seed Stage
        seed_investment = strategy['initial_investment']
        if capital_remaining >= seed_investment:
            capital_remaining -= seed_investment
            total_invested += seed_investment
            current_stake_value = seed_investment
        else:
            continue

        # Series A Stage
        if np.random.rand() < GRADUATION_RATES['seed_to_a']:
            current_stake_value *= VALUATION_STEPS['seed_to_a']
            series_a_investment = strategy['follow_on_a']
            if capital_remaining >= series_a_investment:
                capital_remaining -= series_a_investment
                total_invested += series_a_investment
                current_stake_value += series_a_investment
        else:
            total_final_value += 0
            continue

        # Series B Stage
        if np.random.rand() < GRADUATION_RATES['a_to_b']:
            current_stake_value *= VALUATION_STEPS['a_to_b']
            series_b_investment = strategy['follow_on_b']
            if capital_remaining >= series_b_investment:
                capital_remaining -= series_b_investment
                total_invested += series_b_investment
                current_stake_value += series_b_investment
        else:
            total_final_value += 0
            continue

        # Exit Stage
        if np.random.rand() < GRADUATION_RATES['b_to_exit']:
            exit_multiple = np.random.lognormal(
                mean=EXIT_DISTRIBUTION_PARAMS['mu'],
                sigma=EXIT_DISTRIBUTION_PARAMS['sigma']
            )
            total_final_value += current_stake_value * exit_multiple
        else:
            total_final_value += 0

    if total_invested == 0:
        return 0, 0

    return total_final_value / total_invested, total_invested


def run_simulation():
    """Runs the full Monte Carlo simulation for both strategies."""
    print("Running simulations... this may take a moment.")
    results_strat1 = [simulate_one_portfolio(STRATEGY_1) for _ in range(N_SIMULATIONS)]
    results_strat2 = [simulate_one_portfolio(STRATEGY_2) for _ in range(N_SIMULATIONS)]
    print("Simulations complete.")

    # Unpack results into separate arrays for multiples and invested capital
    multiples1 = np.array([res[0] for res in results_strat1])
    invested1 = np.array([res[1] for res in results_strat1])
    multiples2 = np.array([res[0] for res in results_strat2])
    invested2 = np.array([res[1] for res in results_strat2])

    return multiples1, invested1, multiples2, invested2


# --- 3. VISUALIZATION ---

def generate_visualizations(m1, i1, m2, i2):
    """Generates and saves all plots for the analysis."""
    print("\nGenerating and saving visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Histogram (Original) ---
    plt.figure(figsize=(12, 7))
    clip_max = 15
    sns.histplot(np.clip(m1, 0, clip_max), bins=50, kde=True, stat='density',
                 label=STRATEGY_1['name'], color='royalblue', alpha=0.7)
    sns.histplot(np.clip(m2, 0, clip_max), bins=50, kde=True, stat='density',
                 label=STRATEGY_2['name'], color='darkorange', alpha=0.7)
    plt.title('Distribution of Portfolio Return Multiples (Clipped at 15x)', fontsize=16)
    plt.xlabel('Portfolio Return Multiple (x)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.savefig('1_histogram_returns_distribution.png', dpi=300)
    plt.close()

    # --- Plot 2: Box Plot (New) ---
    plt.figure(figsize=(10, 8))
    plot_data = pd.DataFrame({
        STRATEGY_1['name']: m1,
        STRATEGY_2['name']: m2
    })
    sns.boxplot(data=plot_data, palette=['royalblue', 'darkorange'])
    plt.title('Comparison of Portfolio Return Distributions', fontsize=16)
    plt.ylabel('Portfolio Return Multiple (x)', fontsize=12)
    # Use a log scale to better visualize the spread of high-return outliers
    plt.yscale('log')
    plt.ylim(top=200)  # Adjust ylim to focus on the main distribution
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('2_boxplot_comparison.png', dpi=300)
    plt.close()

    # --- Plot 3: Cumulative Distribution Function (CDF) (New) ---
    plt.figure(figsize=(12, 7))
    sns.ecdfplot(m1, label=STRATEGY_1['name'], color='royalblue')
    sns.ecdfplot(m2, label=STRATEGY_2['name'], color='darkorange')
    plt.title('Cumulative Distribution of Portfolio Returns', fontsize=16)
    plt.xlabel('Portfolio Return Multiple (x)', fontsize=12)
    plt.ylabel('Probability (P(Return <= x))', fontsize=12)
    plt.xlim(0, 10)  # Focus on the 0-10x range
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axvline(3, color='green', linestyle='--', label='3x Return Threshold')
    plt.axvline(1, color='red', linestyle='--', label='1x (Breakeven)')
    plt.legend()
    plt.savefig('3_cdf_plot.png', dpi=300)
    plt.close()

    # --- Plot 4: Invested Capital vs. Return (New) ---
    plt.figure(figsize=(12, 7))
    plt.scatter(i1 / 1e6, m1, alpha=0.3, label=STRATEGY_1['name'], color='royalblue')
    plt.scatter(i2 / 1e6, m2, alpha=0.3, label=STRATEGY_2['name'], color='darkorange')
    plt.title('Invested Capital vs. Portfolio Return', fontsize=16)
    plt.xlabel('Total Capital Invested ($ Millions)', fontsize=12)
    plt.ylabel('Portfolio Return Multiple (x)', fontsize=12)
    plt.yscale('log')  # Log scale helps see the wide distribution of returns
    plt.axhline(1, color='red', linestyle='--', label='1x (Breakeven)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('4_invested_capital_vs_return.png', dpi=300)
    plt.close()

    print("All visualizations have been saved to disk.")

def analyze_and_present_results(m1, i1, m2, i2):
    """Performs statistical analysis and prints results to console."""
    print("\n" + "=" * 50)
    print("      PERFORMANCE SUMMARY & STATISTICAL ANALYSIS")
    print("=" * 50)

    stats = {
        STRATEGY_1['name']: {
            'Average Multiple': np.mean(m1), 'Median Multiple': np.median(m1),
            'Std Dev of Returns': np.std(m1), 'Prob. >3x Return': np.mean(m1 > 3) * 100,
            'Prob. >5x Return': np.mean(m1 > 5) * 100, 'Prob. Loss (<1x)': np.mean(m1 < 1) * 100,
            'Avg Capital Deployed ($M)': np.mean(i1) / 1e6,
        },
        STRATEGY_2['name']: {
            'Average Multiple': np.mean(m2), 'Median Multiple': np.median(m2),
            'Std Dev of Returns': np.std(m2), 'Prob. >3x Return': np.mean(m2 > 3) * 100,
            'Prob. >5x Return': np.mean(m2 > 5) * 100, 'Prob. Loss (<1x)': np.mean(m2 < 1) * 100,
            'Avg Capital Deployed ($M)': np.mean(i2) / 1e6,
        }
    }

    print("\n--- Summary Statistics ---")
    for name, data in stats.items():
        print(f"\nStrategy: {name}")
        for key, val in data.items():
            unit = '%' if 'Prob.' in key else 'x' if 'Multiple' in key or 'Std Dev' in key else ''
            print(f"  {key:<28}: {val:.2f}{unit}")

    # --- Hypothesis Test: Mann-Whitney U Test ---
    u_statistic, p_value = mannwhitneyu(m1, m2, alternative='less')

    print("\n\n--- Hypothesis Test: Mann-Whitney U Test ---")
    print(f"H₀: There is no significant difference in the mean portfolio return multiple.")
    print(f"H₁: The mean return of '{STRATEGY_1['name']}' is greater than '{STRATEGY_2['name']}'.")
    print(f"\nU-Statistic: {u_statistic:.2f}")
    print(f"P-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: P-value < {alpha}. We REJECT the null hypothesis.")
    else:
        print(f"\nConclusion: P-value > {alpha}. We FAIL TO REJECT the null hypothesis.")

    # --- Confidence Intervals using Bootstrapping ---
    def bootstrap_median_ci(data, n_bootstrap=10000, ci=0.95):
        medians = [np.median(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
        lower = np.percentile(medians, (1 - ci) / 2 * 100)
        upper = np.percentile(medians, (1 + ci) / 2 * 100)
        return lower, upper

    ci1 = bootstrap_median_ci(m1)
    ci2 = bootstrap_median_ci(m2)

    print("\n\n--- 95% Confidence Intervals for Median Return Multiple ---")
    print(f"'{STRATEGY_1['name']}': ({ci1[0]:.2f}x, {ci1[1]:.2f}x)")
    print(f"'{STRATEGY_2['name']}': ({ci2[0]:.2f}x, {ci2[1]:.2f}x)")
    print("=" * 50)


if __name__ == '__main__':
    multiples1, invested1, multiples2, invested2 = run_simulation()
    analyze_and_present_results(multiples1, invested1, multiples2, invested2)
    generate_visualizations(multiples1, invested1, multiples2, invested2)