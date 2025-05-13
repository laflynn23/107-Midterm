# code/cct.py

import pandas as pd
import numpy as np
import pymc as pm
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt

# ----------------------------------------
# Load the Data
# ----------------------------------------
def load_plant_knowledge_data(path="data/plant_knowledge.csv"):
    """
    Loads the plant knowledge dataset from CSV and returns:
    - responses: N x M binary matrix (excluding Informant column)
    - informants: array of informant names
    """
    df = pd.read_csv(path)
    informants = df["Informant"].values
    responses = df.drop(columns=["Informant"]).values.astype(int)
    return responses, informants

# ----------------------------------------
# Define and Run the CCT Model
# ----------------------------------------
def build_and_sample_model(responses, draws=2000, tune=1000, chains=4):
    """
    Builds and samples from a Cultural Consensus Theory model.
    
    D[i] = competence of informant i ~ Uniform(0.5, 1)
    Z[j] = latent consensus answer for question j ~ Bernoulli(0.5)
    p_ij = Z[j] * D[i] + (1 - Z[j]) * (1 - D[i])
    X_ij ~ Bernoulli(p_ij)
    """
    N, M = responses.shape

    with pm.Model() as cct_model:
        # Define Priors
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)      # Competence
        Z = pm.Bernoulli("Z", p=0.5, shape=M)                   # Consensus answers

        # Broadcast D and Z to compute p_ij
        D_reshaped = D[:, None]       # shape (N, 1)
        Z_reshaped = Z[None, :]       # shape (1, M)

        # Calculate pij using compact formula
        p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)

        # Likelihood
        X = pm.Bernoulli("X", p=p, observed=responses)

        # MCMC sampling
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, return_inferencedata=True, random_seed=42)

    return trace

# ----------------------------------------
# Analyze Posterior Results
# ----------------------------------------
def analyze_results(trace, responses, informants):
    """
    Summarize posterior estimates, compute R-hat diagnostics,
    and compare consensus to naive majority vote.
    """
    # Posterior means
    D_mean = trace.posterior["D"].mean(dim=("chain", "draw")).values
    Z_mean = trace.posterior["Z"].mean(dim=("chain", "draw")).values
    consensus_key = np.round(Z_mean).astype(int)

    # Naive majority vote
    majority_vote = np.round(responses.mean(axis=0)).astype(int)

    # Output: competence estimates
    print("=== Informant Competence Estimates ===")
    most = np.argmax(D_mean)
    least = np.argmin(D_mean)
    for i, name in enumerate(informants):
        marker = " <== MOST" if i == most else " <== LEAST" if i == least else ""
        print(f"{name}: {D_mean[i]:.3f}{marker}")

    # Output: consensus answers
    print("\n=== Posterior Consensus Answer Key (Z) ===")
    print(" ".join(map(str, consensus_key)))

    print("\n=== Naive Majority Vote ===")
    print(" ".join(map(str, majority_vote)))

    # Output: disagreements
    print("\n=== Differences (Consensus ≠ Majority) ===")
    print(" ".join(["✔" if a != b else " " for a, b in zip(consensus_key, majority_vote)]))

    # R-hat diagnostics
    summary = az.summary(trace, var_names=["D", "Z"])
    print("\n=== R-hat Summary ===")
    print(summary[["mean", "hdi_3%", "hdi_97%", "r_hat"]])

    return D_mean, Z_mean, consensus_key, majority_vote

# ----------------------------------------
# Plot Posterior Distributions
# ----------------------------------------
def plot_posteriors(trace):
    az.plot_posterior(trace, var_names=["D"], hdi_prob=0.94)
    plt.suptitle("Posterior of Informant Competence (D)", fontsize=14)
    plt.tight_layout()
    plt.show()

    az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.94)
    plt.suptitle("Posterior of Consensus Answers (Z)", fontsize=14)
    plt.tight_layout()
    plt.show()

# ----------------------------------------
# Run everything
# ----------------------------------------
if __name__ == "__main__":
    responses, informants = load_plant_knowledge_data()
    trace = build_and_sample_model(responses)
    analyze_results(trace, responses, informants)
    plot_posteriors(trace)