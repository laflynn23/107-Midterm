import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

def load_plant_knowledge_data(path="data/plant_knowledge.csv"):
    df = pd.read_csv(path)
    informants = df["Informant"].values
    responses = df.drop(columns=["Informant"]).values.astype(int)
    return responses, informants

def build_and_sample_model(responses, draws=2000, tune=1000, chains=4):
    N, M = responses.shape

    with pm.Model() as cct_model:
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        D_reshaped = D[:, None]
        Z_reshaped = Z[None, :]
        p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)

        X = pm.Bernoulli("X", p=p, observed=responses)

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, return_inferencedata=True, random_seed=42)

    return trace

def analyze_results(trace, responses, informants):
    D_mean = trace.posterior["D"].mean(dim=("chain", "draw")).values
    Z_mean = trace.posterior["Z"].mean(dim=("chain", "draw")).values
    consensus_key = np.round(Z_mean).astype(int)
    majority_vote = np.round(responses.mean(axis=0)).astype(int)

    print("=== Informant Competence Estimates ===")
    most = np.argmax(D_mean)
    least = np.argmin(D_mean)
    for i, name in enumerate(informants):
        marker = " <== MOST" if i == most else " <== LEAST" if i == least else ""
        print(f"{name}: {D_mean[i]:.3f}{marker}")

    print("\n=== Posterior Consensus Answer Key (Z) ===")
    print(" ".join(map(str, consensus_key)))

    print("\n=== Naive Majority Vote ===")
    print(" ".join(map(str, majority_vote)))

    print("\n=== Differences (Consensus ≠ Majority) ===")
    print(" ".join(["\u2714" if a != b else " " for a, b in zip(consensus_key, majority_vote)]))

    summary = az.summary(trace, var_names=["D", "Z"])
    print("\n=== R-hat Summary ===")
    print(summary[["mean", "hdi_3%", "hdi_97%", "r_hat"]])

    return D_mean, Z_mean, consensus_key, majority_vote

def plot_posteriors(trace):
    az.plot_posterior(trace, var_names=["D"], hdi_prob=0.94)
    plt.suptitle("Posterior of Informant Competence (D)", fontsize=14)
    plt.tight_layout()
    plt.show()

    az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.94)
    plt.suptitle("Posterior of Consensus Answers (Z)", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    responses, informants = load_plant_knowledge_data()
    trace = build_and_sample_model(responses)
    analyze_results(trace, responses, informants)
    plot_posteriors(trace)
