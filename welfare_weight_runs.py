#%%

from mimosa import MIMOSA, load_params
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize

regions = ['CAN', 'USA', 'MEX', 'RCAM', 'BRA', 'RSAM', 'NAF', 'WAF', 'EAF', 'SAF', 'WEU', 'CEU', 'TUR', 'UKR', 'STAN', 'RUS', 'ME', 'INDIA', 'KOR', 'CHN', 'SEAS', 'INDO', 'JAP', 'OCE', 'RSAS', 'RSAF']

r5_oecd = ['TUR', 'CAN', 'USA', 'WEU', 'OCE', 'CEU']

r5_lam = ['MEX', 'BRA', 'RCAM', 'RSAM']

r5_eena = ['UKR', 'STAN', 'RUS']

r5_maf = ['NAF', 'WAF', 'EAF', 'SAF', 'ME', 'RSAF']

r5_asia = ['INDIA', 'KOR', 'CHN', 'SEAS', 'INDO', 'JAP', 'RSAS']

r5_map = {}
for region in r5_oecd:
    r5_map[region] = 'R5_OECD'
for region in r5_lam:
    r5_map[region] = 'R5_LAM'
for region in r5_eena:
    r5_map[region] = 'R5_REF'
for region in r5_maf:
    r5_map[region] = 'R5_MAF'
for region in r5_asia:
    r5_map[region] = 'R5_ASIA'

#%% functions

def plot_vars(data_list, label_list):

    fig, axs = plt.subplots(3, 9, figsize=(15, 10), sharex=True, sharey=True)

    for i, region in enumerate(data_list[0].Region.unique()):
        ax = axs[i // 9, i % 9]
        region_data = [data[data.Region == region].iloc[:, 3:] for data in data_list]
        for i in range(len(data_list)):
            ax.plot(np.arange(2020, 2155, 5), region_data[i].values.flatten(), label=label_list[i], color='blue' if i == 0 else 'purple')
        ax.set_title(region)
        ax.set_xlabel('Year')
        ax.set_ylabel('Damages (Fraction of GDP)')
        if i == 25:
            ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)        

def run_mimosa(carbon_budget, welfare, weights=None, elasmu=1.01):

    params = load_params()

    params["model"]["welfare module"] = welfare
    params["economics"]["elasmu"] = elasmu
    params["emissions"]["carbonbudget"] = carbon_budget

    if weights is not None:
        params["region_weights"] = weights

    params["time"]["end"] = 2100

    params["emissions"]["global min level"] = "-100 GtCO2/yr"
    params["emissions"]["regional min level"] = "-50 GtCO2/yr"

    model = MIMOSA(params)
    model.solve()

    savepath = f"run_{welfare}_carbonbudget_{carbon_budget.replace(' ', '_')}_elasmu_{elasmu:.2f}"

    model.save(savepath)

    return model, savepath

def plot_regional_budgets(savepath, r5=True, show_plots=True):
    
    emissions_df = pd.read_csv("./output/"+savepath+".csv")
    emissions_df = emissions_df.loc[emissions_df.Variable=='regional_emissions']

    cumulative_emissions = pd.concat(
        [
            emissions_df.iloc[:, 1:3],
            emissions_df.iloc[:, 3:].apply(
                lambda row: np.insert(
                    np.cumsum((row.values[:-1] + row.values[1:]) / 2 * 5), 0, 0
                )[-1],
                axis=1
            )
        ],
        axis=1
    )

    cumulative_emissions['R5'] = cumulative_emissions['Region'].map(r5_map)

    value_col = cumulative_emissions.columns[-2]  # usually the last column before 'R5'
    r5_cumulative = cumulative_emissions.groupby('R5', as_index=False)[value_col].sum()

    if not show_plots == False:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title('Very unequal distribution of remaining carbon budget → unequitable?')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel('Remaining carbon budget until 2100 (GtCO2)')
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if r5:
            ax.bar(list(r5_cumulative.R5), r5_cumulative.iloc[:, 1], color='teal')
            ax.set_xticklabels(r5_cumulative['R5'])
            ax.set_ylim(-150, 450)
        else:
            ax.bar(list(cumulative_emissions.Region), cumulative_emissions.iloc[:, -2], color='teal')
            ax.set_xticklabels(cumulative_emissions['Region'])
            ax.set_ylim(-30, 130)
        
    return r5_cumulative.iloc[:, -1].values if r5 else cumulative_emissions.iloc[:, -2].values

#%%

welfare = "weighted_welfare_loss_minimising"
carbon_budget = "775 GtCO2"
elasmu = 2

r5_weights = {
    'R5_OECD': 1.52,
    'R5_LAM': 0.55,
    'R5_REF': 0.51,
    'R5_MAF': 0.58,
    'R5_ASIA': 0.99
}

region_weights = {k: r5_weights[v] for k, v in r5_map.items()}

model, savepath = run_mimosa(carbon_budget, welfare, weights=region_weights, elasmu=elasmu)

#%%

cumulative = plot_regional_budgets(savepath, r5=True)

# %%
# minimise distance to scenario

# scenario_budget = np.array([408.762635, -68.14466, 155.302528, 228.120371, 50.5201793,]) # message C3
# scenario_budget = np.array([331.4506577, -116.585561, 64.33625214, -2.924358264, 62.99636074]) # image C2
# scenario_budget = np.array([241.3040656, -143.9911212, 81.8121524, 181.8422117, 31.40160669]) # message C2
scenario_budget = np.array([406.4454569, 9.4014275, 139.8349114, 137.5121418, 9.7702623]) # remind C2

scenario = "remind_C2" # "image_C2" # message_C2 # remind_C2
# carbon_budget = "339 GtCO2" # image C2
# carbon_budget = "392 GtCO2" # message C2
carbon_budget = "703 GtCO2" # remind C2
tol = 1e-1
opt_elasmu = False

def objective(weights, elasmu=1.01):
    # Map weights to R5 groups
    r5_keys = list(r5_weights.keys())
    test_r5_weights = dict(zip(r5_keys, weights[1:])) if len(weights)==6 else dict(zip(r5_keys, weights))
    region_weights = {k: test_r5_weights[v] for k, v in r5_map.items()}
    # Run model
    model, savepath = run_mimosa(carbon_budget, welfare, weights=region_weights, elasmu=weights[0] if len(weights)==6 else elasmu)
    # Get cumulative budgets for R5 regions
    cumulative_arr = np.array(plot_regional_budgets(savepath, r5=True, show_plots=False))
    if len(weights)==6:
        print("Trying weights:", weights[1:], "Objective:", np.sum((cumulative_arr - scenario_budget) ** 2))
        print("elasmu:", weights[0])
    else:
        print("Trying weights:", weights, "Objective:", np.sum((cumulative_arr - scenario_budget) ** 2))
    return np.sum((cumulative_arr - scenario_budget) ** 2)

if opt_elasmu:
    initial_guess = [1.01, 1.0, 1.0, 1.0, 1.0, 1.0]
    bounds = [(0.2, 5.0)] * 6
else: 
    # Initial guess for the 5 weights (same order as r5_weights)
    initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]  # Adjust as needed
    bounds = [(0.1, 5.0)] * 5

# Run the optimizer
result = minimize(objective, initial_guess, bounds=bounds, tol=tol, method="Powell")

print("Optimized R5 weights:", result.x[1:])
if opt_elasmu:
    print("Optimized elasmu:", result.x[0])
print("Objective function value:", result.fun)

if opt_elasmu:
    # Use the optimized weights to run the final model with elasmu
    opt_r5_weights = dict(zip(r5_weights.keys(), result.x[1:]))
    elasmu = result.x[0]
else:
    # Use the optimized weights to run the final model
    opt_r5_weights = dict(zip(r5_weights.keys(), result.x))

region_weights = {k: opt_r5_weights[v] for k, v in r5_map.items()}

if opt_elasmu:
    model, savepath = run_mimosa(carbon_budget, welfare, weights=region_weights, elasmu=elasmu)
else:
    model, savepath = run_mimosa(carbon_budget, welfare, weights=region_weights)
cumulative = plot_regional_budgets(savepath, r5=True)
print("Final cumulative budgets:", cumulative)
print("Scenario budgets:", scenario_budget)
print("Difference:", np.array(cumulative) - scenario_budget)

# %%
# save opt_r5_weights to csv
opt_r5_weights_df = pd.DataFrame.from_dict(opt_r5_weights, orient='index', columns=['Weight'])
opt_r5_weights_df.index.name = 'Region'
opt_r5_weights_df.to_csv(f'opt_r5_weights_{scenario}.csv')

# save elasmu to csv
if opt_elasmu:
    elasmu_df = pd.DataFrame({'elasmu': [elasmu]})
    elasmu_df.to_csv(f'opt_elasmu_{scenario}.csv', index=False)


# %%
