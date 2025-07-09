#%%

from mimosa import MIMOSA, load_params
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def run_mimosa(carbon_budget, welfare, weights=None):

    params = load_params()

    params["model"]["welfare module"] = welfare
    params["emissions"]["carbonbudget"] = "775 GtCO2"

    if weights is not None:
        params["region_weights"] = weights

    params["time"]["end"] = 2100

    params["emissions"]["global min level"] = "-100 GtCO2/yr"
    params["emissions"]["regional min level"] = "-50 GtCO2/yr"

    model = MIMOSA(params)
    model.solve()

    savepath = f"run_{welfare}_carbonbudget_{carbon_budget.replace(' ', '_')}"

    model.save(savepath)

    return model, savepath

def plot_regional_budgets(savepath, r5=True):
    
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
        ax.set_ylim(-50, 400)
    else:
        ax.bar(list(cumulative_emissions.Region), cumulative_emissions.iloc[:, -2], color='teal')
        ax.set_xticklabels(cumulative_emissions['Region'])
        ax.set_ylim(-30, 130)
        
    return r5_cumulative.iloc[:, -1].values if r5 else cumulative_emissions.iloc[:, -2].values

#%%

welfare = "weighted_welfare_loss_minimising"
carbon_budget = "775 GtCO2"

r5_weights = {
    'R5_OECD': 0.5,
    'R5_LAM': 5.0,
    'R5_REF': 1.0,
    'R5_MAF': 1.0,
    'R5_ASIA': 1.0
}

region_weights = {k: r5_weights[v] for k, v in r5_map.items()}

model, savepath = run_mimosa(carbon_budget, welfare, weights=region_weights)

#%%

cumulative = plot_regional_budgets(savepath, r5=True)

# %%
