from mimosa import MIMOSA, load_params

params = load_params()

params["economics"]["damages"]["ignore damages"] = True
params["economics"]["damages"]["scale factor"] = 0  # (1)!
params["time"]["end"] = 2150

model = MIMOSA(params)

simulation = model.run_nopolicy_baseline()

model.save_simulation(simulation, "baseline_ignore_damages")
