# %%
import jax
jax.config.update("jax_enable_x64", False)
print("JAX CONFIG ENABLE X64 ",jax.config.jax_enable_x64)  # Should print True
# %%
# %%
# %%
import sys
sys.path.append('/leonardo_work/IscrC_MiTheGe1/MWWATER/FLOWS/flow_diagrams/') 
import flow_diagrams

# %%
import jax
import time
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
from IPython.display import clear_output
from jax import Array
# from openmm import unit
import optax
from flow_diagrams.utils.conditioning import convert_from_reduced_p, convert_from_reduced_t
from matplotlib import colors
import pickle



# %%
jax.devices()
from jax_md import space, partition
import argparse
# %%
from flow_diagrams.utils.train import log_weights_given_latent, normalize_weights, sampling_efficiency, effective_sample_size, delta_f_to_prior
from flow_diagrams.utils.visualization import radial_distribution_function
from flow_diagrams.utils.data import NumpyLoader, split_data
from flow_diagrams.utils.symmetry import *

from jax import numpy as jnp

#from flow_diagrams.energy.lennard_jones import fd_lennard_jones_neighbor_list
from jax_md.energy import stillinger_weber_neighbor_list

from flow_diagrams.models.coupling_flows import ConditionalCouplingFlowNPT

from flow_diagrams.utils.train import running_average
# from flow_diagrams.utils.lattice import volume_to_box
from flow_diagrams.utils.weights import get_weights, get_biases
from flow_diagrams.utils.jax import key_chain
from IPython.display import clear_output
from flow_diagrams.utils.conditioning import grid_conditional_variables
import time


# %%
chain = key_chain(1)

# %%
EPSILON_cal   = 6.189 #in units kcal/mol
EPSILON = EPSILON_cal / 0.239005736 #in units kJ/mol
SIGMA_ang = 2.3925 #in units Angstroms
Sigma_nm = SIGMA_ang * 1e-1 #in units nm
SIGMA = Sigma_nm #in units nm
a         = 1.80 #dimensionless
lam       = 23.15 #dimensionless
gamma     = 1.20 #dimensionless
A         = 7.049556277 #dimensionless
B         = 0.6022245584 #dimensionless
p         = 4. #dimensionless
q         = 0. #dimensionless
#theta_0   = np.radians(109.47)
NUM_PARTICLES = 180
NUM_SAMPLES = 20000
SPATIAL_DIMENSIONS = 3
kB = 0.00831446261815324 # in (unit.kilojoule_per_mole/unit.kelvin)
kB_cal = kB * 0.239005736
CUTOFF= a * SIGMA

# %%
def remove_disp_of_first_atom(displacements):
    # assert displacements.shape == (NUM_PARTICLES, SPATIAL_DIMENSIONS)

    disp_at_1 = displacements[0,:]

    return displacements - disp_at_1


def transform_abs_coords_to_rel_coords(absolute_coordinates: Array, side_length: Array):
    """Transforms relative coordinates inside the unit cube to absolute coordinates given a 3d box_vector."""
    assert absolute_coordinates.shape[-1] == SPATIAL_DIMENSIONS
    assert side_length.shape == (3,)
    return absolute_coordinates / side_length

# %%
def wrap_to_unit_cube(pos, lower, upper):
    width = upper - lower
    return jnp.mod(pos - lower, width) + lower

def wrap_to_box(pos, box):
    return pos % box

# %%
jax.config.update("jax_enable_x64", False) #changed to True when running on GPU for 120 particle system


# %%
LOWER = 0.
UPPER = 1.
CUT_TYPE = 'switch'

Press_atm = 1.0 #in atm
Press_1e30_Pa_per_mol = 1.01325 * 6.022 * 1e-2 * Press_atm
PRIOR_PRESSURE = Press_1e30_Pa_per_mol # in units sigma_nm**3 / epsilon_kJ/mol = 1e3 Pa/mol
TEMP_PRIOR  = int(args.temp)
md_seed = 301098
TEMP_INT = int(TEMP_PRIOR)

REDUCED_TEMP_PRIOR = TEMP_PRIOR / convert_from_reduced_t(EPSILON, kB)
REDUCED_PRESS_PRIOR = PRIOR_PRESSURE / convert_from_reduced_p(EPSILON, SIGMA)

#[T]mW = 3114.4 K, and [p]mW = 31 400 bars.

filename_path = "/leonardo_work/IscrC_MWWATER/MWWATER/FLOWS/T_"+str(TEMP_INT)+"/DeltaP_0.08_80_20_180_VK/"
filename_prior = filename_path + "1000_20000000_sim_dump_T_"+str(TEMP_INT)+"_P_0.0_N_180_Ns_20000_seed_63782_pbc_tonpz.npz"

data_prior = jnp.load(filename_prior)

positions_prior_abs = data_prior['pos'] * 1e-1 #in units nm
box_prior = data_prior['box'] * 1e-1 #in units nm
vols_prior = jnp.prod(box_prior, axis=-1)
BOX_EDGES = np.mean(box_prior, axis=0)

# fix first atom in origin and wrap to box
positions_prior = jax.vmap(wrap_to_box)(jax.vmap(remove_disp_of_first_atom)(positions_prior_abs),box_prior)
MEAN_CONFIG = np.mean(positions_prior,axis=0)

# scale to [0,1]
positions_prior= jax.vmap(transform_abs_coords_to_rel_coords)(positions_prior,box_prior)
positions_prior = wrap_to_unit_cube(positions_prior,LOWER,UPPER)    

scale_prior = box_prior[:,0] / BOX_EDGES[0]
energies_prior = data_prior['ene'] / 0.239005736 #in units kJ/mol

assert np.logical_and(1. >= UPPER, positions_prior >= LOWER).all()
assert np.allclose(positions_prior[:,0,:],0,atol=1e-7)

n_samples = positions_prior.shape[0] 

print('# Prior samples', n_samples)

# %%
BATCH_SIZE = 32

# %%
train_fraction = 0.1
# Store all displacements relative to first one (which stays at its equilibrium position)
dataset_prior_train, dataset_prior_test = split_data(train_fraction, positions_prior,
                        energies_prior,
                       scale_prior)
dataloader_train = NumpyLoader(dataset_prior_train,BATCH_SIZE,False)



# %%
len(dataset_prior_train), len(dataset_prior_test)

# %%
dtype = np.float32
format = partition.Dense

# %%
displacement_frac, shift_frac = space.periodic_general(BOX_EDGES, fractional_coordinates=False)

neighbor_fn, energy_fn = stillinger_weber_neighbor_list(
    displacement=displacement_frac,
    box_size=BOX_EDGES,
    sigma=SIGMA,
    A = A,
    B = B,
    lam = lam,
    gamma = gamma,
    epsilon= EPSILON,
    cutoff = CUTOFF,
    dr_threshold= 5, #LARGE threshold to match lammps calculations
    fractional_coordinates=False,
    format = format
    ) 

NEIGHBOR_LIST = neighbor_fn.allocate(MEAN_CONFIG)

# %%
def compute_sw_energy(pos_rel: jnp.ndarray, scale):
    box= scale * BOX_EDGES
    nbrs = NEIGHBOR_LIST.update(pos_rel * box)
    sw_energy = energy_fn(pos_rel * box, nbrs, box=box)

    return sw_energy


# %%
print(REDUCED_TEMP_PRIOR, REDUCED_PRESS_PRIOR, REDUCED_TEMP_PRIOR * convert_from_reduced_t(EPSILON, kB), REDUCED_PRESS_PRIOR * convert_from_reduced_p(EPSILON, SIGMA))

# %%
INTERVAL_FRACTION_P = 0.05 #0.05 -> 0.2 -> 0.8 -> 0.1


p_max = INTERVAL_FRACTION_P    * convert_from_reduced_p(EPSILON, SIGMA)
p_min = - INTERVAL_FRACTION_P  * convert_from_reduced_p(EPSILON, SIGMA) 

t_max = 0.1 * convert_from_reduced_t(EPSILON, kB)
t_min = 0.06 * convert_from_reduced_t(EPSILON, kB)

grid_length = 20
conditioning_states= grid_conditional_variables(t_min,t_max,p_min, p_max, grid_length,grid_length)

# %%
assert conditioning_states[0,0] == t_min
assert conditioning_states[0,1] == p_min

assert conditioning_states[-1,0] == t_max
assert conditioning_states[-1,1] == p_max

# %%
print("\nPrior temperature = ", TEMP_PRIOR, "K, \nPrior pressure = ", PRIOR_PRESSURE, "10^30 Pa/mol, or ", PRIOR_PRESSURE/ (1.01325 * 6.022 * 1e-2) , "atm")
print("\nReduced temperature = ", REDUCED_TEMP_PRIOR, "\nReduced pressure = ", REDUCED_PRESS_PRIOR, "\n\n")

print("t_min = ", t_min, " K")
print("reduced t_min = ", t_min / convert_from_reduced_t(EPSILON, SIGMA))
print("t_max = ", t_max, " K")
print("reduced t_max = ", t_max / convert_from_reduced_t(EPSILON, SIGMA))

print("p_min = ", p_min, " 10^30 Pa/mol, or ", p_min/ (1.01325 * 6.022 * 1e-2), " atm")
print("reduced p_min = ", p_min /convert_from_reduced_p(EPSILON, SIGMA))
print("p_max = ", p_max, " 10^30 Pa/mol, or ", p_max/ (1.01325 * 6.022 * 1e-2), " atm")

# %%
identity_initialization = True

flow = ConditionalCouplingFlowNPT(n_layers=1,
                            num_hidden=2,
                            dim_hidden=32,
                            num_hidden_shape=4,
                            dim_hidden_shape=16,
                            dim_embedd =32,
                            lower=0,
                            upper=1,
                            n_bins=16,
                            n_heads=1,
                            t_max=t_max,
                            p_max=p_max,
                            use_layer_norm=True,
                            n_blocks=1,
                            use_circular_shift=True,
                            n_freqs=8,
                            init_identity=identity_initialization,
                            n_particles=NUM_PARTICLES ,
                            key= next(chain))

params, static = eqx.partition(flow, eqx.is_array)

param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print("param_count: ", f"{param_count:_}", "\n")

# %%
learning_rate = 1e-5
optim = optax.adam(learning_rate) 
optim = optax.chain(optax.clip_by_global_norm(1e4), optim)

# %%
# %%
@eqx.filter_jit
def evaluate_flow(flow,pos,scale,press,temp):
    return flow(pos=pos,scale=scale,press=press,temp=temp)


def calculate_new_pos_and_scale_batch(batch_pos, batch_scale, p_rand, t_rand, flow):

    def single_flow(pos, scale, press, temp):
        return flow.forward(pos, scale, press=press, temp=temp)[0:3]

    return jax.vmap(single_flow)(batch_pos, batch_scale, p_rand, t_rand)


# %%

# === 1. Carica lo stato del training (pickle) ===
with open(filename_path + "training_state.pkl", "rb") as f:
    training_state = pickle.load(f)

print("Training state keys:", training_state.keys())  # se è un dizionario

# === 2. Carica il modello Equinox ===
flow = eqx.tree_deserialise_leaves(filename_path + "flow.eqx", like=flow)

# === 3. Carica lo stato dell’ottimizzatore ===
opt_state = eqx.tree_deserialise_leaves(filename_path + "opt_state.eqx", like=optim)

print("Flow model loaded:", type(flow))
print("Optimizer state loaded:", type(opt_state))

# %%
def log_weights_and_conf_given_latent(
    pos_prior,
    scale_prior,
    prior_energy,
    temp_and_pressure_target,
    temp_and_pressure_flow,
    reference_box,
    n_particles,
    pressure_prior,
    temp_prior,
    target_energy_fn,
    flow,
):
    """Computes the weights for one sample.

    params:
    --------------------------
    logw: unnormalized weights"""

    prior_vol = jnp.prod(scale_prior * reference_box)

    target_temp = temp_and_pressure_target[0]
    target_press = temp_and_pressure_target[1]

    flow_temp = temp_and_pressure_flow[0]
    flow_press = temp_and_pressure_flow[1]

    new_pos, new_scale, ldj = flow.forward(
        pos=pos_prior, scale=scale_prior, temp=flow_temp, press=flow_press
    )
    new_box = new_scale * reference_box
    new_vol = jnp.prod(new_box)

    target_energy = target_energy_fn(new_pos, new_scale)

    ldj_initial = n_particles * (jnp.log(1.0 / prior_vol))
    ldj_final = n_particles * (jnp.log(new_vol))

    ldj += ldj_final + ldj_initial
    logw = (
        -(target_energy + new_vol * target_press) / (kB * target_temp)
        + ldj
        + (prior_energy + prior_vol * pressure_prior) / (kB * temp_prior)
    )

    return logw, new_pos, new_scale, target_energy


# %%
results = {}  # qui salviamo i dati prima di scrivere il file

for (target_temp, target_press) in conditioning_states:
    print(f"target_temp: {target_temp}, target_press: {target_press}")
    all_energies = []
    all_boxes = []
    all_weights = []
    all_positions = []   # <-- aggiunto

    # Genera batch e calcola energie/volumi
    for ibatch, (batch_pos, batch_ene, batch_scale) in enumerate(dataloader_train):
        
        t_batch = jnp.full((batch_pos.shape[0],), target_temp) 
        p_batch = jnp.full((batch_pos.shape[0],), target_press)
        
        logw, new_pos, new_scale, energy_batch = jax.vmap(lambda pos, ene, scal: log_weights_and_conf_given_latent(
            pos_prior=pos,
            prior_energy=ene,
            scale_prior=scal,
            flow=flow,
            temp_and_pressure_target=jnp.array([target_temp, target_press]),
            temp_and_pressure_flow=jnp.array([target_temp, target_press]),
            n_particles=NUM_PARTICLES,
            pressure_prior=PRIOR_PRESSURE,
            temp_prior=TEMP_PRIOR,
            reference_box=BOX_EDGES,
            target_energy_fn=compute_sw_energy,
        ))(batch_pos, batch_ene, batch_scale)

        box_batch = jax.device_get(new_scale)[:, jnp.newaxis] * BOX_EDGES

        all_energies.append(energy_batch)
        all_boxes.append(box_batch)
        all_weights.append(logw)
        all_positions.append(jax.device_get(new_pos))   # <-- aggiunto

    # concatena
    new_energies = np.concatenate(all_energies)[:n_samples]
    new_boxes = np.concatenate(all_boxes)[:n_samples]
    new_vol = np.prod(new_boxes, axis=1)
    new_weights = np.concatenate(all_weights)[:n_samples]
    new_positions = np.concatenate(all_positions)[:n_samples]  # <-- aggiunto

    key_prefix = f"{target_temp}_{target_press}"
    results[f"{key_prefix}_energy"] = new_energies
    results[f"{key_prefix}_volume"] = new_vol
    results[f"{key_prefix}_weight"] = new_weights
    results[f"{key_prefix}_positions"] = new_positions   # <-- aggiunto


# === Salva in un unico file npz ===
output_file = "generated_data_" + str(TEMP_INT)+ "_" + str(len(dataset_prior_train)) +".npz"
np.savez(output_file, **results)
print("Salvato in" + output_file)

# %%


# %%



