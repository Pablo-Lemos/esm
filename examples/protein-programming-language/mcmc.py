from language import EsmFoldv1

folding_callback = EsmFoldv1()
folding_callback.load(device="cuda:0")

from language import FixedLengthSequenceSegment
from language import ProgramNode
import wandb

wandb.init(project="NS-benchmark", name="mcmc")

protomer = FixedLengthSequenceSegment(25)
def make_protomer_node():
    return ProgramNode(sequence_segment=protomer)

N = 3
_node = ProgramNode(
    children=[make_protomer_node() for _ in range(N)],
)

from language import MaximizePTM, MaximizePLDDT, SymmetryRing

# Define the program.
program = ProgramNode(
    energy_function_terms=[MaximizePTM(), MaximizePLDDT(), SymmetryRing()],
    energy_function_weights=[1.0, 1.0, 1.0],
    children=[make_protomer_node() for _ in range(N)],
)

# Set up the program.
sequence, residue_indices = program.get_sequence_and_set_residue_index_ranges()

# Compute and print the energy function.
energy_terms = program.get_energy_term_functions()
folding_output = folding_callback.fold(sequence, residue_indices)
for name, weight, energy_fn in energy_terms:
    print(f"{name} = {weight:.1f} * {energy_fn(folding_output):.2f}")

from language import run_simulated_annealing

optimized_program = run_simulated_annealing(
    program=program,
    initial_temperature=1.0,
    annealing_rate=0.97,
    total_num_steps=10_000,
    folding_callback=folding_callback,
    display_progress=True,
)
print("Final sequence = {}".format(optimized_program.get_sequence_and_set_residue_index_ranges()[0]))
