from language import EsmFoldv1

folding_callback = EsmFoldv1()
folding_callback.load(device="cuda")

from language import FixedLengthSequenceSegment
from language import ProgramNode
from language import run_nested_sampling
import wandb

iterations = 10000
num_live_points = 50

wandb.init(project="NS-benchmark", name=f"ns_iter{iterations}_nlive{num_live_points}")


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

run_nested_sampling(
    program=program,
    N=3,
    length=25,
    total_num_steps=iterations,
    num_live_points=num_live_points,
    folding_callback=folding_callback,
    display_progress=False,
    progress_verbose_print=True
)
