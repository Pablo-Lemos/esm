# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from rich.live import Live
from rich.table import Table

from language.sequence import FixedLengthSequenceSegment
from language.folding_callbacks import FoldingCallback
from language.program import ProgramNode

import wandb 


@dataclass
class SingleState:
    program: ProgramNode
    energy: float
    energy_term_fn_values: list

@dataclass
class NestedSamplingState:
    live_states: list
    dead_states: list
    #live_energies: list
    num_steps: int


def get_energy(
        candidate: ProgramNode,
        folding_callback: FoldingCallback
) -> (float, list):
    sequence, residue_indices = candidate.get_sequence_and_set_residue_index_ranges()
    folding_output = folding_callback.fold(sequence, residue_indices)

    energy_term_fns = candidate.get_energy_term_functions()
    candidate_energy_term_fn_values = [
        (name, weight, energy_fn(folding_output)) for name, weight, energy_fn in energy_term_fns
    ]

    candidate_energy: float = sum(
        [weight * value for _, weight, value in candidate_energy_term_fn_values]
    )

    # Taking negative energy, as we want to minimize
    candidate_energy = - candidate_energy
    return sequence, candidate_energy, candidate_energy_term_fn_values


def nested_sampling_step(
    state: NestedSamplingState,
    folding_callback: FoldingCallback,
    verbose: bool = False,
) -> SingleState:

    # Find the lowest energy state
    lowest_state: SingleState = min(state.live_states, key=lambda x: x.energy)

    # Randomly select a state to mutate
    state_to_mutate: SingleState = np.random.choice(state.live_states)

    min_energy = lowest_state.energy
    candidate_energy = - np.infty
    while candidate_energy < min_energy:
        candidate: ProgramNode = deepcopy(state_to_mutate.program)
        candidate.mutate()

        sequence, candidate_energy, candidate_energy_term_fn_values = get_energy(candidate, folding_callback)
        state.num_steps += 1

        dict_log = {"num_steps": state.num_steps,
                    "sequence": sequence,
                    "energy": candidate_energy,
                    "accept_candidate": candidate_energy < min_energy}

        for name, weight, value in candidate_energy_term_fn_values:
            dict_log[name] = value
        wandb.log(dict_log)

    if verbose:
        print(f"{state.num_steps}, {candidate_energy:.4f}")
        #print(f"Accepted {sequence} with energy {candidate_energy:.2f}.")

    # Replace the lowest energy state with the candidate
    state.dead_states.append(lowest_state)
    state.live_states.remove(lowest_state)
    state.live_states.append(SingleState(program=candidate,
                                         energy=candidate_energy,
                                         energy_term_fn_values=candidate_energy_term_fn_values))

    return state


def make_protomer_node(protomer):
    return ProgramNode(sequence_segment=protomer)


def run_nested_sampling(
    program: ProgramNode,
    N: int,
    length: int,
    num_live_points: int,
    total_num_steps: int,
    folding_callback: FoldingCallback,
    display_progress: bool = True,
    progress_verbose_print: bool = False,
) -> ProgramNode:
    # TODO(scandido): Track accept rate.

    if progress_verbose_print:
        print("Starting nested sampling...")
        print(f"Number of live points: {num_live_points}")
        print(f"Total number of steps: {total_num_steps}")

    live_states = []
    for i in range(num_live_points):
        protomer = FixedLengthSequenceSegment(length)
        program = ProgramNode(
            energy_function_terms=program.energy_function_terms,
            energy_function_weights=program.energy_function_weights,
            children=[ProgramNode(sequence_segment=protomer) for _ in range(N)],
        )
        sequence, energy, energy_term_fn_values = get_energy(program, folding_callback)
        live_states.append(SingleState(program=program, energy=energy, energy_term_fn_values=energy_term_fn_values))

        dict_log = {"num_steps": i,
                    "sequence": sequence,
                    "energy": energy,
                    "accept_candidate": True}

        for name, weight, value in energy_term_fn_values:
            dict_log[name] = value
        wandb.log(dict_log)

    state = NestedSamplingState(live_states=live_states, dead_states=[], num_steps=num_live_points)

    if progress_verbose_print:
        print("Generated initial live points.")

    while state.num_steps < total_num_steps:
        state = nested_sampling_step(
            state,
            folding_callback,
            verbose=progress_verbose_print,
        )

    # with Live() as live:
    #     for _ in range(1, total_num_steps + 1):
    #         state = metropolis_hastings_step(
    #             state,
    #             folding_callback,
    #             verbose=progress_verbose_print,
    #         )
    #         if display_progress:
    #             live.update(_generate_table(state))

    return state

