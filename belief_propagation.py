import networkx as nx
import random
from linearcodes import LinearCode

class MessagePassingTree:
    """
    Implements a message passing algorithm on a NetworkX tree.
    Algorithm sends integer messages, adding them at each iteration.
    
    Args:
        - num_vertices [int]: Number of vertices in the tree.
        - initial_vertex_values [list[int]]: Optional list of initial values for each vertex.
            If not given, then defaults to 1 for all vertices.
    """
    def __init__(self,
                 graph: nx.Graph,
                 initial_vertex_values: list[int] | None = None):
        self.graph = graph
        self.num_vertices = len(graph.nodes)  

        if initial_vertex_values is None:
            initial_vertex_values = [1]*self.num_vertices
        if len(initial_vertex_values) != self.num_vertices:
            raise ValueError("Length of initial_vertex_values must match num_vertices.")
        
        self.neighbours = [list(self.graph.neighbors(u)) for u in self.graph.nodes]
        self.initial_vertex_values = initial_vertex_values
        
        self.directed_edges = [(u, v) for u in self.graph.nodes for v in self.graph.neighbors(u)]
        
    @property
    def initial_messages(self) -> dict[tuple[int,int],int]:
        """
        Creates list of initial messages.
        The message from a vertex u to a neighbour v is the inital value of u.
        """
        return {edge : self.initial_vertex_values[edge[0]] for edge in self.directed_edges}

    def run(self, max_interations: int = 100, verbose: bool = False) -> tuple[int,list[int]]:
        """
        Run message passing until convergence or max_iterations reached.
        
        Args:
            - max_iterations [int]: Maximum number of iterations of the algorithm.
            - verbose [bool]: If True, prints messages and vertex values at each iteration.
        
        Returns:
            list[int]: The final vertex values after message passing.
        """
        vertex_values = self.initial_vertex_values
        messages = self.initial_messages.copy()
        for iteration in range(max_interations):
            vertex_values = self._update_vertex_values(vertex_values, messages)
            
            new_messages = self._update_messages(messages)
            if all(value == 0 for value in new_messages.values()):
                    if verbose:
                        print(f"Converged at iteration {iteration+1}.")
                    return (iteration, vertex_values)
            messages = new_messages

            if verbose:
                print(f"\nIteration {iteration+1}")
                print(f"Messages: {messages}")
                print(f"Vertex values: {vertex_values}")

        return (iteration,vertex_values)

    def _update_messages(self,
                        messages: dict[tuple[int,int], int]) -> dict[tuple[int,int], int]:
        """
        Compute new messages based on incoming messages.
        """
        new_messages = {edge : 0 for edge in self.directed_edges}
        for u, v in self.directed_edges:
            new_message = 0
            for w in self.neighbours[u]:
                if w != v:
                    try:
                        new_message += messages[(w,u)]
                    except:
                        pass
            new_messages[(u,v)] = new_message
        return new_messages
    
    def _update_vertex_values(self,
                              vertex_values: list[int],
                              messages: dict[tuple[int,int], int]) -> list[int]:
        """
        Update vertex values based on the messages received from neighbours.
        """
        new_vertex_values = vertex_values.copy()
        for u, v in self.directed_edges:
            new_vertex_values[v] += messages[(u,v)]
        return new_vertex_values

def find_number_vertices_in_random_tree(max_num_vertices: int = 50, num_iterations: int = 100):
    num_vertices = random.choice(range(max_num_vertices))
    graph = nx.generators.random_labeled_tree(num_vertices)
    mp = MessagePassing(graph)
    final_info = mp.run(num_iterations)
    return (final_info[0], final_info[1][0])        
    

class BeliefPropagation:

    """
    Edge set must only contain ordered edges.
    The check vertices will be the sources of the edges
    and the bit vertices will be the targets of the edges.
    """
    def __init__(self,
                 code: LinearCode):
        self.code: LinearCode = code
        self.graph: nx.Graph = self.code.tanner_graph        
        
        self.check_vertices: list[int] = [
            node for node, data in self.graph.nodes(data=True) if data.get('bipartite') == 'check'
            ]
        self.bit_vertices: list[int] = [
            node for node, data in self.graph.nodes(data=True) if data.get('bipartite') == 'bit'
            ]
        self.check_neighbourhood: dict[int,list[int]] = {
            check_vertex : list(self.graph.neighbors(check_vertex)) for check_vertex in self.check_vertices
            }
        self.bit_neighbourhood: dict[int,list[int]] = {
            bit_vertex : list(self.graph.neighbors(bit_vertex)) for bit_vertex in self.bit_vertices
            }

        self.initial_bit_probabilities: dict[int, float] = {}
        self.bit_to_check_messages: dict[int, dict[int, float]] = {}
        self.check_to_bit_messages: dict[int, dict[int, float]] = {}
        self.frozen_bits: dict[int, tuple[int, float]] = {}
        self.marginals: dict[int, float] = {}
        

    def run(self, 
            received_message: dict,
            channel_probabilities: dict,
            num_parity_check_passes: int = 20, 
            max_iterations: int = 100,
            print_iteration_summary: bool = False,
            ) -> dict[int, tuple[int, float]]:
        """
        Run belief propagation to estimate the marginals P(s_i = 0 | r).

        Steps:
            1. Compute initial bit probabilities.
            2. Initialise bit to check messages and check to bit messages.
            3. Perform message updates:
                - update messages from check nodes to bit nodes
                - update messages from bit nodes to check nodes

        The algorithm performs passes over all check vertices.
        After each full pass, the counter increases.
        Once twenty passes have completed, marginals are returned.

        Args:
            - received_message dict[int, int]: received bits r_i in {0, 1} for each bit i.
            - channel_probabilities dict[int, float]: flip probability p_i for each bit.
            - num_parity_check_passes [int]: number of passes over all check vertices.
             - max_iterations [int]: maximum number of message updates.

        Returns:
            dict[int, float]: Maps bit i to its marginal probability.
        """
        self._initialise_bit_probabilities(received_message, channel_probabilities)
        self._initialise_messages()
        remaining_check_vertices = self.check_vertices.copy()
        if freeze_threshold is None:
            freeze_threshold = {bit : 0.01 for bit in self.bit_vertices}
        passes_completed = 0
        for _ in range(max_iterations):

            check_vertex = self._select_check_vertex(remaining_check_vertices)
            if check_vertex is None:
                break

            self._update_check_to_bit_messages(check_vertex)
            self._update_bit_to_check_messages()
            self._update_frozen_bits(freeze_threshold)

            if print_iteration_summary:
                self._print_iteration_summary(iteration, check_vertex)

            remaining_check_vertices.remove(check_vertex)
            if not remaining_check_vertices:
                passes_completed += 1
                if passes_completed < num_parity_check_passes:
                    remaining_check_vertices = self.check_vertices.copy()
                else:
                    break
                
        return self._assemble_bit_results()
    
    def _print_iteration_summary(self, iteration: int, check_vertex):
        print(f"\n=== Iteration {iteration} Summary ===")
        print(f"\nCurrent check vertex: {check_vertex}")
        print(f"\nCheck to bit messages:")
        for check, bits in self.check_to_bit_messages.items():
            print(f"Check {check}: { {bit: f'{val:.3f}' for bit, val in bits.items()} }")
        print(f"\nBit to check messages:")
        for bit, checks in self.bit_to_check_messages.items():
            print(f"Bit {bit}: { {chk: f'{val:.3f}' for chk, val in checks.items()} }")
        if not self.frozen_bits:
            print(f"\nNo bits frozen.")
        else:
            print(f"\nFrozen bits:")
            for bit, final_data in self.frozen_bits.items():
                print(f"Frozen bit {bit} at value {final_data[0]} (prob = {final_data[1]})")

    def _initialise_messages(self):
        """
        Create the initial messages for belief propagation algorithm.

        For sent bit s and received bit r.
            - Bit to check messages start at P(r | s = 0).
            - Check to bit messages start at 0.5.
        
        Args:
            - initial_bit_probabilities (dict[int, float]):
                For each bit i with sent bits s_i and received bits s_i.
                Maps a bit i to P(r_i | s_i = 0).

        Returns:
            - bit_to_check_messages (dict[int, dict[int, float]]):
                The messages from bit vertices to check vertices.
                    bit_to_check_messages[i][a] = messages from bit i to check a.
            - check_to_bit_messages (dict[int, dict[int, float]]):
                The messages from check vertices to bit vertices.
                    check_to_bit_messages[a][i] = messages from check a to bit i.
        """
        self.bit_to_check_messages = {bit : {} for bit in self.bit_vertices}
        self.check_to_bit_messages = {check : {} for check in self.check_vertices}
        for check_vertex, bit_neighbours in self.check_neighbourhood.items():
            for bit_vertex in bit_neighbours:
                self.bit_to_check_messages[bit_vertex][check_vertex] = self.initial_bit_probabilities[bit_vertex]
                self.check_to_bit_messages[check_vertex][bit_vertex] = 0.5

    def _initialise_bit_probabilities(self,
                                      received_message: dict[int, int],
                                      channel_probabilities: dict[int, float]):
        """
        Update initial_bit_probabilities attribute.
        For bit i with sent bit s_i and received bit r_i, computes 
            P(r_i | s_i = 0)
        using channel flip probabilities.
        
        Suppose a bit flips with probability p_i.
        - If r_i = 0, then
            P(r_i | s_i = 0) = 1 - p_i.
        - If r_i = 1, then
            P(r_i | s_i = 0) = p_i.                
        
        Args:
            - received_message (dict[int, int]): 
                Maps bit i to received value r_i: either 0 or 1.
            - channel_probabilities (dict[int, float]): 
                Maps bit i to probability it will flip p_i.
        """
        initial_bit_probs: dict[int, float] = {}
        for bit in self.bit_vertices:
            prob_bit_flip = channel_probabilities[bit]
            if received_message[bit] == 0:
                initial_bit_probs[bit] = 1-prob_bit_flip
            else:
                initial_bit_probs[bit] = prob_bit_flip
        self.initial_bit_probabilities = initial_bit_probs

    def _select_check_vertex(self, candidates: list[int]) -> int | None:
        """
        Select the next check vertex to update.
        TODO: implement an algorithm to do this.

        Args:
            - candidates (list[int]): list of check vertices not used in this pass.
        
        Returns:
            - int | None: a check vertex or None if the candidates list is empty.
        """
        if not candidates:
            return None
        return random.choice(candidates)

    def _update_check_to_bit_messages(self, check_vertex: int):
        """
        Update the messages from the check vertices to the bit vertices.
        
        For a check a and target bit i, the new message is
            0.5 * (1 - product_{neighbour bits except i} (1 - 2(messsage from bit to a)) )

        Args:
            - check_vertex (int): the check vertex being updated.
            - check_to_bit_messages (dict[int, dict[int, float]]): current check to bit messages.
            - bit_to_check_messages (dict[int, dict[int, float]]): current bit to check messages.
        """
        neighbour_bits = self.check_neighbourhood[check_vertex]
        for target_bit in neighbour_bits:
            prod = 1.0
            for new_bit in neighbour_bits:
                if target_bit == new_bit:
                    continue
                prob_other_bit_0 = self.bit_to_check_messages[new_bit][check_vertex]
                prod *= (1-2*prob_other_bit_0)
            self.check_to_bit_messages[check_vertex][target_bit] = 0.5*(1 - prod)

    def _update_bit_to_check_messages(self, check_vertex: int):        
        """
        Update the messages from bit vertices to check vertices.

        For bit i and check a, updates using the formula:
            P(r_i | s_i = 0)*( prod_{neighbour checks except a} message from check to i) / norm
                
        Args:
            - initial_bit_probabilities (dict[int, float]): the initial probabilities P(r_i | s_i = 0)
            - bit_to_check_messages (dict[int, dict[int, float]]): current bit to check messages.
            - check_to_bit_messages (dict[int, dict[int, float]]): current check to bit messages.
        """
        for bit in self.check_neighbourhood[check_vertex]:
            bit_neighbourhood = self.bit_neighbourhood[bit]
            for target_check in bit_neighbourhood:
                prod_0 = 1.0
                prod_1 = 1.0
                for other_check in bit_neighbourhood:
                    if other_check == target_check:
                        continue
                    message = self.check_to_bit_messages[other_check][bit]
                    prod_0 *= message
                    prod_1 *= 1 - message
                prob_bit_0 = self.initial_bit_probabilities[bit]
                self.bit_to_check_messages[bit][target_check] = (prob_bit_0 * prod_0) / (prob_bit_0 * prod_0 + (1-prob_bit_0) * prod_1)
    
    def _update_frozen_bits(self, freeze_threshold: dict[int, float]):
        self._update_bit_marginals()
        for bit in self.bit_to_check_messages.keys():
            if bit in self.frozen_bits.keys():
                continue
            marginal = self.marginals[bit]
            if marginal < freeze_threshold[bit]:
                self.frozen_bits[bit] = (1, marginal)
                for check in self.bit_to_check_messages[bit].keys():
                    self.bit_to_check_messages[bit][check] = 0.0
            if marginal > 1 - freeze_threshold[bit]:
                self.frozen_bits[bit] = (0, marginal)
                for check in self.bit_to_check_messages[bit].keys():
                    self.bit_to_check_messages[bit][check] = 1.0

    
    def _update_bit_marginals(self):
        """
        Compute the marginal probabilities for each bit after message passing.

        For bit i with sent bit s_i and received bit r_i, compute
            P(r_i | s_i = 0)*(prod_{neighbouring check vertices} message check to i) / norm.

        Args:
            - initial_bit_probabilities (dict[int, float]): maps bit to initial probability P(r_i | s_i = 0).
            - check_to_bit_messages (dict[int, dict[int, float]]): messages from check vertices to bit vertices.

        Returns:
            - dict[int, float]:
                Maps a bit to the probability the code bit was a 0.
        """
        for bit in self.bit_vertices:
            bit_neighbourhood = self.bit_neighbourhood[bit]
            prod_0 = 1.0
            prod_1 = 1.0
            for check in bit_neighbourhood:
                message = self.check_to_bit_messages[check][bit]
                prod_0 *= message
                prod_1 *= 1 - message
            prob_bit_0 = self.initial_bit_probabilities[bit]
            self.marginals[bit] = (prob_bit_0 * prod_0) / (prob_bit_0 * prod_0 + (1 - prob_bit_0) * prod_1)
    
    def _assemble_bit_results(self) -> dict[int, tuple[int, float]]:
        """
        Combine frozen bits and remaining bits to produce final estimates.

        For frozen bits, keep their frozen value and marginal.
        For other bits, assign 0 or 1 based on marginal > 0.5, 
        and also return the marginal probability.
        """
        results: dict[int, tuple[int, float]] = {}
        self._update_bit_marginals()
        for bit in self.bit_vertices:
            if bit in self.frozen_bits:
                results[bit] = self.frozen_bits[bit]
            else:
                marginal = self.marginals[bit]
                if marginal > 0.5:
                    value = 0
                else:
                    value = 1
                results[bit] = (value, marginal)
        return results

class BPExample:
    """Run a Belief Propagation example on a given LinearCode."""
    
    def __init__(self, code: LinearCode, channel_prob: float = 0.25):
        self.code = code
        self.channel_prob = channel_prob
        self.bp = BeliefPropagation(code)
        
        self.codeword = random.choice(code.codewords)
        
        self.transmitted = code.transmit_codeword(self.codeword.bits,
                                                  [channel_prob]*code.length)
        
        self.message = {bit: self.transmitted[bit_idx] 
                        for bit_idx, bit in enumerate(self.bp.bit_vertices)}
        self.channel_probabilities = {bit: channel_prob for bit in self.bp.bit_vertices}
        
        self.initial_bit_states = self.bp.initialise_bit_state(self.message,
                                                               self.channel_probabilities)
        self.bit_to_check_messages, self.check_to_bit_messages = self.bp.initialise_messages(
            self.initial_bit_states
        )
        
    def print_setup(self):
        print(f"\n=== Code:")
        print(self.code.generator_matrix.array)
        print(f"Codewords: {len(self.code.codewords)}")
        print("\nSelected codeword:")
        print(f"Bits: {self.codeword.bits}\n")
        print("Transmitted message through channel:")
        print(self.transmitted)
        print("\nParity check equations:")
        for idx, eq in enumerate(self.code.parity_check_eqns):
            print(f"Check {idx}: {eq}")
        print("\nBelief Propagation Graph:")
        print(f"Check vertices: {self.bp.check_vertices}")
        print(f"Bit vertices: {self.bp.bit_vertices}")
        print(f"Check neighbourhood: {self.bp.check_neighbourhood}")
        print(f"Bit neighbourhood: {self.bp.bit_neighbourhood}")
        print("\nInitial bit states:")
        for bit, state in self.initial_bit_states.items():
            print(f"Bit {bit}: {state:.3f}")
        print("\nInitial bit to check messages:")
        for bit, checks in self.bit_to_check_messages.items():
            print(f"Bit {bit}: { {chk: f'{val:.3f}' for chk, val in checks.items()} }")
        print("\nInitial check to bit messages:")
        for check, bits in self.check_to_bit_messages.items():
            print(f"Check {check}: { {bit: f'{val:.3f}' for bit, val in bits.items()} }")
            
    def run_bp(self):
        print("\nRunning Belief Propagation.")
        marginals = self.bp.run(self.message, self.channel_probabilities)
        print("\nFinal bit marginals:")
        for bit, prob in marginals.items():
            print(f"Bit {bit}: {prob:.3f}")
        return marginals


if __name__ == "__main__":

    from linearcodes import RepetitionCode, RandomLDPC

    code = RandomLDPC(7, 3, [2])
    example = BPExample(code)
    example.print_setup()
    marginals = example.run_bp()

    code = RepetitionCode(3)
    example = BPExample(code)
    example.print_setup()
    example.run_bp()