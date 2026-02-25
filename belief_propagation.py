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
        self.code = code
        self.graph = self.code.graph        

        self.check_vertices: list = [v for v, d in self.graph.nodes(data=True) if d.get('bipartite') == 'check']
        self.bit_vertices: list = [v for v, d in self.graph.nodes(data=True) if d.get('bipartite') == 'bit']
        self.check_neighbourhood: dict = {v : list(self.graph.neighbors(v)) for v in self.check_vertices}
        self.bit_neighbourhood: dict = {v : list(self.graph.neighbors(v)) for v in self.bit_vertices}

        self.code_length = len(self.bit_vertices)
        self.num_parity_check_eqns = len(self.check_vertices)
        self.code_rank = self.code_length - self.num_parity_check_eqns
        

    def run(self, 
            received_message: dict,
            channel_probabilities: dict, 
            max_iterations: int = 100):
        initial_bit_states = self.initialise_bit_state(received_message, channel_probabilities)
        bit_to_check_messages, check_to_bit_messages = self.initialise_messages(initial_bit_states)
        previous_vertex = None
        for _ in range(max_iterations):
            check_vertex = self.select_check_vertex(previous_vertex)
            if check_vertex is None:
                break
            check_to_bit_messages = self.update_check_to_bit_messages(check_vertex, check_to_bit_messages, bit_to_check_messages)
            bit_to_check_messages = self.update_bit_to_check_messages(initial_bit_states, bit_to_check_messages, check_to_bit_messages)
            previous_vertex = check_vertex
        return self.compute_marginals(initial_bit_states, check_to_bit_messages)

    def initialise_messages(self, initial_bit_state: dict):
        """
        Create dictionaries to hold the messages to and from bit and check vertices.
        Initialise bit to check as P(mi given ci=0).
        Initialise check to bit as 0.0.
        """
        bit_to_check_messages = {bit : {} for bit in self.bit_vertices}
        check_to_bit_messages = {check : {} for check in self.check_vertices}
        for check_vertex, bit_neighbours in self.check_neighbourhood.items():
            for bit_vertex in bit_neighbours:
                bit_to_check_messages[bit_vertex][check_vertex] = initial_bit_state[bit_vertex]
                check_to_bit_messages[check_vertex][bit_vertex] = 0.0
        return bit_to_check_messages, check_to_bit_messages

    def initialise_bit_state(self,
                             received_message: dict, 
                             channel_probabilities: dict):
        """
        Returns dictionary with keys given by the bit vertices and whose values are
        P(bit = received_message given c=0).
        """
        initial_bit_state = {}
        for bit in self.bit_vertices:
            p = channel_probabilities[bit]
            if received_message[bit] == 0:
                initial_bit_state[bit] = 1-p
            else:
                initial_bit_state[bit] = p
        return initial_bit_state

    def select_check_vertex(self, previous_vertex = None):
        """
        Select the next check vertex to consider. Currently just selects randomly.
        TODO: implement an algorithm to do this.
        """
        candidate_vertices = [v for v in self.check_vertices if v != previous_vertex]
        return random.choice(candidate_vertices)

    def update_check_to_bit_messages(self, check_vertex, check_to_bit_messages: dict, bit_to_check_messages: dict) -> dict:
        neighbour_bits = self.check_neighbourhood[check_vertex]
        for target_bit in neighbour_bits:
            prod = 1.0
            for new_bit in neighbour_bits:
                if target_bit == new_bit:
                    continue
                prob_other_bit_0 = bit_to_check_messages[new_bit][check_vertex]
                prod *= (1-2*prob_other_bit_0)
            check_to_bit_messages[check_vertex][target_bit] = 0.5*(1 - prod)
        return check_to_bit_messages

    def update_bit_to_check_messages(self, initial_bit_states: dict, bit_to_check_messages: dict, check_to_bit_messages: dict):        
        for bit in self.bit_vertices:
            bit_neighbourhood = self.bit_neighbourhood[bit]
            for target_check in bit_neighbourhood:
                prod_0 = 1.0
                prod_1 = 1.0
                for other_check in bit_neighbourhood:
                    if other_check == target_check:
                        continue
                    c = check_to_bit_messages[other_check][bit]
                    prod_0 *= c
                    prod_1 *= 1 - c
                p_bit = initial_bit_states[bit]
                bit_to_check_messages[bit][target_check] = (p_bit * prod_0) / (p_bit * prod_0 + (1-p_bit) * prod_1)
        return bit_to_check_messages
    
    def compute_marginals(self, initial_bit_states: dict, check_to_bit_messages: dict):
        marginals = {bit : 0.0 for bit in self.bit_vertices}
        for bit in self.bit_vertices:
            bit_neighbourhood = self.bit_neighbourhood[bit]
            prod_0 = 1.0
            prod_1 = 1.0
            for check in bit_neighbourhood:
                c = check_to_bit_messages[check][bit]
                prod_0 *= c
                prod_1 *= 1-c
            p_bit = initial_bit_states[bit]
            marginals[bit] = (p_bit * prod_0) / (p_bit * prod_0 + (1-p_bit) * prod_1)
        return marginals

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