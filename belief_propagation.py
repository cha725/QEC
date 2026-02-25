import networkx as nx

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
                 edge_set: list):
        self.edge_set = edge_set
        vertex_info = self._initialise_vertices()
        self.check_vertices = vertex_info["check_vertices"]
        self.bit_vertices = vertex_info["bit_vertices"]
        self.check_neighbourhood = vertex_info["check_neighbourhood"]
        self.bit_neighbourhood = vertex_info["bit_neighbourhood"]
        self.graph = self._create_bipartite_graph()

    def _initialise_vertices(self) -> dict:
        """
        Create check and bit vertex lists from edge set.
        Also create dictionaries to store neighbourhoods of each vertex.
        """
        check_vertices = set()
        bit_vertices = set()
        check_neighbourhood = defaultdict(set)
        bit_neighbourhood = defaultdict(set)

        for source, target in self.edge_set:
            check_vertices.add(source)
            bit_vertices.add(target)
            check_neighbourhood[source].add(target)
            bit_neighbourhood[target].add(source)

        return {"check_vertices": list(check_vertices),
                "bit_vertices": list(bit_vertices),
                "check_neighbourhood": dict(check_neighbourhood),
                "bit_neighbourhood": dict(bit_neighbourhood),}

    def _create_bipartite_graph(self):
        """
        Create NetworkX bipartite graph. 
        Check vertices have the bipartite label 'check',
        bit vertices havd the bipartite label 'bit'.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.check_vertices, bipartite='check')
        graph.add_nodes_from(self.bit_vertices, bipartite='bit')
        graph.add_edges_from(self.edge_set)
        return graph
        
    def run(self, 
            received_message: list[int],
            channel_probabilities: list[float], 
            max_iterations: int = 100):
        bit_state = self.initialise_bit_state(received_message, channel_probabilities)
        bit_to_check_messages, check_to_bit_messages = self.initialise_messages(bit_state)

        for _ in range(max_iterations):
            check_vertex = self.select_check_vertex()
            if check_vertex is None:
                break
            bit_state = self.get_bit_state(check_vertex, bit_state, check_to_bit_messages)
            check_update = self.compute_check_update(check_vertex, bit_to_check_messages)
            self.apply_bit_update(check_vertex, check_update, bit_state)

    def initialise_messages(self, bit_state: dict):
        """
        Create dictionaries to hold the messages to and from bit and check vertices.
        Initialise bit to check as P(mi given ci=0).
        Initialise check to bit as 0.0.
        """
        bit_to_check_messages = {}
        check_to_bit_messages = {}
        for check_vertex, bit_neighbours in self.check_neighbourhood.items():
            for bit_vertex in bit_neighbours:
                bit_to_check_messages[(bit_vertex, check_vertex)] = bit_state[bit_vertex]
                check_to_bit_messages[(check_vertex, bit_vertex)] = 0.0
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

    def select_check_vertex(self):
        """
        Select the next check vertex to consider. Currently just selects randomly.
        TODO: implement an algorithm to do this.
        """
        return random.choice(self.check_vertices)

    def get_bit_state(self, check_vertex, bit_state: dict):
        """
        Get bit state from all bits connected to this check vertex.
        """
        neighbour_bit_states = {}
        for bit in self.check_neighbourhood[check_vertex]:
            neighbour_bit_states[bit] = bit_state[bit]
        return neighbour_bit_states

    def compute_check_update(self, neighbour_bit_states: dict) -> dict:
        check_to_bit_messages = {}
        neighbour_bits = list(neighbour_bit_states.keys())
        for target_bit in neighbour_bits:
            other_bits = [bit for bit in neighbour_bits if bit != target_bit]
            check_to_bit_messages[target_bit] = self._sum_probs_over_bits(other_bits, neighbour_bit_states)
        return check_to_bit_messages

        pass


# Repetition code parity check
edge_set = [('a', 0), ('a', 1), ('b', 1), ('b', 2)]
received_message = {0: 0, 1: 1, 2: 0}
channel_probabilities = {0: 0.25, 1: 0.25, 2: 0.25}

bp = BeliefPropagation(edge_set)

print("Check vertices:", bp.check_vertices)
print("Bit vertices:", bp.bit_vertices)
print("Check neighbourhood:", bp.check_neighbourhood)
print("Bit neighbourhood:", bp.bit_neighbourhood)

print("Graph nodes:", bp.graph.nodes(data=True))
print("Graph edges:", bp.graph.edges())


bit_state = bp.initialise_bit_state(received_message, channel_probabilities)
# m=010, should be: 0.75, 0.25, 0.75 
print("Initial bit state:", bit_state)


bit_to_check_messages, check_to_bit_messages = bp.initialise_messages(bit_state)
# Should be {(0,a):0.75, (1,a):0.25, (1,b):0.25, (2,b):0.75}
print("Bit to check messages:", bit_to_check_messages)
# Should be {(a,0):0, (a,1):0, (b,1):0, (b,2):0}
print("Check to bit messages:", check_to_bit_messages)

random_check = bp.select_check_vertex()
print("Randomly selected check vertex:", random_check)

# Set a vertex for the rest of the time to check the rest of the methods correctly

random_check = 'a'

neighbour_state = bp.get_bit_state(random_check, bit_state)
# Should be {0:0.75, 1:0.25}
print(f"Neighbouring bit states of check {random_check}:", neighbour_state)

