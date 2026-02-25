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
        
    def run(self, channel_probabilities: list[float], max_iterations: int = 100):
        messages = self.initialise_messages()
        bit_state = self.initialise_bit_state(channel_probabilities)

        for _ in range(max_iterations):
            check_vertex = self.select_check_vertex()
            if check_vertex is None:
                break
            bit_state = self.get_bit_state(check_node, bit_state, messages)
            check_update = self.compute_check_update(check_node, incoming)
            self.apply_bit_update(check_node, check_update, bit_state, messages)

    def initialise_messages(self):
        pass

    def initialise_bit_state(self, channel_probabilities):
        pass

    def select_check_vertex(self):
        pass

    def get_bit_state(self, check_node, bit_state, messages):
        pass

    def compute_check_update(self, check_node, incoming):
        pass

    def apply_bit_update(self, check_node, check_update, bit_state, messages):
        pass

