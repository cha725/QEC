import networkx as nx

class MessagePassingTree:
    """
    Message passing in a tree with num_vertices vertices.
    Want to pass messages and add them up to create final messages.
    """
    def __init__(self,
                 num_vertices: int,
                 initial_node_information: list[int] | None = None):
        if initial_node_information is None:
            initial_node_information = [1]*num_vertices
        if len(initial_node_information) != num_vertices:
            raise ValueError("Each node needs initial information.")
        self.num_vertices = num_vertices
        self.tree = nx.generators.random_labeled_tree(num_vertices)
        self.neighbours = [list(self.tree.neighbors(u)) for u in self.tree.nodes]
        self.initial_node_information = initial_node_information
        self.directed_edges = [(u, v) for u in self.tree.nodes for v in self.tree.neighbors(u)]
        self.initial_messages = {edge : self.initial_node_information[edge[0]] for edge in self.directed_edges}

    def run(self, num_interations: int, verbose: bool = False):
        """
        Run the message passing algorithm for num_iterations iterations.
        For a print out statement at each iteration including the new messages and node information
        set verbose to be True.
        Since the algorithm is running on a tree it will stop in finitely many iterations.
        This function stops once the iteration is producing the same result.
        """
        node_information = self.initial_node_information
        messages = self.initial_messages
        for iteration_idx in range(num_interations):
            new_node_information = self.update_node_information(node_information, messages)
            if new_node_information == node_information:
                print(f"Stopped at iteration {iteration_idx+1}.")
                return node_information
            node_information = new_node_information
            messages = self.update_messages(messages)
            if verbose:
                print(f"At iteration {iteration_idx+1}: messages {messages}, node information {node_information}.")
        return node_information

    def update_messages(self,
                        messages: dict[tuple[int,int], int]) -> dict[tuple[int,int], int]:
        """
        Given incoming messages update the messages for the next iteration.
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
    
    def update_node_information(self, 
                                node_information: list[int], 
                                messages: dict[tuple[int,int], int]) -> list[int]:
        """
        Update the node information with the incoming messages.
        """
        nodes = messages.keys()
        new_node_information = node_information.copy()
        for u, v in nodes:
            message = messages[(u,v)]
            new_node_information[v] += message
        return new_node_information
    

mp = MessagePassingTree(num_vertices=1000)

final_info = mp.run(1000)
print("Final node information:", final_info)
        
    