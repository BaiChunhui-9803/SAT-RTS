class ClusterNode:
    def __init__(self, state, cluster_id):
        self.state = state
        self.cluster_id = cluster_id
        self.children = {}
        self.state_list = []

    def add_child(self, distance, child):
        self.children[distance] = child

    def add_state(self, state):
        self.state_list.append(state)


class BKTree:
    def __init__(self, distance_func, distance_index=0):
        self.root = None
        self.distance_func = distance_func
        self.distance_index = distance_index
        self.next_cluster_id = 2

    def insert(self, node, parent=None):
        if parent is None:
            self.root = node
            node.cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            return
        dist = self.distance_func(node.state, parent.state)[self.distance_index]
        if dist in parent.children:
            self.insert(node, parent.children[dist])
        else:
            parent.add_child(dist, node)
            node.cluster_id = self.next_cluster_id
            self.next_cluster_id += 1

    def query(self, state, threshold):
        def search(node, dist):
            if dist < threshold:
                return node.cluster_id
            for d, child in node.children.items():
                if abs(d - dist) < threshold:
                    result = search(
                        child,
                        self.distance_func(state, child.state)[self.distance_index],
                    )
                    if result is not None:
                        return result
            return None

        if self.root is None:
            return None
        return search(
            self.root, self.distance_func(state, self.root.state)[self.distance_index]
        )

    def get_next_cluster_id(self):
        return self.next_cluster_id

    def find_node_by_cluster_id(self, cluster_id):
        """
        Recursively find the BKTreeNode with the specified cluster_id
        """

        def search_node(node):
            if node.cluster_id == cluster_id:
                return node
            for child in node.children.values():
                result = search_node(child)
                if result:
                    return result
            return None

        if self.root:
            return search_node(self.root)
        return None


def classify_new_state(new_state, bktree, threshold=1.0):
    cluster_id = bktree.query(new_state, threshold)
    if cluster_id is not None:
        return cluster_id
    else:
        new_cluster_id = bktree.get_next_cluster_id()
        new_node = ClusterNode(new_state, new_cluster_id)
        bktree.insert(new_node, bktree.root)
        return new_cluster_id
