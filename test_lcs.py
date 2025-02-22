class SuffixTreeNode:
    def __init__(self):
        self.children = {}
        self.start = -1
        self.end = -1
        self.suffix_link = None
        self.string_set = set()

class SuffixTree:
    def __init__(self, text1, text2):
        self.text = text1 + "#" + text2 + "$"
        self.separator_index = len(text1)
        self.root = SuffixTreeNode()
        self.build_suffix_tree()

    def build_suffix_tree(self):
        n = len(self.text)
        self.root.end = -1
        self.root.suffix_link = self.root
        active_node = self.root

        active_edge = -1
        active_length = 0
        remaining_suffix_count = 0
        leaf_end = -1

        def edge_length(node):
            return (node.end if node.end != -1 else leaf_end) - node.start + 1

        def walk_down(node):
            nonlocal active_node, active_edge, active_length
            if active_length >= edge_length(node):
                active_edge += edge_length(node)
                active_length -= edge_length(node)
                active_node = node
                return True
            return False

        for i in range(n):
            leaf_end = i
            remaining_suffix_count += 1
            last_new_node = None

            while remaining_suffix_count > 0:
                if active_length == 0:
                    active_edge = i

                if self.text[active_edge] not in active_node.children:
                    new_leaf = SuffixTreeNode()
                    new_leaf.start = i
                    new_leaf.end = leaf_end
                    active_node.children[self.text[active_edge]] = new_leaf
                    if last_new_node:
                        last_new_node.suffix_link = active_node
                        last_new_node = None
                else:
                    next_node = active_node.children[self.text[active_edge]]
                    if walk_down(next_node):
                        continue

                    if self.text[next_node.start + active_length] == self.text[i]:
                        if last_new_node and active_node != self.root:
                            last_new_node.suffix_link = active_node
                            last_new_node = None
                        active_length += 1
                        break

                    split = SuffixTreeNode()
                    split.start = next_node.start
                    split.end = next_node.start + active_length - 1
                    active_node.children[self.text[active_edge]] = split

                    new_leaf = SuffixTreeNode()
                    new_leaf.start = i
                    new_leaf.end = leaf_end
                    split.children[self.text[i]] = new_leaf

                    next_node.start += active_length
                    split.children[self.text[next_node.start]] = next_node

                    if last_new_node:
                        last_new_node.suffix_link = split

                    last_new_node = split

                remaining_suffix_count -= 1
                if active_node == self.root and active_length > 0:
                    active_length -= 1
                    active_edge = i - remaining_suffix_count + 1
                elif active_node != self.root:
                    active_node = active_node.suffix_link

    def mark_suffixes(self, node=None, depth=0):
        if node is None:
            node = self.root

        is_leaf = len(node.children) == 0
        if is_leaf:
            if node.start <= self.separator_index:
                node.string_set.add(1)
            else:
                node.string_set.add(2)
            return node.string_set

        for child in node.children.values():
            child_set = self.mark_suffixes(child, depth + (child.end - child.start + 1))
            node.string_set.update(child_set)

        return node.string_set

    def find_lcs(self):
        def dfs(node, depth):
            nonlocal max_length, lcs_start
            if 1 in node.string_set and 2 in node.string_set:
                if depth > max_length:
                    max_length = depth
                    lcs_start = node.end - depth + 1

            for child in node.children.values():
                dfs(child, depth + (child.end - child.start + 1))

        max_length = 0
        lcs_start = -1
        self.mark_suffixes()
        dfs(self.root, 0)

        return self.text[lcs_start:lcs_start + max_length] if max_length > 0 else ""

def longest_common_subsequence(text1, text2):
    suffix_tree = SuffixTree(text1, text2)
    return suffix_tree.find_lcs()

# # Example Usage:
# text1 = "ABAB"
# text2 = "BABA"
# print(longest_common_subsequence(text1, text2))