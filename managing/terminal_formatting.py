

def parse_nodes(node_string):
    node_list = []
    current_buffer_stack = [""]
    start_node = None

    def add_node(name):
        node_list.append(name)

    def empty_buffer():
        if current_buffer_stack[-1] == "":
            return
        nonlocal start_node
        if start_node is None:
            add_node(''.join(current_buffer_stack))
        else:
            fmt = f"0{len(start_node)}d"
            for node in range(int(start_node), int(current_buffer_stack[-1]) + 1):
                add_node(''.join(current_buffer_stack[:-1]) + f"{node:{fmt}}")
        current_buffer_stack[-1] = ""
        start_node = None

    for i in range(len(node_string)):
        if node_string[i] == "]":
            empty_buffer()
            current_buffer_stack.pop()
            current_buffer_stack[-1] = ""
            continue
        if node_string[i] == ",":
            empty_buffer()
            continue
        if node_string[i] == "-":
            start_node = current_buffer_stack[-1]
            current_buffer_stack[-1] = ""
            continue
        if node_string[i] == "[":
            current_buffer_stack.append("")
            continue
        current_buffer_stack[-1] += node_string[i]

    empty_buffer()
    return node_list


def cartesian_product_of_nodes_and_ports(node_list, ports):
    node_and_port_list = []
    for node in node_list:
        for port in ports:
            node_and_port_list.append(f"{node}:{port}")
    return node_and_port_list

def get_ansi_foreground_color(color_literal):
    supported_colors = {
        "gray": 90,
        "red": 31,
        "green": 32,
        "yellow": 93,
    }
    default_color = 0
    return f"\033[{supported_colors.get(color_literal, default_color)}m"


def clen(s):
    in_ansi_escape = False
    ans = 0
    for i in range(len(s)):
        if s[i] == "\033":
            in_ansi_escape = True
        if not in_ansi_escape:
            ans += 1
        if s[i] == "m":
            in_ansi_escape = False
    return ans


def refresh_string(string, printed_with_new_line=True):
    """
    After printing the string with or without a new line, call this to get the sequence needed to move back the cursor.
    Print this sequence without a new line.
    """
    num_of_lines = string.count("\n") + int(printed_with_new_line)
    return f"\033[{len(num_of_lines)}A\033[1G"

def printable_table(a_list_of_dicts, header, old_maximum_widths=None):
    """
    a_list_of_dicts: rows, where columns are values of keys in header
    old_maximum_widths: for keeping the table from shrinking in width
    """
    maximum_widths = []
    for i in range(len(header)):
        if len(maximum_widths) <= i:
            maximum_widths.append(clen(header[i]))
        maximum_width = maximum_widths[i]
        for j in range(len(a_list_of_dicts)):
            maximum_width = max(maximum_width, clen(str(a_list_of_dicts[j][header[i]])))
        maximum_width = maximum_width // 4 * 4 + 4
        if old_maximum_widths:
            maximum_width = max(maximum_width, old_maximum_widths[i])
        maximum_widths[i] = maximum_width

    table_str = ""

    def print_max_width(mw, txt, space=" "):
        return space * (max(0, mw - clen(str(txt)))) + str(txt)

    for i in range(len(header)):
        # print header
        table_str += print_max_width( maximum_widths[i], header[i])
    table_str += "\n"
    for i in range(len(header)):
        table_str += print_max_width(maximum_widths[i], "", "-")
    table_str += "\n"
    for j in range(len(a_list_of_dicts)):
        for i in range(len(header)):
            table_str += print_max_width(maximum_widths[i], a_list_of_dicts[j][header[i]])
        table_str += "\n"
    return table_str, maximum_widths


if __name__ == '__main__':
    import sys
    print(parse_nodes(sys.argv[1]))