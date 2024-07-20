import graphviz
from tree.TreeNode import TreeNode
from tree.DecisionTreeClassifier import DecisionTreeClassifier as dtc

####### HYPER PARAMS #######
leaf_shape = 'ellipse'
leaf_color = 'orange'

branch_shape = 'box'
branch_color = 'lightblue'

node_fontsize = '20'
node_fontname = 'NSimSun'
node_style = 'filled'

edge_fontsize = '15'
edge_fontname = 'NSimSun'
############################

def set_node(graph, node_type):
    assert node_type in [ 'leaf', 'branch' ]
    if node_type == 'leaf':
        graph.attr('node', shape=leaf_shape, style=node_style,
                   color=leaf_color, fontname=node_fontname, fontsize=node_fontsize)
    else:
        graph.attr('node', shape=branch_shape, style=node_style,
                   color=branch_color, fontname=node_fontname, fontsize=node_fontsize)


def set_edge(graph):
    graph.attr('edge', fontname=edge_fontname, fontsize=edge_fontsize)


def draw_one_node(graph, cur_node: TreeNode, features_names, continue_flags):
    """
    画一个结点
    """
    # 叶子结点
    if cur_node.is_leaf():
        set_node(graph, 'leaf')
        graph.node(name=str(cur_node.node_id), label=str(cur_node.final_label))
    # 分支结点 (包括 root )
    else:
        set_node(graph, 'branch')
        if continue_flags[ cur_node.divided_feature_id ]:
            graph.node(
                name=str(cur_node.node_id),
                label=
                str(features_names[ cur_node.divided_feature_id ]) +
                ' ≤ ' + "%.3f" % cur_node.split_continue_feature_val + " ?")
        else:
            graph.node(name=str(cur_node.node_id),
                       label=str(features_names[ cur_node.divided_feature_id ]) + ' = ?')


def draw_children(graph, cur_node: TreeNode, features_names, continue_flags):
    """
    画出当前结点及其子结点, 并用边连起来
    """
    for child_id in range(len(cur_node.children)):
        child = cur_node.children[ child_id ]
        draw_one_node(graph, child, features_names, continue_flags)
        if continue_flags[ cur_node.divided_feature_id ]:
            if cur_node.child_div_feature_vals[ child_id ] == 'smaller':
                graph.edge(str(cur_node.node_id), str(child.node_id), label='是')
            else:
                graph.edge(str(cur_node.node_id), str(child.node_id), label='否')
        else:
            graph.edge(str(cur_node.node_id), str(child.node_id),
                       label=str(cur_node.child_div_feature_vals[ child_id ]))


def draw_tree(tree: dtc, filename=""):
    """
    用 bfs 的方式把一整棵树画出来
    """
    tree_graph = graphviz.Graph(name="DecisionTree", filename=filename + '_Tree',
                                directory="out", format='png')
    root = tree.root
    features_names = tree.features_names
    continue_flags = tree.continue_flags
    draw_one_node(tree_graph, root, features_names, continue_flags)
    set_edge(tree_graph)
    que = [ root ]
    i = 0
    while i < len(que):
        draw_children(tree_graph, que[ i ], features_names, continue_flags)
        que.extend(que[ i ].children)
        i += 1
    # 生成最终的 png 图片 并 清理中间文件
    tree_graph.view(cleanup=True)
