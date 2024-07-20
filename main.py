import utils
import data_class as dc
from tree import DecisionTreeClassifier, draw_tree

METHOD_ZOO = [ 'entropy', 'ratio', 'gini' ]
KNOB = 0
method = utils.divide_feature_method(METHOD_ZOO[ KNOB ])

if __name__ == '__main__':
    filename = 'datasets/melon2.csv'
    tree_name = filename[ 9:-4 ] + '_by_ ' + METHOD_ZOO[ KNOB ]
    data = dc.from_csv(filename)
    dt = DecisionTreeClassifier()
    dt.fit(data, method)
    draw_tree(dt, tree_name)
    print(f"Tree Image saved in [ out/{tree_name}.png ] successfully!")
