import yaml_loader as yl
import tensorflow as tf

graph_root = yl.load('abies_model.yaml')
pf = tf.placeholder(tf.float32, [3,64,64,3])
graph_root.build({'root': pf})

print('-- variables --')
for node in tf.all_variables():
    print(node.name)

print('-- ops --')
for op in tf.get_default_graph().get_operations():
    print(op.name)

sess = tf.Session()
writer = tf.train.SummaryWriter('logs', sess.graph)
init = tf.initialize_all_variables()
sess.run(init)
writer.add_graph(tf.get_default_graph())
writer.close()

