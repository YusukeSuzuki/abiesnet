from argparse import ArgumentParser as AP
from pathlib import Path
import fnmatch
import tensorflow as tf
import image_loader as il
import yaml_loader as yl

ROOT_VARIABLE_SCOPE='abiesnet'
MODELS_DIR='models'
MODEL_YAML_PATH='abies_model.yaml'

INPUT_WIDTH=256
INPUT_HEIGHT=144
INPUT_CHANNELS=3

loader_param = {
    'flip_up_down': True,
    'flip_left_right': True,
    'random_brightness': True,
    'random_contrast': True
    }

network_param = {
    'device' : 'gpu:0',
    }

# --------------------------------------------------------------------------------
# sub command methods
# --------------------------------------------------------------------------------

def do_train(namespace):
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir/namespace.modelfile
    model_backup_path = models_dir/(namespace.modelfile+'.back')

    # build
    reader = tf.WholeFileReader()

    with tf.variable_scope('image_loader'), tf.device('/cpu:0'):
        print('read samples directory')
        samples_dir = Path(namespace.samples)
        samples = []

        if samples_dir.is_dir():
            samples = [str(p.resolve())
                for p in samples_dir.iterdir() if p.suffix == '.jpg']
        else:
            samples = [l.rstrip() for l in  samples_dir.open().readlines()]

        batch_images = il.build_full_network(
            samples, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, 16, reader,
            **loader_param)
        image_summary = tf.image_summary('input_image', batch_images)

    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        print('build network')
        graph_root = yl.load(MODEL_YAML_PATH)
        tags = graph_root.build(feed_dict={'root': batch_images})
        #out, train1 = am.build_full_network(batch_images,
        #    with_conv1_pre_train=True)

    # get optimizer for train
    train = tf.get_default_graph().get_operation_by_name(namespace.optimizer)

    # create saver and logger
    saver = tf.train.Saver()
    merged = tf.merge_all_summaries()

    # ready to run

    print('initialize')
    sess = tf.Session()
    writer = tf.train.SummaryWriter(namespace.logdir, sess.graph)
    sess.run(tf.initialize_all_variables())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # run

    if namespace.restore:
        print('restore {}'.format(namespace.restore))
        saver.restore(sess, namespace.restore)

    print('train')
    writer.add_graph(tf.get_default_graph())

    for i in range(0, 100000):
        print('loop: {}'.format(i))
        summary, res = sess.run( (merged, train), feed_dict={} )
        writer.add_summary(summary, i)

        if i % 2000 == 1:
            print('save backup to: {}'.format(model_backup_path))
            saver.save(sess, str(model_backup_path))

    print('save to: {}'.format(model_path))
    saver.save(sess, str(model_path))

    # finalize

    writer.close()

def do_test(namespace):
    # stub
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

def do_eval(namespace):
    # stub
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

def do_dump_network(namespace):
    # build
    reader = tf.WholeFileReader()
    with tf.variable_scope('image_loader'), tf.device('/cpu:0'):
        samples = ['dummy']

        batch_images = il.build_full_network(
            samples, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, 16, reader,
            **loader_param)
        image_summary = tf.image_summary('input_image', batch_images)

    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        graph_root = yl.load(MODEL_YAML_PATH)
        tags = graph_root.build(feed_dict={'root': batch_images})

    print('-- variables')
    for variable in tf.all_variables():
        if fnmatch.fnmatch(variable.name, namespace.pattern):
            print(variable.name)

    print('-- operations')
    for operation in tf.get_default_graph().get_operations():
        if fnmatch.fnmatch(operation.name, namespace.pattern):
            print(operation.name)

# --------------------------------------------------------------------------------
# command line option parser
# --------------------------------------------------------------------------------

def create_parser():
    parser = AP(prog='abies')
    parser.set_defaults(func=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--modelfile', type=str, default='model.ckpt')
    parser.add_argument('--restore', type=str, default='')
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser('train')
    sub_parser.set_defaults(func=do_train)
    sub_parser.add_argument('--optimizer', type=str, required=True)
    sub_parser.add_argument('--samples', type=str, default='./samples')
    sub_parser = sub_parsers.add_parser('test')
    sub_parser.set_defaults(func=do_test)
    sub_parser = sub_parsers.add_parser('eval')
    sub_parser.set_defaults(func=do_eval)
    sub_parser = sub_parsers.add_parser('dump-network')
    sub_parser.set_defaults(func=do_dump_network)
    sub_parser.add_argument('--pattern', type=str, default='*')

    return parser

# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

def run():
    parser = create_parser()
    namespace = parser.parse_args()

    if namespace.func:
        namespace.func(namespace)
    else:
        parser.print_help()

if __name__ == '__main__':
    run()

