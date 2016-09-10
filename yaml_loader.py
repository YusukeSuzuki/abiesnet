import yaml
import tensorflow as tf

# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------

def weight_variable(shape, dev=0.35, name=None):
    """create weight variable for conv2d(weight sharing)"""

    return tf.get_variable(name, shape,
        initializer=tf.truncated_normal_initializer(stddev=dev))

def bias_variable(shape, val=0.1, name=None):
    """create bias variable for conv2d(weight sharing)"""

    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(val))

class WithNone:
    def __enter__(self): pass
    def __exit__(self,t,v,tb): pass

# ------------------------------------------------------------
# YAML Graph Nodes
# ------------------------------------------------------------

class Loader(yaml.Loader):
    def __init__(self, stream):
        self.tags = {}
        super(Loader, self).__init__(stream)

class Root(yaml.YAMLObject):
    yaml_tag = u'!root'

    def __init__(self, nodes, nodes_required):
        self.nodes = nodes
        self.nodes_required = nodes_required

    def __repr__(self):
        return 'Root'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'nodes': yaml_dict.get('nodes', None),
            'nodes_required': yaml_dict.get('nodes_required', None),
        }

        assert type(args['nodes']) is list, \
            'line {}: nodes must be list'.format(node.end_mark.line+1)
        assert type(args['nodes_required']) is list, \
            'line {}: nodes_required must be list'.format(node.end_mark.line+1)

        for required_node in args['nodes_required']:
            assert required_node not in loader.tags, \
                'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, required_node)
            print(required_node)
            loader.tags[required_node] = node.end_mark

        return cls(**args)

    def build(self, feed_dict={}):
        self.__tags = {}

        for key, val in feed_dict.items():
            self.__tags[key]  = val

        for required_node in self.nodes_required:
            if required_node not in self.__tags:
                raise ValueError('feed_dict requires {}.'.format(self.nodes_required))

        for node in self.nodes:
            self.__tags = node.build(self.__tags)

class With(yaml.YAMLObject):
    yaml_tag = u'!with'

    def __init__(self, nodes, variable=None, name=None, device=None):
        self.nodes = nodes

        if not variable and not name and not device:
            raise ValueError('variable, name, device are all None')

        self.variable = variable
        self.name = name
        self.device = device

    def __repr__(self):
        return 'With'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)
        nodes = yaml_dict['nodes']
        variable = yaml_dict.get('variable', None)
        name = yaml_dict.get('name', None)
        device = yaml_dict.get('device', None)
        return cls(nodes, variable=variable, name=name, device=device)

    def build(self, tags):
        vs = lambda x: tf.variable_scope(x) if x else WithNone()
        ns = lambda x: tf.name_scope(x) if x else WithNone()
        dv = lambda x: tf.device(x) if x else WithNone()

        with vs(self.variable), ns(self.name), dv(self.device):
            for node in self.nodes:
                tags = node.build(tags)

        return tags

class Conv2d(yaml.YAMLObject):
    yaml_tag = u'!conv2d'

    def __init__(self, tag, source, width, height, kernels_num, strides=[1,1,1,1],
        b_init=0.1, padding='SAME', name=None, variable_scope=None):

        self.tag = tag
        self.name = name
        self.variable_scope = variable_scope
        self.width = int(width)
        self.height = int(height)
        self.kernels_num = int(kernels_num)
        self.source = source
        self.strides= strides
        self.padding = padding
        self.b_init = float(b_init)

    def __repr__(self):
        return 'Conv2d'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'tag': yaml_dict.get('tag', None),
            'name': yaml_dict.get('name', None),
            'variable_scope': yaml_dict.get('variable_scope', None),
            'width': yaml_dict.get('width', 0),
            'height': yaml_dict.get('height', 0),
            'kernels_num': yaml_dict.get('kernels_num', 0),
            'source': yaml_dict.get('source', None),
            'strides': yaml_dict.get('strides', [1,1,1,1]),
            'padding': yaml_dict.get('padding', 'SAME'),
            'b_init': yaml_dict.get('b_init', 0.1),
        }

        assert args['tag'] not in loader.tags, \
            'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, args['tag'])
        assert args['width'] > 0, \
            'line {}: width must be > 0'.format(node.end_mark.line+1)
        assert args['height'] > 0, \
            'line {}: height must be > 0'.format(node.end_mark.line+1)
        assert args['kernels_num'] > 0, \
            'line {}: kernels_num must be > 0'.format(node.end_mark.line+1)
        assert args['source'], \
            'line {}: source is required'.format(node.end_mark.line+1)

        loader.tags[args['tag']] = node.end_mark

        return cls(**args)

    def build(self, tags):
        source_node = tags[self.source]
        channels = source_node.get_shape()[3]

        with tf.variable_scope(self.variable_scope) if self.variable_scope else WithNone():
            w = weight_variable(
                [self.height, self.width, channels,self.kernels_num], name="weight")
            b = bias_variable([self.kernels_num], val=self.b_init, name="bias")

            tags[self.tag] = tf.add( tf.nn.conv2d(
                source_node, w, strides=self.strides, padding=self.padding), b, name=self.name)

        return tags

class Conv2dTranspose(yaml.YAMLObject):
    yaml_tag = u'!conv2d_transpose'

    def __init__(self, tag, source, shape_as, width, height, strides=[1,1,1,1],
        b_init=0.1, padding='SAME', name=None, variable_scope=None):

        self.tag = tag
        self.name = name
        self.variable_scope = variable_scope
        self.source = source
        self.shape_as = shape_as
        self.width = width
        self.height = height
        self.strides= strides
        self.padding = padding
        self.b_init = b_init

    def __repr__(self):
        return 'Conv2dTranspose'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'tag': yaml_dict.get('tag', None),
            'name': yaml_dict.get('name', None),
            'variable_scope': yaml_dict.get('variable_scope', None),
            'width': yaml_dict.get('width', None),
            'height': yaml_dict.get('height', None),
            'source': yaml_dict.get('source', None),
            'shape_as': yaml_dict.get('shape_as', None),
            'strides': yaml_dict.get('strides', [1,1,1,1]),
            'padding': yaml_dict.get('padding', 'SAME'),
            'b_init': yaml_dict.get('b_init', 0.1),
        }

        assert args['tag'] not in loader.tags, \
            'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, args['tag'])
        assert args['width'] > 0, \
            'line {}: width must be > 0'.format(node.end_mark.line+1)
        assert args['height'] > 0, \
            'line {}: height must be > 0'.format(node.end_mark.line+1)
        assert args['source'], \
            'line {}: source is required'.format(node.end_mark.line+1)

        loader.tags[args['tag']] = node.end_mark

        return cls(**args)

    def build(self, tags):
        source_node = tags[self.source]
        shape_as = tags[self.shape_as]

        with tf.variable_scope(self.variable_scope) if self.variable_scope else WithNone():
            shape = source_node.get_shape()
            out_shape = shape_as.get_shape()
            w = tf.get_variable('weight',
                [self.height, self.width, out_shape[3], shape[3]],
                initializer=tf.truncated_normal_initializer(stddev=0.35))

            tags[self.tag] = tf.nn.conv2d_transpose(
                source_node, w, out_shape, strides=self.strides, padding=self.padding, name=self.name)

        return tags

class Conv2dAELoss(yaml.YAMLObject):
    yaml_tag = u'!conv2d_ae_loss'

    def __init__(self, tag, source1, source2, variable_scope=None, name=None):
        self.tag = tag
        self.name = name
        self.variable_scope = variable_scope
        self.source1 = source1
        self.source2 = source2

    def __repr__(self):
        return 'Conv2dAELoss'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'tag': yaml_dict.get('tag', None),
            'name': yaml_dict.get('name', None),
            'variable_scope': yaml_dict.get('variable_scope', None),
            'source1': yaml_dict.get('source1', None),
            'source2': yaml_dict.get('source2', None),
        }

        assert args['tag'] not in loader.tags, \
            'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, args['tag'])
        assert args['source1'], \
            'line {}: source1 is required'.format(node.end_mark.line+1)
        assert args['source2'], \
            'line {}: source2 is required'.format(node.end_mark.line+1)

        loader.tags[args['tag']] = node.end_mark

        return cls(**args)

    def build(self, tags):
        source1 = tags[self.source1]
        source2 = tags[self.source2]

        with tf.variable_scope(self.variable_scope) if self.variable_scope else WithNone():
            tags[self.tag] = tf.squared_difference(source1, source2, name=self.name)

        return tags

class AdamOptimizer(yaml.YAMLObject):
    yaml_tag = u'!adam_optimizer'

    def __init__(self, tag, source, val=1e-4, name=None):
        self.tag = tag
        self.name = name
        self.source = source
        self.val = val

    def __repr__(self):
        return 'AdamOptimizer'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'tag': yaml_dict.get('tag', None),
            'name': yaml_dict.get('name', None),
            'source': yaml_dict.get('source', None),
            'val': float(yaml_dict.get('val', 1e-4)),
        }

        assert args['tag'] not in loader.tags, \
            'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, args['tag'])
        assert args['source'], \
            'line {}: source is required'.format(node.end_mark.line+1)

        loader.tags[args['tag']] = node.end_mark

        return cls(**args)

    def build(self, tags):
        source_node = tags[self.source]

        global_step = tf.get_variable(
            'global_step', (),
            initializer=tf.constant_initializer(0), trainable=False)
        tags[self.tag] = tf.train.AdamOptimizer(self.val).minimize(
            source_node, global_step=global_step, name=self.name)

        return tags

class MaxPool2x2(yaml.YAMLObject):
    yaml_tag = u'!max_pool_2x2'

    def __init__(self, tag, source, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',
        name=None):

        self.tag = tag
        self.name = name
        self.source = source
        self.ksize = ksize
        self.strides = strides

    def __repr__(self):
        return 'MaxPool2x2'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'tag': yaml_dict.get('tag', None),
            'name': yaml_dict.get('name', None),
            'source': yaml_dict.get('source', None),
            'ksize': yaml_dict.get('ksize', [1,2,2,1]),
            'strides': yaml_dict.get('strides', [1,2,2,1]),
        }

        assert args['tag'] not in loader.tags, \
            'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, args['tag'])
        assert args['source'], \
            'line {}: source is required'.format(node.end_mark.line+1)

        loader.tags[args['tag']] = node.end_mark

        return cls(**args)

    def build(self, tags):
        source_node = tags[self.source]
        tags[self.tag] = tf.nn.max_pool(
            source_node, ksize=self.ksize, strides=self.strides,
            padding='SAME', name=self.name)
        return tags

class ReduceMean(yaml.YAMLObject):
    yaml_tag = u'!reduce_mean'

    def __init__(self, tag, source, reduction_indices=None, keep_dims=False, name=None):

        self.tag = tag
        self.name = name
        self.source = source
        self.reduction_indices = reduction_indices
        self.keep_dims = keep_dims

    def __repr__(self):
        return 'ReduceMean'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'tag': yaml_dict.get('tag', None),
            'name': yaml_dict.get('name', None),
            'source': yaml_dict.get('source', None),
            'reduction_indices': yaml_dict.get('reduction_indices', None),
            'keep_dims': yaml_dict.get('keep_dims', False),
        }

        assert args['tag'] not in loader.tags, \
            'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, args['tag'])
        assert args['source'], \
            'line {}: source is required'.format(node.end_mark.line+1)

        loader.tags[args['tag']] = node.end_mark

        return cls(**args)

    def build(self, tags):
        source_node = tags[self.source]
        tags[self.tag] = tf.reduce_mean(
            source_node, reduction_indices=self.reduction_indices,
            keep_dims=self.keep_dims, name=self.name)
        return tags

class ScalarSummary(yaml.YAMLObject):
    yaml_tag = u'!scalar_summary'

    def __init__(self, tag, summary_tag, source, name=None):

        self.tag = tag
        self.name = name
        self.summary_tag = summary_tag
        self.source = source

    def __repr__(self):
        return 'ScalarSummary'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'tag': yaml_dict.get('tag', None),
            'summary_tag': yaml_dict.get('summary_tag', ''),
            'name': yaml_dict.get('name', None),
            'source': yaml_dict.get('source', None),
        }

        assert args['tag'] not in loader.tags, \
            'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, args['tag'])
        assert args['source'], \
            'line {}: source is required'.format(node.end_mark.line+1)
        assert args['summary_tag'], \
            'line {}: summary_tag is required'.format(node.end_mark.line+1)

        loader.tags[args['tag']] = node.end_mark

        return cls(**args)

    def build(self, tags):
        source_node = tags[self.source]
        tags[self.tag] = tf.scalar_summary(
            self.summary_tag, source_node, name=self.name)
        return tags

class ImageSummary(yaml.YAMLObject):
    yaml_tag = u'!image_summary'

    def __init__(self, tag, summary_tag, source, max_images=3, name=None):

        self.tag = tag
        self.name = name
        self.summary_tag = summary_tag
        self.source = source
        self.max_images = max_images

    def __repr__(self):
        return 'ImageSummary'

    @classmethod
    def from_yaml(cls, loader, node):
        yaml_dict = loader.construct_mapping(node)

        args = {
            'tag': yaml_dict.get('tag', None),
            'summary_tag': yaml_dict.get('summary_tag', ''),
            'name': yaml_dict.get('name', None),
            'source': yaml_dict.get('source', None),
            'max_images': yaml_dict.get('max_images', 3),
        }

        assert args['tag'] not in loader.tags, \
            'line {}: tag "{}" is already exists'.format(node.end_mark.line+1, args['tag'])
        assert args['source'], \
            'line {}: source is required'.format(node.end_mark.line+1)
        assert args['summary_tag'], \
            'line {}: summary_tag is required'.format(node.end_mark.line+1)
        assert args['max_images'] > 0, \
            'line {}: max_images must be > 0'.format(node.end_mark.line+1)

        loader.tags[args['tag']] = node.end_mark

        return cls(**args)

    def build(self, tags):
        source_node = tags[self.source]
        tags[self.tag] = tf.image_summary(
            self.summary_tag, source_node, max_images=self.max_images, name=self.name)
        return tags

# ------------------------------------------------------------
# Loader function
# ------------------------------------------------------------

def load(path):
    graph= yaml.load(open(str(path)).read(), Loader=Loader)

    if type(graph['root']) is not Root:
        raise IOError("no Root in yaml file")

    return graph['root']

