from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):                                                   #**kwargs:keyworded variable length of arguments,named arguments
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()                                       #GCN class _build() is called.

        # region Build sequential layer model
        # Build sequential layer model
        '''
        Add layers in self.activations list: input(1433,16),hidden1(16,7),hidden2/Output(7,7)
        '''
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        # endregion

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()                                            #Call _loss() of GCN class.
        self._accuracy()                                        #Call _accuracy() of GCN class.
        self.opt_op = self.optimizer.minimize(self.loss)        #Optimise the loss and return none.

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(Model):
    '''
    GCN class for the GCN Model which inherits Model class defined above.
    '''
    def __init__(self, placeholders, input_dim, **kwargs):
        '''
        :param placeholders:
        :param input_dim:
        :param kwargs:
        This is the constructor of the class GCN
        '''
        # region Initialise attributes of the class
        super(GCN, self).__init__(**kwargs)                                             #Super class constructor call for name and logging.
        self.inputs = placeholders['features']                                          #Input to GCN is Feature Matrix X(2708,1433)
        self.input_dim = input_dim                                                      #Input or First Layer Dimension : 1433(Feature Dimension)
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]               #Output or Last Layer Dimension :7
        self.placeholders = placeholders                                                #Initialise placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)      #Initialise optimizer as Adam
        # endregion

        self.build()                                                                    #Build Model.Go to build() of Model Class.

    def _loss(self):
        '''
        This procedure will calculate loss of the model.
        '''
        # Weight decay loss (NOT SURE WHY NEEDED)
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],self.placeholders['labels_mask'])

    def _accuracy(self):
        '''
        This procedure will calculate the accuracy of the model.
        '''
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],self.placeholders['labels_mask'])

    def _build(self):
        '''
         This procedure will add Two Graph-Convolution Layers (Class in layer.py)
        '''
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))


        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
