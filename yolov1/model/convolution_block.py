import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
  def __init__(self,filters,
               kernel_size=3,
               stride=1,
               kernel_initializer='he_normal',
               padding = 'same',
               maxpool = True,
               pool_size=2):
    super(Conv_Block,self).__init__()
    self.conv_filters = filters
    self.conv_kernel_size = kernel_size
    self.conv_stride = stride
    self.conv_kernel_initializer = kernel_initializer
    self.conv_padding = padding
    self.maxpool = maxpool
    self.max_pool_size = pool_size
    self.conv_layer = tf.keras.layers.Conv2D(self.conv_filters,
                                             self.conv_kernel_size,
                                             strides=self.conv_stride,
                                             padding=self.conv_padding,
                                             kernel_initializer=\
                                             self.conv_kernel_initializer)
    self.bn = tf.keras.layers.BatchNormalization()
    self.l_relu = tf.keras.layers.LeakyReLU(0.01)
    self.max_pool = tf.keras.layers.MaxPool2D(pool_size=self.max_pool_size)
  def call(self,input_tensor):
    x = self.conv_layer(input_tensor)
    x = self.bn(x)
    x = self.l_relu(x)
    if self.maxpool:
      x = self.max_pool(x)
    return x
