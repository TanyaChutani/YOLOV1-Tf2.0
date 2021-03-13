import tensorflow as tf


class YoloBlock(tf.keras.layers.Layer):
  def __init__(self,filters,
               maxpool = True):
    super(Yolo_block,self).__init__()
    self.filters = filters
    self.maxpool = maxpool
    self.conv1 = Conv_Block(filters)
    self.conv2 = Conv_Block(filters*2)
    self.conv3 = Conv_Block(filters*4)
    self.conv4 = Conv_Block(filters*8)
    self.conv5 = Conv_Block(filters*16)
    self.conv6 = Conv_Block(filters*32)
    self.conv7 = Conv_Block(filters*64,maxpool=False)
    self.make_block = self.make_blocks()
    
  def call(self,input_tensor):
    return self.make_block(input_tensor)

  def make_blocks(self):
    label = []
    label.append(self.conv1)
    label.append(self.conv2)
    label.append(self.conv3)
    label.append(self.conv4)
    label.append(self.conv5)
    label.append(self.conv6)
    label.append(self.conv7)
    return tf.keras.Sequential(label)
