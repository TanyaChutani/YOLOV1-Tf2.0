import tensorflow as tf

class YoloV1(tf.keras.models.Model):
  def __init__(self):
    super(YoloV1,self).__init__()
    self.conv_block = Yolo_block(filters=16)
    self.conv1 = Conv_Block(1024,maxpool=False)
    self.conv2 = Conv_Block(1024,maxpool=False)
    self.out = self.out = tf.keras.layers.Conv2D(20,1)
    
  def call(self,input_tensor):
    x = self.conv_block(input_tensor)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.out(x)
    return x
