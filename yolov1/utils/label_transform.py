import tensorflow as tf

class LabelTransform:
  def __init__(self,image_width = 448,image_height= 448,
               no_of_grids = 7.0,    no_of_classes = 10):
    self.image_width = image_width
    self.image_height = image_height
    self.no_of_grids = no_of_grids
    self.no_of_classes = no_of_classes

  def create_normalised_label(self,img, label):
    image = tf.image.decode_png(img,channels=3)
    image.set_shape((self.image_width,self.image_height,3))
    image = tf.cast(image,tf.float32)
    image = image / 255.0

    label_coordinates = (label[:,:4])
    label_coordinates = x1y1x2y2_to_xywh(label_coordinates)

    image_width = tf.cast(image_width,tf.float32)
    self.no_of_grids = tf.cast(self.no_of_grids,tf.float32)
    index_center = int(label_coordinates[:,:2]/image_width*self.no_of_grids)
    classes_no = int(label[:,4])
    classes = tf.one_hot(classes_no,depth=self.no_of_classes,dtype=tf.float32)
    confidence = tf.ones((len(label),1),tf.float32)
    
    out_label = tf.concat([confidence,label_coordinates,classes],axis=-1)
    
    self.no_of_grids = tf.cast(self.no_of_grids,tf.int64)
    output = tf.Variable(lambda : tf.zeros((*[no_of_grids,no_of_grids],5+self.no_of_classes)))
    output = tf.cast(output,tf.float32)
    stop = len(label)
    i = tf.constant(0,tf.int32)
    while tf.less(i, stop):
      output[int(index_center[i,1]),int(index_center[i,0])].assign(out_label[i,:])  
      i += 1
    image = tf.convert_to_tensor(image)
    output = tf.convert_to_tensor(output)
    return image,output
