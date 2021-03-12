import tensorflow as tf

class MakeTfrecords:
  def __init__(self,image_path,label_path,out_path):
    self.image_path = image_path
    self.label_path = label_path
    self.out_path = out_path
    
  @staticmethod
  def read_image(img):
    with tf.io.gfile.GFile(img,'rb') as file:
      image = file.read()
    return image
  
  @staticmethod
  def wrap_bytes(img):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
  
  @staticmethod
  def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def convert_tf_record(self,image,label,out_paths):
    with tf.io.TFRecordWriter(out_paths) as writer:
        output = InputDataTransform()
        label = output.transform_labels(label)
        if not label:
          pass
        else:
          image_bytes = self.read_image(image)
          ftr = {
              'image' : self.wrap_bytes(image_bytes),
              'x' : self.wrap_float(np.array(label)[:,0]),
              'y' : self.wrap_float(np.array(label)[:,1]),
              'w' : self.wrap_float(np.array(label)[:,2]),
              'h' : self.wrap_float(np.array(label)[:,3]),
              'class' : self.wrap_float(np.array(label)[:,4]),
          }
          feature = tf.train.Features(feature=ftr)
          example = tf.train.Example(features=feature)
          serialized = example.SerializeToString()
          writer.write(serialized)
  
  def write_tfrecord(self):
    for img,label in zip(sorted(os.listdir(self.image_path)),sorted(os.listdir(self.label_path))):
      path_image = os.path.join(self.image_path,str(img))
      path_label = os.path.join( self.label_path,str(label))
      self.convert_tf_record(image=path_image,
                             label=path_label,
                             out_paths=self.out_path)
