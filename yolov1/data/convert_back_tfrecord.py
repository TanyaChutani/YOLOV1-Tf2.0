import tensorflow as tf
from yolov1.utils.label_transform import LabelTransform

def convert_back(serialized):
  feature = {
      'image' : tf.io.FixedLenFeature([],tf.string),
      'x' : tf.io.VarLenFeature(tf.float32),
      'y' : tf.io.VarLenFeature(tf.float32),
      'w' : tf.io.VarLenFeature(tf.float32),
      'h' : tf.io.VarLenFeature(tf.float32),
      'class' : tf.io.VarLenFeature(tf.float32),
  }
  parsed_example = tf.io.parse_single_example(serialized=serialized,
                                              features=feature)
  labels = tf.stack([
                    tf.sparse.to_dense(parsed_example['x']),
                    tf.sparse.to_dense(parsed_example['y']),
                    tf.sparse.to_dense(parsed_example['w']),
                    tf.sparse.to_dense(parsed_example['h']),
                    tf.sparse.to_dense(parsed_example['class'])

  ],axis=-1)
  image, label = LabelTransform().create_normalised_label(img=parsed_example['image'],label=labels)

  return image,label
