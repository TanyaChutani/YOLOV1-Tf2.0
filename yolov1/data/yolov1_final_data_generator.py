import tensorflow as tf

def data_generator(files):
  dataset = tf.data.TFRecordDataset(filenames=files)
  dataset = dataset.map(convert_back, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.shuffle(8)
  dataset = dataset.batch(batch_size=4)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
