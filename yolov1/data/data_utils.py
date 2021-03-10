def xywh_to_x1y1x2y2(label):
  return tf.stack([label[...,0]-label[...,2]/2.0,
                   label[...,1]-label[...,3]/2.0,
                   label[...,0]+label[...,2]/2.0,
                   label[...,1]+label[...,3]/2.0],
                  axis=-1)


def x1y1x2y2_to_xywh(label):
  return tf.stack([(label[...,0]+label[...,2])/2.0,
                   (label[...,1]+label[...,3])/2.0,
                   label[...,2]-label[...,0],
                   label[...,3]-label[...,1]],
                  axis = -1)
