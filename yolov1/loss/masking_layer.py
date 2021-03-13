import tensorflow as tf


class MaskingLayer(tf.keras.layers.Layer):
  def __init__(self,no_of_boxes,
               image_dim,
               no_of_grids,
               **kwargs):
    super(Masking_Layer,self).__init__(name='denormalize_mask_layer',
                                         **kwargs)
    self.no_of_boxes = no_of_boxes
    self.image_dim = image_dim
    self.no_of_grids = no_of_grids

  @staticmethod
  def calculate_iou(box_1,box_2):
    x1 = tf.maximum(box_1[:,:,:,:,0],box_2[:,:,:,:,0])
    y1 = tf.maximum(box_1[:,:,:,:,1],box_2[:,:,:,:,1])
    x2 = tf.minimum(box_1[:,:,:,:,2],box_2[:,:,:,:,2])
    y2 = tf.minimum(box_1[:,:,:,:,3],box_2[:,:,:,:,3])
    
    intersection_area = tf.math.maximum(0.0,x2-x1)*tf.math.maximum(0.0,y2-y1)

    box1_area = (box_1[:,:,:,:,2]-box_1[:,:,:,:,0])*(box_1[:,:,:,:,3]-box_1[:,:,:,:,1])
    box2_area = (box_2[:,:,:,:,2]-box_2[:,:,:,:,0])*(box_2[:,:,:,:,3]-box_2[:,:,:,:,1])

    union_area = tf.math.maximum(1e-10,box1_area+box2_area-intersection_area)
    iou = intersection_area/union_area
    return tf.clip_by_value(iou,0.0,1.0)

  def meshgrid(self):
    mesh_grid = tf.stack(tf.meshgrid(tf.range(self.no_of_grids),tf.range(self.no_of_grids)),axis=-1)
    mesh_grid = tf.reshape(mesh_grid,[1,
                                      self.no_of_grids,self.no_of_grids,
                                      1,2])
    mesh_grid = tf.tile(mesh_grid,multiples=[1,1,1,self.no_of_boxes,1])
    mesh_grid = tf.cast(mesh_grid,tf.float32)
    return mesh_grid
  
  @staticmethod
  def xywh_to_x1y1x2y2(label):
    return tf.stack([label[...,0]-label[...,2]/2.0,
                    label[...,1]-label[...,3]/2.0,
                    label[...,0]+label[...,2]/2.0,
                    label[...,1]+label[...,3]/2.0],
                    axis=-1)
    
  def denormalize_box(self,box,mesh_grid):
    no_of_grids = 7
    no_of_grids = tf.cast(no_of_grids,tf.float32)
    box_center = (box[:,:,:,:,:2]+mesh_grid)/no_of_grids
    box_side = tf.square(box[:,:,:,:,2:])
    box_coord = tf.concat((box_center,box_side),axis=-1)
    return box_coord

  
  def call(self,y_true,y_pred):
    true_conf = y_true[:,:,:,0]
    true_conf = tf.expand_dims(true_conf,axis=-1)
    true_box = y_true[:,:,:,1:5]
    true_box = tf.reshape(true_box,[-1,y_true.shape[1],y_true.shape[1],1,4])
    true_box = tf.tile(true_box,multiples=[1,1,1,2,1])

    pred_box = y_pred[:,:,:,self.no_of_boxes:self.no_of_boxes*5]
    pred_box = tf.reshape(pred_box,[-1,pred_box.shape[1],
                                    pred_box.shape[1],
                                    self.no_of_boxes,4])
    mesh_grid = self.meshgrid()
    
    true_box_iou_coord = true_box/448.0
    pred_box_iou_coord = self.denormalize_box(pred_box,mesh_grid)
    
    true_box_x1y1x2y2 = xywh_to_x1y1x2y2(true_box_iou_coord)
    pred_box_x1y1x2y2 = xywh_to_x1y1x2y2(pred_box_iou_coord)

    iou_box = self.calculate_iou(true_box_x1y1x2y2,pred_box_x1y1x2y2)
    iou_gt = tf.reduce_max(iou_box,axis=3,keepdims=True)

    obj_mask = tf.cast(tf.equal(true_conf,1.0),tf.float32)
    box_mask = tf.cast((iou_box>=iou_gt),tf.float32)
    box_mask = box_mask*obj_mask
    return iou_box, box_mask, mesh_grid
