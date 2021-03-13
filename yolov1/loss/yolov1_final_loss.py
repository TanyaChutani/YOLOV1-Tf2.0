import tensorflow as tf


class YoloLoss(tf.keras.losses.Loss):
  def __init__(self,
               no_of_boxes=2,
               no_of_classes=10,
               image_dim=448,
               no_of_grids=7,
               scale_coord=tf.convert_to_tensor(5),
               scale_noobj=tf.convert_to_tensor(0.5),
      
               **kwargs):
    super(Yolo_loss,self).__init__(name='yolo_loss',
                                   **kwargs)
    self.no_of_boxes = no_of_boxes
    self.no_of_classes = no_of_classes
    self.image_dim = image_dim
    self.no_of_grids = no_of_grids
    self.scale_coord = scale_coord
    self.scale_noobj = scale_noobj
    self.mask_layer = Masking_Layer(no_of_boxes,image_dim,no_of_grids)
  
  def denormalize_box_true(self,box,mesh_grid):
    no_of_grids = 7
    no_of_grids = tf.cast(no_of_grids,tf.float32)
    box_center = box[:,:,:,:,:2]*no_of_grids-mesh_grid
    box_side = tf.sqrt(box[:,:,:,:,2:4])
    box_coord = tf.concat((box_center,box_side),axis=-1)
    return box_coord

  def classification_loss(self,true_class,pred_class,obj_mask):
    class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(obj_mask*(true_class-pred_class)),axis=[1,2,3]))
    return class_loss
  
  def confidence_loss(self,pred_conf,box_mask,iou_box,noobj_mask):
    conf_loss_obj = tf.reduce_mean(tf.reduce_sum(tf.square(box_mask*(iou_box-pred_conf)),axis=[1,2,3]))
    conf_loss_noobj = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_mask*(pred_conf)),axis=[1,2,3]))
    return conf_loss_obj, conf_loss_noobj

  def localization_loss(self,true_box,pred_box,box_mask,mesh_grid):
    box_mask = tf.expand_dims(box_mask,-1)
    loc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(box_mask*(true_box-pred_box)),axis=[1,2,3,4]))
    return loc_loss

  def call(self,y_true,y_pred):
    true_conf = y_true[:,:,:,0]
    true_conf = tf.expand_dims(true_conf,axis=-1)
    true_box = y_true[:,:,:,1:5]
    true_box = tf.reshape(true_box,[-1,y_true.shape[1],y_true.shape[1],1,4])
    true_box = tf.tile(true_box,multiples=[1,1,1,2,1])
    true_class = y_true[:,:,:,5:]
    true_box = true_box/448.0
    
    pred_conf = y_pred[:,:,:,:self.no_of_boxes]
    pred_box = y_pred[:,:,:,self.no_of_boxes:self.no_of_boxes*5]
    pred_box = tf.reshape(pred_box,[-1,pred_box.shape[1],
                                    pred_box.shape[1],
                                    self.no_of_boxes,4])
    pred_class = y_pred[:,:,:,self.no_of_boxes*5:]   
    
    iou_box, box_mask, mesh_grid = self.mask_layer(y_true,y_pred)
    true_box = self.denormalize_box_true(true_box,mesh_grid)

    obj_mask = tf.cast(tf.equal(true_conf,1.0),tf.float32)
    noobj_mask = tf.cast(tf.ones_like(box_mask) - box_mask,tf.float32)
    class_loss = self.classification_loss(true_class,pred_class,obj_mask)
    conf_loss_obj, conf_loss_noobj = self.confidence_loss(pred_conf,box_mask,iou_box,noobj_mask)
    loc_loss = self.localization_loss(true_box,pred_box,box_mask,mesh_grid)
    return (loc_loss+conf_loss_obj+(0.5*conf_loss_noobj)+class_loss)
