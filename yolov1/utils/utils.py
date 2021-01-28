def meshgrid(no_of_grids,no_of_boxes):
  mesh_grid = tf.stack(tf.meshgrid(tf.range(no_of_grids),tf.range(no_of_grids)),axis=-1)
  mesh_grid = tf.reshape(mesh_grid,[1,
                                    no_of_grids,no_of_grids,
                                    1,2])
  mesh_grid = tf.tile(mesh_grid,multiples=[1,1,1,no_of_boxes,1])
  mesh_grid = tf.cast(mesh_grid,tf.float32)
  return mesh_grid
