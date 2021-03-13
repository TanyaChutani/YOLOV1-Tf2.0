import tensorflow as tf

class Visualizer:
  def __init__(self,input,color=(255,0,0),thickness=3):
    self.input = input
    self.color = color
    self.thickness = thickness

  def show_train_image(self):
    image, label = self.input[0], self.input[1]
    image = cv2.imread(image)
    output = InputDataTransform()
    label = output.transform_labels(label)
    for val in label:
      start = (int(val[0]),int(val[1]))
      end = (int(val[2]),int(val[3]))
      cv2.rectangle(image,start,end,self.color,self.thickness)
    plt.imshow(image)
    plt.show()
  
  def show_test_image(self):
    image, label = self.input[0], self.input[1]
    image=np.uint8(image)
    for val in label:
      start = (int(val[0]),int(val[1]))
      end = (int(val[2]),int(val[3]))
      cv2.rectangle(image,start,end,self.color,self.thickness)
    plt.imshow(image)
    plt.show()
