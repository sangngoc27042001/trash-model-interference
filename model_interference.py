import tensorflow as tf
import cv2

class AIModel:
  def __init__(self,model_file_path):
    self.model = tf.keras.models.load_model(model_file_path)
    self.classes = ['ALU', 'FOAM_BOX', 'MILKBOX', 'PAPER_CUP', 'PET']
    self.prob_thresh = 0.4
  def predict(self, rgb_image):
    rgb_image = cv2.resize(rgb_image, (224,224))[:,:,0:3]
    result = self.model.predict(rgb_image.reshape(1,224,224,3), verbose = False)[0]
    predicted_class, predicted_prob = self.classes[result.argmax()], result.max()
    if (predicted_class not in ['ALU', 'MILKBOX', 'PET']) or (predicted_prob < self.prob_thresh):
      predicted_class = 'OTHER'
      predicted_prob = result[1] + result[3]
    return {
        'class': predicted_class,
        'probability': predicted_prob
    }
