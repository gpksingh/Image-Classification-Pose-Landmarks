import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("yoga_pose_model.h5")

spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
output_path = "yoga_pose_model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path=output_path
)

print("Saved yoga_pose_model.onnx")