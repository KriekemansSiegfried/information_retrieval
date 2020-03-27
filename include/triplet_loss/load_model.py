from tensorflow_core.python.keras.models import Model, model_from_json
from tensorflow_core.python.keras.saving.save import load_model


def load_submodels(model_path, weights_path):
    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)

    # create submodels
    caption_model = Model(inputs=model.get_layer('input_pos').input,
                          outputs=model.get_layer('output_pos').output)
    image_model = Model(inputs=model.get_layer('input_image').input, outputs=model.get_layer('output_image').output)
    caption_model.summary()
    image_model.summary()
    return caption_model, image_model
