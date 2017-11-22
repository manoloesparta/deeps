def load_model(file):
	
	from pathlib import Path
	my_file = Path(file + '.json')

	if my_file.is_file():

		from keras.models import model_from_json

		json_file = open(file + '.json')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights(file + '.h5')

		return loaded_model

if __name__ == "__main__":

	model = load_model('model')