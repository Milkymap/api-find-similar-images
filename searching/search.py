from utilities.utils import * 

if __name__ == '__main__':
	parser = get_parser()
	
	parser.add_argument('--input', help='path to input image', required=True)
	parser.add_argument('--target', help='path to target database', required=True)
	parser.add_argument('--extractor', help='path to [feautures] extractor', default='models/vgg16.h5')
	parser.add_argument('--features', help='path to features dump', default='features/features.pkl')

	parser_map = to_map(parser)

	logger.debug(' ... [searching] ... ')
	
	current_dir = get_location(__file__)

	path_to_target_database = path.join(current_dir, '..', parser_map['target'])
	path_to_input_image = path.join(current_dir, '..', parser_map['input'])
	path_to_extractor = path.join(current_dir, '..', parser_map['extractor'])
	path_to_features = path.join(current_dir, '..', parser_map['features'])
	
	print(path_to_target_database)
	print(path_to_input_image)

	
	vgg16 = load_vgg16(path_to_extractor)
	classifier = build_classifier()

	augmenter = get_augmenter()

	with open(path_to_features, 'rb') as fp:
		features_database = pickle.load(fp)

	image_paths = [path_to_input_image] * len(features_database)
	images = [ cv2.imread(p) for p in image_paths ]
	augmented_version = augmenter(images=images[1:])
	augmented_version.append(images[0])

	current_features = []
	for idx, mtx in enumerate(augmented_version):
		current_features.append(extract_features(mtx, vgg16))
		logger.success(f'image n° {idx: 03d} was processed')
	train_model(classifier, current_features, features_database)
	
	df = get_similar_images(classifier, vgg16, path_to_target_database)
	new_df = df[ df['status'] == True ]
	
	print(new_df)
	csv_filename = parser_map['input'].split('/')[-1]
	new_df.to_csv(f'csv_dump/{csv_filename}-similars.csv', sep=';')




