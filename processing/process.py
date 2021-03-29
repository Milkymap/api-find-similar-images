from utilities.utils import * 

if __name__ == '__main__':
	parser = get_parser()
	parser.add_argument('--source_data', help='path to image source data', default='source_data')
	parser.add_argument('--extractor', help='path to [feautures] extractor', default='models/vgg16.h5')
	parser.add_argument('--features', help='path to features dump', default='features/features.pkl')
	parser_map = to_map(parser)
	logger.debug(parser_map)

	current_dir = get_location(__file__)
	path_to_source_data = path.join(current_dir, '..', parser_map['source_data'])
	path_to_features = path.join(current_dir, '..', parser_map['features'])
	path_to_vgg16 = path.join(current_dir, '..', parser_map['extractor'])

	directories = get_directories(path_to_source_data)
	logger.success(directories)

	filepaths = []
	for dr in directories:
		path_to_directory = path.join(path_to_source_data, dr)
		filepaths.append(pull_files(path_to_directory))

	filepaths = list(it.chain(*filepaths))
	vgg16 = load_vgg16(path_to_vgg16)
	print(vgg16.summary())

	accumulator = []
	for img_path in filepaths:
		if path.isfile(img_path):
			image = cv2.imread(img_path, cv2.IMREAD_COLOR)
			features = extract_features(image, vgg16)
			accumulator.append(features)
			logger.success(f'{path.split(img_path)[-1]} was processed ...!')

	with open(path_to_features, 'wb') as fp:
		pickle.dump(accumulator, fp)

	logger.success('all features were dump...!')
