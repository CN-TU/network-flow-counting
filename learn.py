__author__ = "Fares Meghdouri"
__copyright__ = "Copyright 2021, TU Wien, CN group"
__license__ = "GPL V3.0"
__version__ = "1.0.0"
__maintainer__ = "Fares Meghdouri"
__email__ = "fares.meghdouri@tuwien.ac.at"
__status__ = "In Progress"

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, LeakyReLU
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from os import path
import tensorflow as tf
import shap
from collections import Counter
import random

# make sure the gpu is used
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FEATURES = ['ipTotalLength', 'iat', 'flowDirection']
N_FEATURES = len(FEATURES)
SEED = 2021
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def generate_sequences_count():
    sequences = []
    counter = 0
    storage = np.zeros((opt.window, N_FEATURES))
    checker = set()
    label_sequences = []
    sub_label_sequences = []
    
    for index, line in tqdm(data.iterrows()):
        if counter >= opt.window:
            sequences.append(storage)
            label_sequences.append(sub_label_sequences)
            sub_label_sequences = []
            storage = np.zeros((opt.window, N_FEATURES))
            checker = set()
            counter = 0

        # definition of a flow srcIP -> dstIP and backward direction
        if '{}{}'.format(line['destinationIPAddress'], line['sourceIPAddress']) not in checker:
            checker.add('{}{}'.format(line['sourceIPAddress'], line['destinationIPAddress']))

        sub_label_sequences.append(len(checker))

        storage[counter, 0] = line[FEATURES[0]] # length
        storage[counter, 1] = line[FEATURES[1]] # iat
        storage[counter, 2] = line[FEATURES[2]] # pktdirection
        counter += 1
    return sequences, label_sequences

def _scaling(x, __min, __max):
	return (x-__min)/(__max - __min)

def _back_scaling(x, __min, __max):
    return x * (__max - __min) + __min

def _log_scale(x):
	return np.log(x+1) # +1 to avoid 0's

def scale_count(sequences, labels, log=True):
	# 1. scale the sequences
	# all this because it has many features
	n_lines = sequences.shape[0]
	s = sequences.reshape((n_lines*opt.window, N_FEATURES))
	# 1.a log scale the iits because of the large range
	s[1][s[1] < 0] = 0 # fix negative iits
	if log:
		s[1] = _log_scale(s[1])
	# 1.b scale the rest of the features
	scalers = MinMaxScaler(feature_range=(0, 1))
	scaled_s = scalers.fit_transform(s)
	scaled_sequences = scaled_s.reshape((n_lines, opt.window, N_FEATURES))

	# 2. scale the labels
	l = labels.reshape((n_lines*opt.window, 1))
	scalerl = MinMaxScaler(feature_range=(0, 1))
	scaled_l = scalerl.fit_transform(l)
	scaled_labels = scaled_l.reshape((n_lines, opt.window))
	return scaled_sequences, scaled_labels, scalers, scalerl

def lstm_count_1():
	# first test network
	model = Sequential()
	model.add(LSTM(128, input_shape=(scaled_sequences.shape[1], scaled_sequences.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')
	return model

def lstm_count_2():
	# 2nd test network
	model = Sequential()
	model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(scaled_sequences.shape[1], scaled_sequences.shape[2])))
	model.add(LeakyReLU())
	model.add(Dropout(0.2))
	model.add(LSTM(16, return_sequences=True, activation='tanh'))
	model.add(Dropout(0.2))
	model.add(LeakyReLU())
	model.add(LSTM(8, return_sequences=True, activation='tanh'))
	model.add(LeakyReLU())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='mae', optimizer='adam')
	return model

def plot_distributions(model, X_test, y_test, scaler_l):
	# plot similarity histogram and marginal error
	# FIXME
	predicted_labels = _back_scaling(model.predict(X_test), scaler_l.data_min_, scaler_l.data_max_)
	true_labels = _back_scaling(y_test, scaler_l.data_min_, scaler_l.data_max_)
	plt.figure()
	plt.hist(np.round(predicted_labels[:,opt.window-1,:]), bins=opt.window, label='Predicted counts')
	plt.hist(true_labels[:,opt.window-1,:], bins=opt.window, alpha=0.7, label='Original counts')
	plt.xlabel('Flows in a window')
	plt.ylabel('Count')
	plt.legend()
	plt.savefig("count_performance_{}_{}.pdf".format(opt.window, datetime.now().strftime("%Y%m%d-%H%M%S")), bbox_inches='tight')

def get_metrics(model, X_test, y_test, scaler_l):
	predicted_labels = _back_scaling(model.predict(X_test), scaler_l.data_min_, scaler_l.data_max_)
	true_labels = _back_scaling(y_test, scaler_l.data_min_, scaler_l.data_max_)
	
	print('NMAE: {}'.format(np.sum(np.abs(predicted_labels[:,opt.window-1,:].flatten() - true_labels[:,opt.window-1,:].flatten())/true_labels[:,opt.window-1,:].flatten())/len(true_labels[:,opt.window-1,:].flatten())))
	print('NMAE-RG: {}'.format(np.sum(np.abs(random.choices(range(1, opt.window+1), k=true_labels.shape[0]) - true_labels[:,opt.window-1,:].flatten())/true_labels[:,opt.window-1,:].flatten())/len(true_labels[:,opt.window-1,:].flatten())))
	
	print('R2: {}'.format(r2_score(true_labels[:,opt.window-1,:].flatten(), predicted_labels[:,opt.window-1,:], multioutput='variance_weighted')))
	
	mae = mean_absolute_error(true_labels[:,opt.window-1,:].flatten(), predicted_labels[:,opt.window-1,:])
	#print('MSE: {}'.format(mse))
	#print('MSE/MSEMG: {}'.format(mae/mean_absolute_error(Counter(true_labels[:,opt.window-1,:].flatten()).most_common(1)[0][0]*np.ones(true_labels.shape[0]), true_labels[:,opt.window-1,:].flatten())))
	#print('MSE/MSERG: {}'.format(mae/mean_absolute_error(random.choices(true_labels[:,opt.window-1,:].flatten(), k=true_labels.shape[0]), true_labels[:,opt.window-1,:].flatten())))
	#print('MSE/MSERSG: {}'.format(mae/mean_absolute_error(random.choices(np.unique(true_labels[:,opt.window-1,:].flatten()), k=true_labels.shape[0]), true_labels[:,opt.window-1,:].flatten())))
	print('MAE/MAERSGA: {}'.format(mae/mean_absolute_error(true_labels[:,opt.window-1,:].flatten(), random.choices(range(1, opt.window+1), k=true_labels.shape[0]))))

def plot_perturbation(model, X, scaler_s):
	step = 1
	n_steps = 4
	_sum = np.sum([x for x in range(1, n_steps+1)])
	#sample = 0.9
	#X = X[int(X.shape[0]*sample):]
	local_importances = np.zeros((2, opt.window, X.shape[0]))
	global_importances = np.zeros((2, opt.window, X.shape[0]))
	p = model.predict(X)

	for feature in range(2):
	    for t in range(opt.window):
	        relative_error = np.zeros(X.shape[0])
	        final_error = np.zeros(X.shape[0])
	        for weight, change in enumerate(tqdm(np.arange(step, step*(n_steps+1), step))):
	            tmp = X.copy()
	            # consider that the features are log scalled when performing the change
	            # thus, scale also the change
	            tmp[:, t, feature] = X[:, t, feature] + np.log(1+change)/(scaler_s.data_max_[feature] - scaler_s.data_min_[feature])#*(1+change)
	            pred = model.predict(tmp)
	            #relative_error += np.abs(pred[:,t,0] - p[:,t,0])/p[:,t,0]
	            #final_error += np.abs(pred[:,-1,0] - p[:,-1,0])/p[:,-1,0]
	            # add the fact that the prediction can't be larger than the current timestamp
	            relative_error += np.abs(np.minimum(pred[:,t,0], t+1) - np.minimum(p[:,t,0], t+1))/np.minimum(p[:,t,0], t+1)*(n_steps-weight)
	            final_error += np.abs(np.minimum(pred[:,-1,0], t+1) - np.minimum(p[:,-1,0], t+1))/np.minimum(p[:,-1,0], t+1)*(n_steps-weight)
	        local_importances[feature, t,:] = relative_error/_sum*100
	        global_importances[feature, t,:] = final_error/_sum*100

	np.save('local_importances_{}_{}_{}_{}.npy'.format(step, n_steps, opt.window, datetime.now().strftime("%Y%m%d-%H%M%S")), local_importances)
	np.save('global_importances_{}_{}_{}_{}.npy'.format(step, n_steps, opt.window, datetime.now().strftime("%Y%m%d-%H%M%S")), global_importances)

	#plt.figure()
	x = np.arange(opt.window)
	f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,5))
	mean_1 = np.mean(local_importances[0,:,:], axis=1)
	std_1 = np.std(local_importances[0,:,:], axis=1)
	mean_2 = np.mean(local_importances[1,:,:], axis=1)
	std_2 = np.std(local_importances[1,:,:], axis=1)

	ax1.plot(x, mean_1, 'b-', label='pktSize')
	ax1.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
	ax1.plot(x, mean_2, 'r--', label='IAT')
	ax1.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
	ax1.set_yscale('log')
	ax1.set_title('Local Perturbation')
	ax1.set_xlabel('packet')
	ax1.set_ylabel('perturbation (%)')
	ax1.legend()

	mean_1 = np.mean(global_importances[0,:,:], axis=1)
	std_1 = np.std(global_importances[0,:,:], axis=1)
	mean_2 = np.mean(global_importances[1,:,:], axis=1)
	std_2 = np.std(global_importances[1,:,:], axis=1)
	ax2.plot(x, mean_1, 'b-', label='pktSize')
	ax2.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
	ax2.plot(x, mean_2, 'r--', label='IAT')
	ax2.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
	ax2.set_yscale('log')
	ax2.set_title('Global Perturbation')
	ax2.set_xlabel('packet')
	ax2.set_ylabel('perturbation (%)')
	ax2.legend()

	f.tight_layout(pad=3.0)
	plt.savefig("perturbations_{}_{}.pdf".format(opt.window, datetime.now().strftime("%Y%m%d-%H%M%S")), bbox_inches='tight')

def plot_shap_importance(model, X):
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()
	n,m = 100,100
	background = X_test[np.random.choice(X_test.shape[0], n, replace=False)]
	e = shap.DeepExplainer((model.layers[0].input, tf.math.reduce_sum(model.layers[-1].output, 1)), background)
	shap_values = e.shap_values(X_test[np.random.choice(X_test.shape[0], m, replace=False)], check_additivity=False)
	#TODO
	import tensorflow as tf

def fix_direction(x):
	return 0 if x else 1

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--task', required=True, help='task to execute')
	parser.add_argument('--dataroot', help='path to dataset')
	parser.add_argument('--window', type=int, default=100, help='window size to scan for number of flows')
	parser.add_argument('--function', help='the function that is going to be called')
	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--model', help='path to pre-trained model')
	parser.add_argument('--plot', help='which plot to create (coma separated) [hist, perturbations]')
	parser.add_argument('--external_data', action='store_true')
	parser.add_argument('--evaluate', action='store_true')

	opt = parser.parse_args()

	if opt.dataroot:
		print('>> Reading Data ########')
		data = pd.read_csv(opt.dataroot).fillna(0)
		# quick fix binary direction
		data[FEATURES[2]] = data[FEATURES[2]].apply(fix_direction)

	if opt.task == "count":
		if not path.exists('countv2/{}_sequences_count.npy'.format(opt.window)) or opt.external_data:
			print('>> Generating Sequences ########')
			sequences, label_sequences = generate_sequences_count()
			sequences = np.array(sequences)
			label_sequences = np.array(label_sequences)
			print('Average flows/seq: {}'.format(np.mean(label_sequences[:,-1])))
			if not opt.external_data:
				np.save('countv2/{}_sequences_count.npy'.format(opt.window), sequences)
				np.save('countv2/{}_label_sequences_count.npy'.format(opt.window), label_sequences)
			del data
		else:
			print('>> Loading Sequences ########')
			sequences = np.load('countv2/{}_sequences_count.npy'.format(opt.window))
			label_sequences = np.load('countv2/{}_label_sequences_count.npy'.format(opt.window))
		print('>> Scaling the Data ########')
		scaled_sequences, scaled_labels, scaler_s, scaler_l = scale_count(sequences, label_sequences)
		# fix the labels so that we learn at each timestamp (batch, ts, features)
		scaled_labels = scaled_labels.reshape(-1, opt.window, 1)
		X_train, X_test, y_train, y_test = train_test_split(scaled_sequences, scaled_labels, test_size=0.2, random_state=SEED)

		if opt.function == 'train':
			print('>> Preparing for Training ########')
			model = lstm_count_2()
			filepath = "countv2/saved-model-{epoch:02d}-{loss:.2f}-" + str(opt.window) + ".hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='auto', period=1)
			logdir = "../tf_logs/lstm/" + opt.task + str(opt.window) + datetime.now().strftime("%Y%m%d-%H%M%S")
			tensorboard_callback = TensorBoard(log_dir=logdir)
			model.fit(X_train, y_train, epochs=opt.epochs, batch_size=32, verbose=1, shuffle=False, callbacks=[checkpoint, tensorboard_callback])
		else:
			print('>> Loading a Pre-trained Model ########')
			model = load_model(opt.model)
		print('>> Running Tests ########')

		if opt.plot:
			if 'hist' in opt.plot:
				plot_distirbutions(model, X_test, y_test, scaler_l)
			if 'perturbations' in opt.plot:
				plot_perturbation(model, X_test, scaler_s)
		if opt.evaluate:
			get_metrics(model, X_test, y_test, scaler_l)
			# in case we want all the data and not only the test partition
			#get_metrics(model, scaled_sequences, scaled_labels, scaler_l)

	if opt.task == "separate":
		pass