import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import data_io as io

# Path containing dataset subfolders (each folder contains preprocessed .h5 files)
input_data_path = r'/mnt/class_data/esnyder/raw_data'

settings = {'twin_s': 2}

def process_preprocessed_file(h5path, outdir):
	fk_dehyd, timestamp = io.load_preprocessed_h5(h5path)

	settings_h5 = io.find_settings_h5(h5path)
	tx_data = None
	if settings_h5 is not None:
		s = io.load_settings_preprocessed_h5(h5path)
		if 'rehydration_info' in s:
			rehyd = s['rehydration_info']
			nonzeros = rehyd['nonzeros_mask']
			original_shape = rehyd['target_shape']
			try:
				tx_data = io.rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx')
			except Exception as e:
				print(f"Rehydration failed for {h5path}: {e}")
	if tx_data is None:
		print(f"Skipping plotting for {h5path} (no rehydrated time-domain data)")
		return

	os.makedirs(outdir, exist_ok=True)
	base = os.path.splitext(os.path.basename(h5path))[0]

	# Image of the data (space x time)
	plt.figure(figsize=(8, 4))
	plt.imshow(tx_data, aspect='auto', origin='lower')
	plt.colorbar(label='Amplitude')
	plt.title(f'{base} — time-domain (space x time)')
	plt.xlabel('Time sample')
	plt.ylabel('Space index')
	imgpath = os.path.join(outdir, f"{base}_image.png")
	plt.tight_layout()
	plt.savefig(imgpath)
	plt.close()

	# Mean time series across space
	mean_ts = np.mean(tx_data, axis=0)
	plt.figure(figsize=(8, 3))
	plt.plot(mean_ts)
	plt.title(f'{base} — mean time series')
	plt.xlabel('Time sample')
	plt.ylabel('Mean amplitude')
	tspath = os.path.join(outdir, f"{base}_timeseries.png")
	plt.tight_layout()
	plt.savefig(tspath)
	plt.close()


def iterate_datasets(input_path, out_root=None, pattern='*.h5'):
	if out_root is None:
		out_root = os.path.join(input_path, 'plots')

	# iterate subdirectories directly under input_path
	for entry in sorted(os.listdir(input_path)):
		folder = os.path.join(input_path, entry)
		if not os.path.isdir(folder):
			continue

		# find h5 files in this dataset folder
		files = sorted(glob.glob(os.path.join(folder, pattern)))
		if not files:
			print(f"No '{pattern}' files in {folder}, skipping")
			continue

		dataset_out = os.path.join(out_root, entry)
		for fpath in files:
			print(f"Processing {fpath}")
			process_preprocessed_file(fpath, dataset_out)


if __name__ == '__main__':
	iterate_datasets(input_data_path)
