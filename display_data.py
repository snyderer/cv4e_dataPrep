import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import utils.data_io as io

# Path containing dataset subfolders (each folder contains preprocessed .h5 files)
input_data_path = r'/mnt/class_data/esnyder/raw_data'

def process_preprocessed_file(h5path, outdir):
	fk_dehyd, timestamp = io.load_preprocessed_h5(h5path)

	settings_h5 = io.find_settings_h5(h5path)
	tx_data = None
	if settings_h5 is not None:
		s = io.load_settings_preprocessed_h5(settings_h5)
		if 'rehydration_info' in s:
			rehyd = s['rehydration_info']
			nonzeros = rehyd['nonzeros_mask']
			original_shape = rehyd['target_shape']
			try:
				tx_data = io.rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx')
			except Exception as e:
				print(f"Rehydration failed for {h5path}: {e}")

	if tx_data is None:
		print(f"No rehydrated time-domain data for {h5path}")
		return None, None

	# return the rehydrated time-domain data and timestamp for further processing
	return tx_data, timestamp


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


# NOTE: do not auto-run `iterate_datasets` on import/run —
# keep only the validation/pdf entry point at the end of this file.


# ---------------------------
# Label validation and F-X plotting
# ---------------------------
import csv
import random
from matplotlib.backends.backend_pdf import PdfPages


def _read_labels(csv_path):
	rows = []
	with open(csv_path, 'r', newline='') as f:
		reader = csv.DictReader(f)
		for r in reader:
			rows.append(r)
	return rows


def _group_by_dataset(rows):
	groups = {}
	for r in rows:
		key = r.get('dataset', 'unknown')
		groups.setdefault(key, []).append(r)
	return groups


def _sample_labels(rows, n=100):
	groups = _group_by_dataset(rows)
	n_groups = len(groups)
	per_group = max(1, int(np.ceil(n / max(1, n_groups))))

	sampled = []
	for k, g in groups.items():
		k_n = min(len(g), per_group)
		sampled.extend(random.sample(g, k_n))
	# if we overshot, trim randomly
	if len(sampled) > n:
		sampled = random.sample(sampled, n)
	# if undershot, fill with random draws from all rows
	if len(sampled) < n:
		remaining = [r for r in rows if r not in sampled]
		take = min(len(remaining), n - len(sampled))
		if take > 0:
			sampled.extend(random.sample(remaining, take))
	return sampled


def _find_h5_by_basename(basename, search_root):
	# recursive glob for the basename
	matches = glob.glob(os.path.join(search_root, '**', basename), recursive=True)
	return matches[0] if matches else None


def _map_label_path_to_local(source_file, input_root):
	"""Map Windows-style source_file paths in labels to local input_root.

	Examples:
	  F:\\dataset\\file.h5 -> /mnt/class_data/esnyder/raw_data/dataset/file.h5
	"""
	if not source_file:
		return None
	s = source_file.replace('\\', '/').strip()
	# remove drive letter if present (e.g. 'F:/')
	if ':' in s[:3]:
		# drop the drive (everything before the first slash)
		parts = s.split('/', 1)
		rest = parts[1] if len(parts) > 1 else parts[-1]
	else:
		rest = s.lstrip('/')

	candidate = os.path.join(input_root, rest)
	if os.path.isfile(candidate):
		return candidate

	# try removing common suffixes like '_full' from the first path segment
	parts = rest.split('/')
	if parts:
		first = parts[0]
		if first.endswith('_full'):
			parts2 = parts.copy()
			parts2[0] = first[:-5]
			candidate2 = os.path.join(input_root, *parts2)
			if os.path.isfile(candidate2):
				return candidate2

	# not found
	return None


def validate_labels_and_save_pdf(labels_csv, input_root, out_pdf='validation_samples.pdf', n_samples=100):
	rows = _read_labels(labels_csv)
	if not rows:
		raise RuntimeError('No labels found in ' + labels_csv)

	sampled = _sample_labels(rows, n=n_samples)

	# build per-source ranges for approximate spatial mapping
	source_ranges = {}
	for r in rows:
		src = os.path.basename(r.get('source_file', ''))
		try:
			xmin = float(r.get('x_min_m', 0))
			xmax = float(r.get('x_max_m', 0))
		except Exception:
			xmin, xmax = 0.0, 1.0
		if src not in source_ranges:
			source_ranges[src] = [xmin, xmax]
		else:
			source_ranges[src][0] = min(source_ranges[src][0], xmin)
			source_ranges[src][1] = max(source_ranges[src][1], xmax)

	with PdfPages(out_pdf) as pdf:
		for i, r in enumerate(sampled, 1):
			src_label_path = r.get('source_file', '')
			# basename used later for spatial mapping/plot titles — define it up front
			src_basename = os.path.basename(src_label_path)
			# prefer mapped local path (translate Windows F:\ paths)
			h5path = _map_label_path_to_local(src_label_path, input_root)
			if h5path is None:
				# fallback: search by basename inside input_root
				h5path = _find_h5_by_basename(src_basename, input_root)
			if h5path is None:
				print(f"[{i}] h5 not found for {src_label_path}, skipping")
				continue

			try:
				fk_dehyd, timestamp = io.load_preprocessed_h5(h5path)
			except Exception as e:
				print(f"[{i}] failed loading {h5path}: {e}")
				continue

			# load settings and rehydrate
			try:
				settings_file = io.find_settings_h5(h5path)
				if settings_file is None:
					raise RuntimeError("settings.h5 not found")
				s = io.load_settings_preprocessed_h5(settings_file)
			except Exception:
				s = {}

			tx_data = None
			try:
				if 'rehydration_info' in s:
					rehyd = s['rehydration_info']
					nonzeros = rehyd['nonzeros_mask']
					target_shape = rehyd['target_shape']
					# target_shape should be (nx, ns)
					nx = int(target_shape[0])
					ns = int(target_shape[1])
					amp = 1e9 * io.rehydrate(fk_dehyd, nonzeros, (nx, ns), return_format='tx')
					tx_data = amp
			except Exception as e:
				print(f"[{i}] rehydrate failed: {e}")

			if tx_data is None:
				print(f"[{i}] no tx data for {h5path}, skipping")
				continue

			nx, nt = tx_data.shape

			# build time and space axes from settings when possible
			dt = None
			fs = None
			dx = None
			if 'original_metadata' in s:
				om = s['original_metadata']
				fs = om.get('fs') or om.get('sample_rate')
				dx = om.get('dx')
				# some metadata store dt instead
				if 'dt' in om:
					dt = float(om['dt'])
			if fs is not None:
				try:
					fs = float(fs)
					dt = 1.0 / fs
				except Exception:
					dt = dt or 1.0
			dt = dt or 1.0
			if dx is None:
				dx = 1.0

			t_axis = np.arange(0, nt) * dt
			x_positions = np.arange(0, nx) * float(dx)

			start_t = float(r.get('t', 0.0))
			win_len = float(r.get('win_length_s', r.get('win_length', 2.0)))

			i0 = int(round(start_t / dt))
			i1 = min(nt, i0 + int(round(win_len / dt)))

			if i1 <= i0:
				print(f"[{i}] empty time window for label, skipping")
				continue

			tx_win = tx_data[:, i0:i1]

			# FFT along time
			nwin = tx_win.shape[1]
			fx = np.abs(np.fft.rfft(tx_win, axis=1))
			# frequency axis
			freqs = np.fft.rfftfreq(nwin, d=dt)

			# build approximate spatial axis (meters)
			src_range = source_ranges.get(src_basename, [0.0, float(nx - 1)])
			y_positions = np.linspace(src_range[0], src_range[1], nx)

			# bounding box from label
			fmin = float(r.get('f_min_hz', 0.0))
			fmax = float(r.get('f_max_hz', freqs[-1]))
			xmin_m = float(r.get('x_min_m', y_positions[0]))
			xmax_m = float(r.get('x_max_m', y_positions[-1]))

			# find frequency indices
			fmask = (freqs >= fmin) & (freqs <= fmax)
			if not np.any(fmask):
				print(f"[{i}] no freq bins in label range ({fmin}-{fmax} Hz), skipping")
				continue

			# find y indices
			ymask = (y_positions >= xmin_m) & (y_positions <= xmax_m)
			if not np.any(ymask):
				# try to proceed with full y range
				ymask = slice(None)

			fx_crop = fx[:, fmask]
			freqs_crop = freqs[fmask]

			# plot
			fig, ax = plt.subplots(figsize=(6, 4))
			im = ax.imshow(fx_crop, aspect='auto', origin='lower',
						   extent=[freqs_crop[0], freqs_crop[-1], y_positions[0], y_positions[-1]], 
						   vmin=0, vmax = 0.4)
			ax.set_xlabel('Frequency (Hz)')
			ax.set_ylabel('Distance (m)')
			ax.set_title(f"Sample {i}: {src_basename} t={start_t}s")
			fig.colorbar(im, ax=ax, label='Amplitude')

			# overlay bounding box
			rect_x0 = fmin
			rect_x1 = fmax
			rect_y0 = xmin_m
			rect_y1 = xmax_m
			ax.add_patch(plt.Rectangle((rect_x0, rect_y0), rect_x1 - rect_x0, rect_y1 - rect_y0,
									   edgecolor='red', facecolor='none', linewidth=1.5))

			pdf.savefig(fig)
			plt.close(fig)

	print('Saved validation PDF to', out_pdf)


if __name__ == '__main__':
	# Example use; adjust paths as needed
	labels_csv = os.path.join(os.path.dirname(__file__), 'fx_labels_Bp.csv')
	out_pdf = 'label_validation_samples.pdf'
	validate_labels_and_save_pdf(labels_csv, input_data_path, out_pdf=out_pdf, n_samples=100)
