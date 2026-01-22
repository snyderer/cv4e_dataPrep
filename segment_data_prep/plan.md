Steps:

1. load and rehydrate data
2. get settings required for m and s to pixels (min/max x and t)
3. select settings for tensor dimensions and data reduction
-FFT window length, Nfft, overlap
-x data reduction factor 
-masking settings: std (gaussian mask), tolerance (binary mask), 

4. Create 2D mask
5. Create 3D matrix
    a. select time window, fft
    b. reduce data dimensions in x and f
    c. step ahead to next window (step_size = window_length - overlap) and repeat
        Q: It's essentially spectrogram generation across all x. Could speed up or simplify w/ librosa?
6. Map mask to 3D matrix 
    Q: how will mask map to 3d tensor?
        (take mask points in x,t and map to x and t in tensor space) 

7. Save mask to ???? and tensor to ???? (data format? file structure?)

classes:
-loader: gets settings for dataset, iterates over files
    -set_settings
    -get_tx
    -get_labels_in_window
methods:
-generate_2D_mask (tx_img, x_lab, t_lab, {mask settings})
-map_mask_to_3D(mask_2D, xlim_orig, tlim_orig, xlim_out, tlim_out, Nfreq)
-restructure_data(tx_img, {restructuring params})

Need to think about going from T-X to T-X-F:
FFT length? Overlap? 

Data reduction: 
partially built-in to the FFT. Should probably do FFT at full resolution (Nfft = len(segment)) so I can retain narrowband energy. At a later point I could do a different Nfft specifically designed for blue whales. Could select bands of interest? 
