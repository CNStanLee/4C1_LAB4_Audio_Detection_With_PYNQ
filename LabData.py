import os
import tarfile
import urllib.request
from pathlib import Path
import random

import numpy as np
import librosa
from tqdm import tqdm

from scipy import signal
from python_speech_features import mfcc
from scipy.signal.windows import hann


root = '.'
root_path = Path(root)
data_root = root_path / "data"
sc_version = "speech_commands_v0.02"
sc_url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

full_data_root = data_root / sc_version

LABEL_NAMES = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",
    "silence", "unknown",
]
TARGET_WORDS = LABEL_NAMES[:10]
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"

SR = 16000        
DURATION = 1.0     


tf_desired_samples = 16000
tf_window_size_samples = 480
tf_sample_rate = 16000
tf_window_size_ms = 30.
tf_window_stride_ms = 20.
tf_dct_coefficient_count = 10  


TARGET_NUM_FRAMES = 49

MAX_UNKNOWN_PER_SPLIT = 3000
N_UNKNOWN_TRAIN = MAX_UNKNOWN_PER_SPLIT
N_UNKNOWN_VAL = 400
N_UNKNOWN_TEST = 400

N_SILENCE_TRAIN = 3000
N_SILENCE_VAL = 400
N_SILENCE_TEST = 400

OUT_NPZ = data_root / "kws_12cls_mfcc_10x49_quant_flat.npz"


def download_and_extract_speech_commands():
    data_root.mkdir(parents=True, exist_ok=True)
    tar_path = data_root / f"{sc_version}.tar.gz"
    sc_root = data_root / sc_version  

    if sc_root.exists():
        print(f"[INFO] data dir exists: {sc_root}")
        return sc_root

    if not tar_path.exists():
        print(f"[INFO] downloading Speech Commands v0.02 tar to: {tar_path}")
        urllib.request.urlretrieve(sc_url, tar_path)
        print(f"[INFO] download completed.")

    print(f"[INFO] extracting to: {sc_root}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=sc_root)
    print(f"[INFO] extraction completed: {sc_root}")

    return sc_root


def read_list_file(list_path: Path):
    s = set()
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                s.add(line)
    return s


def py_speech_preprocessing(raw_signal,
                            sample_rate,
                            tf_desired_samples=tf_desired_samples,
                            tf_window_size_samples=tf_window_size_samples,
                            tf_sample_rate=tf_sample_rate,
                            tf_window_size_ms=tf_window_size_ms,
                            tf_window_stride_ms=tf_window_stride_ms,
                            tf_dct_coefficient_count=tf_dct_coefficient_count,
                            target_num_frames=TARGET_NUM_FRAMES):


    if sample_rate != tf_sample_rate:
        num_target_samples = round(tf_sample_rate / sample_rate * len(raw_signal))
        resampled_data = signal.resample(raw_signal, num_target_samples)
    else:
        resampled_data = raw_signal

    max_abs = np.max(np.abs(resampled_data))
    if max_abs > 0:
        rescaled_data = resampled_data / max_abs
    else:
        rescaled_data = resampled_data

    if rescaled_data.shape[-1] < tf_desired_samples:
        padded_data = np.pad(
            rescaled_data,
            (0, tf_desired_samples - rescaled_data.shape[-1]),
            mode="constant",
        )
    else:
        padded_data = rescaled_data[:tf_desired_samples]

    nfft = int(2 ** np.ceil(np.log2(tf_window_size_samples)))  

    mfcc_feat_py = mfcc(
        padded_data,
        samplerate=tf_sample_rate,
        winlen=tf_window_size_ms / 1000.,
        winstep=tf_window_stride_ms / 1000.,
        numcep=tf_dct_coefficient_count,
        nfilt=40,
        nfft=nfft,
        lowfreq=20.0,
        highfreq=4000.0,
        winfunc=hann,
        appendEnergy=False,
        preemph=0.,
        ceplifter=0.,
    )  # shape: (num_frames, numcep)

    mfcc_feat_py = mfcc_feat_py.T.astype(np.float32)  # (10, F)


    num_frames = mfcc_feat_py.shape[1]
    if num_frames < target_num_frames:

        pad_width = target_num_frames - num_frames
        mfcc_feat_py = np.pad(
            mfcc_feat_py,
            ((0, 0), (0, pad_width)),
            mode="edge",
        )
    elif num_frames > target_num_frames:

        mfcc_feat_py = mfcc_feat_py[:, :target_num_frames]

    return mfcc_feat_py


def quantize_input(mfcc_feat_py):

    quant_mfcc_feat = mfcc_feat_py / 0.8298503756523132


    quant_mfcc_feat = np.clip(quant_mfcc_feat, -127., 127.)


    quant_mfcc_feat = np.round(quant_mfcc_feat).astype(np.int8)


    quant_mfcc_feat = quant_mfcc_feat.reshape(-1)

    return quant_mfcc_feat


def wav_to_quantized_feat(wav_path: Path):


    y, sr = librosa.load(wav_path, sr=SR)  
    mfcc_feat_py = py_speech_preprocessing(y, sample_rate=sr)
    feat_q = quantize_input(mfcc_feat_py)
    return feat_q

if __name__ == "__main__":
    download_and_extract_speech_commands()

    val_list = read_list_file(full_data_root / "validation_list.txt")
    test_list = read_list_file(full_data_root / "testing_list.txt")

    label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
    print("[INFO] label to index mapping:")
    for k, v in label_to_idx.items():
        print(f"  {k:>8s} -> {v}")

    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []

    print("\n[INFO] dealing with target words ...")
    for word in TARGET_WORDS:
        word_dir = full_data_root / word
        if not word_dir.exists():
            print(f"[WARN] path does not exist, skipping: {word_dir}")
            continue

        wav_files = sorted(word_dir.glob("*.wav"))
        print(f"[INFO] {word}: {len(wav_files)} samples found")

        for wav_path in tqdm(wav_files, desc=f"{word:>5s}", ncols=80):
            rel_path = wav_path.relative_to(full_data_root).as_posix()

            if rel_path in val_list:
                subset = "val"
            elif rel_path in test_list:
                subset = "test"
            else:
                subset = "train"

            feat = wav_to_quantized_feat(wav_path)  # (490,) int8
            label_idx = label_to_idx[word]

            if subset == "train":
                X_train.append(feat)
                y_train.append(label_idx)
            elif subset == "val":
                X_val.append(feat)
                y_val.append(label_idx)
            else:
                X_test.append(feat)
                y_test.append(label_idx)

    print("\n[INFO] gathering unknown class samples ...")
    all_dirs = [d for d in full_data_root.iterdir() if d.is_dir()]
    unknown_dirs = [
        d for d in all_dirs
        if d.name not in TARGET_WORDS
        and d.name not in ["_background_noise_"]
        and not d.name.startswith(".")
    ]

    unknown_train_files, unknown_val_files, unknown_test_files = [], [], []
    for d in unknown_dirs:
        wav_files = sorted(d.glob("*.wav"))
        for wav_path in wav_files:
            rel_path = wav_path.relative_to(full_data_root).as_posix()
            if rel_path in val_list:
                unknown_val_files.append(wav_path)
            elif rel_path in test_list:
                unknown_test_files.append(wav_path)
            else:
                unknown_train_files.append(wav_path)

    print(
        f"[INFO] unknown ori num: "
        f"train={len(unknown_train_files)}, "
        f"val={len(unknown_val_files)}, "
        f"test={len(unknown_test_files)}"
    )

    def sample_files(file_list, max_n):
        if len(file_list) > max_n:
            return random.sample(file_list, max_n)
        return file_list

    unknown_train_files = sample_files(unknown_train_files, N_UNKNOWN_TRAIN)
    unknown_val_files   = sample_files(unknown_val_files,   N_UNKNOWN_VAL)
    unknown_test_files  = sample_files(unknown_test_files,  N_UNKNOWN_TEST)

    print(
        f"[INFO] unknown sampled: "
        f"train={len(unknown_train_files)}, "
        f"val={len(unknown_val_files)}, "
        f"test={len(unknown_test_files)}"
    )

    unk_idx = label_to_idx[UNKNOWN_LABEL]

    for split_name, file_list in [
        ("train", unknown_train_files),
        ("val",   unknown_val_files),
        ("test",  unknown_test_files),
    ]:
        print(f"[INFO] extracting unknown {split_name} features, total {len(file_list)}")
        for wav_path in tqdm(file_list, desc=f"unk-{split_name}", ncols=80):
            feat = wav_to_quantized_feat(wav_path)  # (490,) int8
            if split_name == "train":
                X_train.append(feat)
                y_train.append(unk_idx)
            elif split_name == "val":
                X_val.append(feat)
                y_val.append(unk_idx)
            else:
                X_test.append(feat)
                y_test.append(unk_idx)


    print("\n[INFO] generating silence samples ...")
    noise_dir = full_data_root / "_background_noise_"
    noise_files = sorted(noise_dir.glob("*.wav"))
    if not noise_files:
        print("[WARN] not found _background_noise_/*.wav, silence class will be empty")
        N_sil_train = N_SILENCE_TRAIN = 0
        N_sil_val   = N_SILENCE_VAL   = 0
        N_sil_test  = N_SILENCE_TEST  = 0
    else:
        noise_clips = []
        for nf in noise_files:
            y, sr = librosa.load(nf, sr=SR)
            noise_clips.append(y)

        def gen_silence_samples(n_samples, desc):
            feats = []
            desired_len = int(SR * DURATION)
            for _ in tqdm(range(n_samples), desc=desc, ncols=80):
                y = random.choice(noise_clips)
                if len(y) > desired_len:
                    start = random.randint(0, len(y) - desired_len)
                    y_seg = y[start:start + desired_len]
                else:
                    y_seg = np.pad(y, (0, desired_len - len(y)))
                mfcc_feat_py = py_speech_preprocessing(y_seg, sample_rate=SR)
                feat = quantize_input(mfcc_feat_py)  # (490,) int8
                feats.append(feat)
            return feats

        sil_idx = label_to_idx[SILENCE_LABEL]

        sil_train = gen_silence_samples(N_SILENCE_TRAIN, "sil-train")
        sil_val   = gen_silence_samples(N_SILENCE_VAL,   "sil-val")
        sil_test  = gen_silence_samples(N_SILENCE_TEST,  "sil-test")

        for feat in sil_train:
            X_train.append(feat)
            y_train.append(sil_idx)
        for feat in sil_val:
            X_val.append(feat)
            y_val.append(sil_idx)
        for feat in sil_test:
            X_test.append(feat)
            y_test.append(sil_idx)

    out_npz = OUT_NPZ

    # X_*: (N, 490), int8
    X_train = np.stack(X_train, axis=0).astype(np.int8)
    X_val   = np.stack(X_val,   axis=0).astype(np.int8)
    X_test  = np.stack(X_test,  axis=0).astype(np.int8)

    # y_*: int16
    y_train = np.array(y_train, dtype=np.int16)
    y_val   = np.array(y_val,   dtype=np.int16)
    y_test  = np.array(y_test,  dtype=np.int16)

    print("\n[INFO] dataset shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val  :", X_val.shape,   "y_val  :", y_val.shape)
    print("  X_test :", X_test.shape,  "y_test :", y_test.shape)

    np.savez_compressed(
        out_npz,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_val,
        y_valid=y_val,
        X_test=X_test,
        y_test=y_test,
        label_names=np.array(LABEL_NAMES),
    )
    print(f"[DONE] preprocessed data saved to: {out_npz}")

    def inspect_npz(npz_path: Path):
        print(f"\n[INFO] checking: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val   = data["X_valid"]
        y_val   = data["y_valid"]
        X_test  = data["X_test"]
        y_test  = data["y_test"]
        label_names = data["label_names"]

        print("\n[INFO] label_names:", label_names)
        print("\n[INFO] shape checking:")
        print("  X_train:", X_train.shape, "y_train:", y_train.shape, "dtype:", X_train.dtype)
        print("  X_val  :", X_val.shape,   "y_val  :", y_val.shape,   "dtype:", X_val.dtype)
        print("  X_test :", X_test.shape,  "y_test :", y_test.shape,  "dtype:", X_test.dtype)
        print("  Y_train: ", y_train.shape, "dtype:", y_train.dtype)

        def print_label_stats(name, y):
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n  {name} label distribution:")
            for u, c in zip(unique, counts):
                lbl = label_names[u] if u < len(label_names) else "NA"
                print(f"    idx={u:2d} ({lbl:8s}): {c:6d}")

        print_label_stats("Train", y_train)
        print_label_stats("Val",   y_val)
        print_label_stats("Test",  y_test)

    inspect_npz(OUT_NPZ)
