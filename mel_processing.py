# 이 문서는 OpenVoice의 mel_processing 부분을 설명한다.


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)
# 위 코드는 입력 x의 동적 범위를 압축한다.
# 입력 x의 동적 범위를 압축하는 것은 이후 작업(신호처리)을 더 원활하게 해준다.
# C는 압축비율, clip_val은 최소값이다. clip_val을 설정하는 이유는 로그 연산에서 발생할 수 있는 방지하기 위해서이다.


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C
# 위 코드는 압축된 데이터를 원래의 동적 범위로 복원한다.


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output
# 위 코드는 스펙트로그램의 값을 정규화한다.
# 스펙트로그램의 값들이 일정한 범위에 들어오면 후속 처리에 좋고 모델 학습이 더 잘 된다.


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output
# 위 코드는 정규화 했던 스펙트로그램 값을 원래 형태로 되돌린다(해제).
# 이 메서드의 사용은 결과를 해석할 때 필요할 수도 있다.


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.1:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.1:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec
# 위 코드는 주어진 오디오 신호 y에 대해 스펙트로그램을 계산한다.
# 신호를 패딩하고, FFT를 적용하여 스펙트로그램을 생성한다.
# 반환값은 주파수 성분의 크기를 나타내는 스펙트로그램이다.


def spectrogram_torch_conv(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    
    # ******************** original ************************#
    # y = y.squeeze(1)
    # spec1 = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
    #                   center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    # ******************** ConvSTFT ************************#
    freq_cutoff = n_fft // 2 + 1
    fourier_basis = torch.view_as_real(torch.fft.fft(torch.eye(n_fft)))
    forward_basis = fourier_basis[:freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
    forward_basis = forward_basis * torch.as_tensor(librosa.util.pad_center(torch.hann_window(win_size), size=n_fft)).float()

    import torch.nn.functional as F

    # if center:
    #     signal = F.pad(y[:, None, None, :], (n_fft // 2, n_fft // 2, 0, 0), mode = 'reflect').squeeze(1)
    assert center is False

    forward_transform_squared = F.conv1d(y, forward_basis.to(y.device), stride = hop_size)
    spec2 = torch.stack([forward_transform_squared[:, :freq_cutoff, :], forward_transform_squared[:, freq_cutoff:, :]], dim = -1)


    # ******************** Verification ************************#
    spec1 = torch.stft(y.squeeze(1), n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    assert torch.allclose(spec1, spec2, atol=1e-4)

    spec = torch.sqrt(spec2.pow(2).sum(-1) + 1e-6)
    return spec
# 위 코드는 spectrogram_torch의 변형으로, 컨볼루션을 사용하여 스펙트로그램을 계산한다.
# FFT 기반의 변환을 사용하여 성능을 개선한다.
# 코드가 spectrogram_torch의 결과와 동일함을 검증하는 코드를 포함한다.
# FFT를 사용하는 기존 방법보다 더 효율적으로 스펙트로그램을 계산하여 딥러닝 모델에 적용할 때 유리하게 한 메서드이다.


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec
# 위 코드는 스펙트로그램을 멜 스케일로 변환한다.
# librosa의 멜 필터 뱅크를 사용하여 스펙트로그램을 멜 스펙트로그램으로 변환한다.


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
# 위 코드는 주어진 오디오 신호 y에 대해 멜 스펙트로그램을 직접 계산한다.
# 스펙트로그램을 계산한 후 멜 변환을 적용하여 최종 결과를 반환한다.
# 최종 반환값은 멜 스펙트로그램이다.
# 이 메서드는 오디오 신호에서 멜 스펙트로그램을 한 번에 계산하여 효율성을 높인다.