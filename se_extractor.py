model_size = "medium"
model = None

def split_audio_whisper(audio_path, audio_name, target_dir='processed'):
    global model
    if model is None:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)

    target_folder = os.path.join(target_dir, audio_name)
    
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)    

    # create directory
    os.makedirs(target_folder, exist_ok=True)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)

    # segments
    s_ind = 0
    start_time = None
    
    for k, w in enumerate(segments):
        # process with the time
        if k == 0:
            start_time = max(0, w.start)

        end_time = w.end

        # calculate confidence
        if len(w.words) > 0:
            confidence = sum([s.probability for s in w.words]) / len(w.words)
        else:
            confidence = 0.
        # clean text
        text = w.text.replace('...', '')

        # left 0.08s for each audios
        audio_seg = audio[int( start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]

        # segment file name
        fname = f"{audio_name}_seg{s_ind}.wav"

        # filter out the segment shorter than 1.5s and longer than 20s
        save = audio_seg.duration_seconds > 1.5 and \
                audio_seg.duration_seconds < 20. and \
                len(text) >= 2 and len(text) < 200 

        if save:
            output_file = os.path.join(wavs_folder, fname)
            audio_seg.export(output_file, format='wav')

        if k < len(segments) - 1:
            start_time = max(0, segments[k+1].start - 0.08)

        s_ind = s_ind + 1
    return wavs_folder
# 위 메서드는 오디오 파일을 세그먼트 단위로 자르는 작업을 수행한다.
# 오디오 파일 경로, 오디오 이름, 그리고 대상 디렉토리를 파라미터로 받는다.
# 전역 변수 model이 None이라면 WhisperModel을 초기화한다. 
# 오디오 로드: pydub의 AudioSegment를 사용하여 오디오 파일을 로드한다.
# 음성 인식: Whisper 모델을 사용하여 오디오를 전사(transcribe)합니다. 단어 단위의 타임스탬프를 포함한다.
# 처리된 오디오 세그먼트를 저장할 디렉토리 구조를 생성한다.
# 각 세그먼트(단어 또는 문장)에 대해 반복
# 필터링 및 저장: WAV 형식으로 저장


def split_audio_vad(audio_path, audio_name, target_dir, split_seconds=10.0):
    SAMPLE_RATE = 16000
    audio_vad = get_audio_tensor(audio_path)
    segments = get_vad_segments(
        audio_vad,
        output_sample=True,
        min_speech_duration=0.1,
        min_silence_duration=1,
        method="silero",
    )
    segments = [(seg["start"], seg["end"]) for seg in segments]
    segments = [(float(s) / SAMPLE_RATE, float(e) / SAMPLE_RATE) for s,e in segments]
    print(segments)
    audio_active = AudioSegment.silent(duration=0)
    audio = AudioSegment.from_file(audio_path)

    for start_time, end_time in segments:
        audio_active += audio[int( start_time * 1000) : int(end_time * 1000)]
    
    audio_dur = audio_active.duration_seconds
    print(f'after vad: dur = {audio_dur}')
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)
    start_time = 0.
    count = 0
    num_splits = int(np.round(audio_dur / split_seconds))
    assert num_splits > 0, 'input audio is too short'
    interval = audio_dur / num_splits

    for i in range(num_splits):
        end_time = min(start_time + interval, audio_dur)
        if i == num_splits - 1:
            end_time = audio_dur
        output_file = f"{wavs_folder}/{audio_name}_seg{count}.wav"
        audio_seg = audio_active[int(start_time * 1000): int(end_time * 1000)]
        audio_seg.export(output_file, format='wav')
        start_time = end_time
        count += 1
    return wavs_folder
# 위 메서드는 VAD(Voice Activity Detection)를 사용하여 오디오 파일을 처리하고 분할하는 작업을 수행한다.
# 오디오 파일 경로, 오디오 이름, 대상 디렉토리, 그리고 분할 시간(기본값 10초)을 파라미터로 받는다.
# VAD: get_audio_tensor 함수를 사용하여 오디오를 텐서로 변환한다. 그리고 get_vad_segments 함수를 사용하여 음성 활동 구간을 검출한다. Silero VAD 모델을 사용한다.
# 활성 음성 구간 추출: 빈 오디오 세그먼트를 생성하고, VAD로 검출된 활성 구간만을 추출하여 연결
# 디렉토리 생성: 처리된 오디오 세그먼트를 저장할 디렉토리 구조를 생성
# 오디오 분할: 전체 활성 음성 구간의 길이 계산 -> split_seconds에 따라 분할 횟수를 계산 -> 분할 길이 계산
# 세그먼트 저장: 분할 구간에 따라 오디오 순차적으로 분할, WAV 형식으로 저장
# 분할하여 저장한 오디오 폴더 경로 반환(return)













