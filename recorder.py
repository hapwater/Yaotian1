# modules/recorder.py
import sounddevice as sd
import soundfile as sf
import numpy as np
import os


class Recorder:
    def __init__(self, sample_rate=16000, channels=1, output_dir="recordings"):
        """
        初始化录音器
        :param sample_rate: 采样率（语音处理常用16000Hz）
        :param channels: 声道数（单声道）
        :param output_dir: 录音文件保存目录
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)  # 创建目录（不存在则创建）

    def record(self, duration=3, filename="recording.wav"):
        """
        录制音频
        :param duration: 录制时长（秒）
        :param filename: 保存的文件名
        :return: 录音文件的完整路径
        """
        output_path = os.path.join(self.output_dir, filename)

        print(f"开始录制 {duration} 秒音频...")
        # 录制音频（返回numpy数组）
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()  # 等待录制完成

        # 保存为WAV文件
        sf.write(output_path, audio_data, self.sample_rate)
        print(f"录制完成，文件保存至：{output_path}")
        return output_path

    def record_until_silence(self, silence_threshold=0.01, silence_duration=1, max_duration=10):
        """
        录制直到检测到静音（适用于不定长语音）
        :param silence_threshold: 静音阈值（音频幅值）
        :param silence_duration: 静音持续时间（秒）
        :param max_duration: 最大录制时长（秒）
        :return: 录音文件路径
        """
        output_path = os.path.join(self.output_dir, "recording_until_silence.wav")
        audio_frames = []
        silence_frames = 0
        silence_frames_threshold = int(silence_duration * self.sample_rate)
        frames_per_callback = 1024  # 定义每次回调的帧数（可根据需求调整）

        def callback(indata, frames, time, status):
            nonlocal silence_frames
            audio_frames.append(indata.copy())
            # 检测静音（当前帧的均方根幅值）
            rms = np.sqrt(np.mean(indata ** 2))
            if rms < silence_threshold:
                silence_frames += frames
            else:
                silence_frames = 0

        # 开始流式录制
        with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback,
                dtype=np.float32,
                blocksize=frames_per_callback  # 指定每次回调的帧数
        ):
            print("开始录制（检测到静音自动停止）...")
            while silence_frames < silence_frames_threshold and len(
                    audio_frames) * frames_per_callback < max_duration * self.sample_rate:
                sd.sleep(100)

        # 拼接音频数据并保存
        audio_data = np.concatenate(audio_frames, axis=0)
        sf.write(output_path, audio_data, self.sample_rate)
        print(f"录制完成，文件保存至：{output_path}")
        return output_path

# 测试代码
if __name__ == "__main__":
    recorder = Recorder()
    recorder.record(duration=3)  # 录制3秒
    # recorder.record_until_silence()  # 录制直到静音