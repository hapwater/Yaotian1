# modules/mfcc_extractor.py
import librosa
import numpy as np
import noisereduce as nr
import os
print("当前工作目录：", os.getcwd())  # 打印运行脚本时的工作目录

class MFCCExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=8, n_fft=512, hop_length=160):
        """
        初始化MFCC提取器
        :param sample_rate: 采样率（需与录音一致）
        :param n_mfcc: MFCC系数数量
        :param n_fft: FFT窗口大小
        :param hop_length: 帧移（10ms）
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def preprocess_audio(self, audio_path):
        # 加载音频
        y, _ = librosa.load(audio_path, sr=self.sample_rate)

        # 暂时注释降噪（先验证MFCC提取）
        # noise_sample = y[:int(0.3 * self.sample_rate)] if len(y) > int(0.3 * self.sample_rate) else y
        # y_denoised = nr.reduce_noise(y, noise_sample)

        if len(y) > int(0.3 * self.sample_rate):
            noise_sample = y[:int(0.3 * self.sample_rate)]  # 取前0.3秒噪声
        else:
            noise_sample = y  # 若音频过短，直接用全部作为噪声样本

        y_denoised = nr.reduce_noise(
            y=y,
            y_noise=noise_sample,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )

        y_denoised = y

        # 2. 预加重（提升高频）
        y_preemphasized = librosa.effects.preemphasis(y_denoised)

        # 3. 端点检测（去除静音段）
        y_trimmed, _ = librosa.effects.trim(y_preemphasized, top_db=20)

        return y_trimmed

    def extract(self, audio_path, normalize=True):
        """
        提取MFCC特征（含一阶、二阶差分）
        :param audio_path: 音频文件路径
        :param normalize: 是否归一化特征
        :return: MFCC特征矩阵（形状：[时间帧数量, 特征维度]）
        """
        # 预处理音频
        y = self.preprocess_audio(audio_path)

        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # 计算一阶、二阶差分
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # 拼接特征（维度：13*3=39）
        mfcc_combined = np.vstack((mfcc, mfcc_delta, mfcc_delta2)).T  # 转置为[帧, 特征]

        # 特征归一化
        if normalize:
            mfcc_combined = (mfcc_combined - np.mean(mfcc_combined, axis=0)) / (np.std(mfcc_combined, axis=0) + 1e-8)

        return mfcc_combined

    def save_features(self, features, save_path):
        """
        保存特征到.npy文件
        :param features: MFCC特征矩阵
        :param save_path: 保存路径
        """
        np.save(save_path, features)
        print(f"特征已保存至：{save_path}")

    def load_features(self, feature_path):
        """
        从.npy文件加载特征
        :param feature_path: 特征文件路径
        :return: MFCC特征矩阵
        """
        return np.load(feature_path)


# 测试代码
if __name__ == "__main__":
    extractor = MFCCExtractor()
    # 提取特征
    mfcc_features = extractor.extract("recordings/recording.wav")
    print("MFCC特征形状：", mfcc_features.shape)  # 输出：(帧数量, 39)
    # 保存特征
    extractor.save_features(mfcc_features, "templates/test_keyword(9).npy")
    # 加载特征
    loaded_features = extractor.load_features("templates/test_keyword(9).npy")
    print("加载的特征形状：", loaded_features.shape)