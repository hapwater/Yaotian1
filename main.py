# main.py
from modules.recorder import Recorder
from modules.mfcc_extractor import MFCCExtractor
from modules.dtw_matcher import DTWMatcher

def main():
    # 初始化模块
    recorder = Recorder()
    extractor = MFCCExtractor()
    matcher = DTWMatcher(templates_dir="templates")

    # 1. 录制音频
    audio_path = recorder.record(duration=3)

    # 2. 提取MFCC特征
    mfcc_features = extractor.extract(audio_path)

    # 3. DTW匹配模板
    match_result = matcher.match(mfcc_features, threshold=0.3)

    # 4. 输出结果
    if match_result["success"]:
        print(f"匹配成功！关键词：{match_result['keyword']}，得分：{match_result['score']:.4f}")
        # 后续逻辑：播放故事、控制GPIO等
    else:
        print(f"匹配失败！最高相似度：{match_result['score']:.4f}")

if __name__ == "__main__":
    main()