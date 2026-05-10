# modules/dtw_matcher.py
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
import os


class DTWMatcher:
    def __init__(self, templates_dir="templates", distance_metric="cosine"):
        """
        初始化DTW匹配器
        :param templates_dir: 模板特征目录（存储.npy文件）
        :param distance_metric: 距离度量方式（"euclidean"或"cosine"）
        """
        self.templates_dir = templates_dir
        self.distance_metric = euclidean if distance_metric == "euclidean" else cosine
        self.templates = self._load_templates()  # 加载所有模板

    def _load_templates(self):
        """加载模板目录下的所有.npy特征文件"""
        templates = {}
        if not os.path.exists(self.templates_dir):
            print(f"模板目录 {self.templates_dir} 不存在！")
            return templates

        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".npy"):
                keyword = os.path.splitext(filename)[0]  # 文件名作为关键词
                feature_path = os.path.join(self.templates_dir, filename)
                templates[keyword] = np.load(feature_path)
        print(f"成功加载 {len(templates)} 个模板：{list(templates.keys())}")
        return templates

    def match(self, input_features, threshold=0.3):
        """
        匹配输入特征与模板
        :param input_features: 输入MFCC特征矩阵
        :param threshold: 匹配阈值（距离越小越相似）
        :return: 匹配结果字典（success, keyword, score）
        """
        if not self.templates:
            return {"success": False, "message": "无模板可匹配"}

        # 计算输入特征与每个模板的DTW距离
        match_results = {}
        for keyword, template_features in self.templates.items():
            distance, _ = fastdtw(input_features, template_features, dist=self.distance_metric)
            # 归一化距离（除以总帧数）
            normalized_distance = distance / (len(input_features) + len(template_features))
            match_results[keyword] = normalized_distance

        # 找到距离最小的模板
        best_keyword = min(match_results, key=match_results.get)
        best_score = match_results[best_keyword]

        # 判断是否匹配成功
        success = best_score < threshold
        return {
            "success": success,
            "keyword": best_keyword if success else None,
            "score": best_score,
            "all_scores": match_results
        }


# 测试代码
if __name__ == "__main__":
    from mfcc_extractor import MFCCExtractor

    # 初始化模块
    extractor = MFCCExtractor()
    matcher = DTWMatcher(templates_dir="templates")

    # 提取测试音频特征
    test_features = extractor.extract("recordings/recording.wav")

    # 匹配模板
    result = matcher.match(test_features, threshold=0.3)
    print("匹配结果：", result)