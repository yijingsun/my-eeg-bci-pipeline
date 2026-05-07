import numpy as np
import mne
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings


class OVOCspFeatureExtractor:
    """
    One-vs-One CSP Feature Extractor
    
    Pipeline:
        原始EEG → CSP空间滤波(OVO配对) → log方差特征 → 特征拼接 → StandardScaler(可选) → LDA降维(可选)
    
    核心思路:
        1. 每对类别训练一个CSP，提取 2 * csp_n_components 维特征
        2. 拼接所有类别对的特征，得到完整特征向量
        3. 可选：LDA进一步降维，提取最具判别性的特征
    """
    
    # ==================================================================
    # 1. 初始化与参数
    # ==================================================================
    
    def __init__(self, 
                 csp_n_components=4,                                      # 每个类别对的CSP成分数
                 csp_reg='ledoit_wolf',                                         # CSP正则化方法
                 log_transform=True,                                            # 是否对CSP特征取log
                 normalize_features=True,                                       # 是否对CSP特征进行标准化
                 lda_n_components=None,                                         # LDA降维目标维度(None则跳过LDA)
                 random_state=37                                                # 随机种子
                 ):                                                   
        self.csp_n_components = csp_n_components
        self.csp_reg = csp_reg
        self.log_transform = log_transform
        self.normalize_features = normalize_features
        self.lda_n_components = lda_n_components
        self.random_state = random_state
        
        # 训练后会填充的对象
        self.pairwise_csp_models = {}    # {(c1,c2): CSP_model}
        self.scaler = StandardScaler() if self.normalize_features else None
        self.lda_projection = None       # LDA模型
        self.class_labels = None         # 所有类别标签
        self.is_fitted = False
    
    
    # ==================================================================
    # 2. 训练流程 (主入口)
    # ==================================================================
    
    def fit(self, eeg_epochs_array, event_labels, verbose=True):
        """
        完整训练流程
        
        参数
        ----
        eeg_epochs_array : ndarray, shape (n_trials, n_channels, n_times)
        event_labels : ndarray, shape (n_trials,)
        verbose : bool
        """
        # 2.1 输入验证
        self._validate_inputs(eeg_epochs_array, event_labels)
        self.class_labels = np.unique(event_labels)
        
        if verbose:
            self._print_training_info(eeg_epochs_array)
        
        # 2.2 训练所有OVO-CSP
        self._train_all_csp_pairs(eeg_epochs_array, event_labels, verbose)
        
        # 2.3 计算原始特征矩阵 (CSP特征拼接)
        feature_matrix = self._compute_feature_matrix(eeg_epochs_array)
        
        if verbose:
            print(f"原始CSP特征维度:【{feature_matrix.shape[1]}】")
        
        # 2.4 特征标准化 (可选)
        if self.normalize_features:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
            if verbose:
                print("✓ 特征标准化完成")
        
        # 2.5 LDA降维 (可选)
        if self.lda_n_components is not None:
            feature_matrix = self._apply_lda(feature_matrix, event_labels, verbose)
        
        self.is_fitted = True
        if verbose:
            print("=" * 50 + "\n训练完成!\n")
        
        return self
    
    def transform(self, eeg_epochs_array):
        """
        对新数据进行特征提取
        
        返回
        ----
        features : ndarray
            如果用了LDA: shape (n_trials, lda_n_components)
            如果没用LDA: shape (n_trials, n_pairs * csp_n_components * 2)
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        # 步骤1: CSP特征提取与拼接
        feature_matrix = self._compute_feature_matrix(eeg_epochs_array)
        
        # 步骤2: 标准化
        if self.scaler is not None:
            feature_matrix = self.scaler.transform(feature_matrix)
        
        # 步骤3: LDA降维
        if self.lda_projection is not None:
            feature_matrix = self.lda_projection.transform(feature_matrix)
        
        return feature_matrix
    
    def fit_transform(self, eeg_epochs_array, event_labels, verbose=True):
        """训练 + 转换"""
        self.fit(eeg_epochs_array, event_labels, verbose=verbose)
        return self.transform(eeg_epochs_array)
    
    
    # ==================================================================
    # 3. 核心算法步骤
    # ==================================================================
    
    def _train_all_csp_pairs(self, eeg_epochs_array, event_labels, verbose):
        """
        步骤1: 为每一对类别训练一个CSP模型
        
        CSP原理简述:
        - 对两类数据计算平均协方差矩阵
        - 求解广义特征值问题，找到最大化两类方差比的空问滤波器
        - 特征值越大 → 第一类方差大/第二类方差小
        - 特征值越小 → 第一类方差小/第二类方差大
        """
        n_classes = len(self.class_labels)
        
        for first_index in range(n_classes):
            for second_index in range(first_index + 1, n_classes):
                class_a = self.class_labels[first_index]
                class_b = self.class_labels[second_index]
                pair_key = (class_a, class_b)
                
                # 提取该类别对的试次
                trials_a = eeg_epochs_array[event_labels == class_a]
                trials_b = eeg_epochs_array[event_labels == class_b]
                
                # 拼接为二分类问题: class_a → 0, class_b → 1
                X_pair = np.concatenate([trials_a, trials_b])
                y_pair = np.concatenate([
                    np.zeros(len(trials_a)),
                    np.ones(len(trials_b))
                ])
                
                # 检查试次数是否足够
                min_trials = min(len(trials_a), len(trials_b))
                if min_trials < self.csp_n_components * 2:
                    warnings.warn(
                        f"类别对 {pair_key} 样本量较小 ({min_trials})，"
                        f"CSP成分数 ({self.csp_n_components}) 可能过多"
                    )
                
                # 构建并训练CSP
                csp = mne.decoding.CSP(
                    n_components=self.csp_n_components,
                    reg=self.csp_reg,
                    log=self.log_transform,
                    norm_trace=False,
                    transform_into='average_power'  # 输出平均功率
                )
                
                try:
                    mne.set_log_level('WARNING')    # 避免训练过程中的冗余信息
                    csp.fit(X_pair, y_pair)
                    mne.set_log_level('INFO')
                    self.pairwise_csp_models[pair_key] = csp
                except Exception as e:
                    mne.set_log_level('INFO')
                    warnings.warn(f"✗ CSP训练失败 {pair_key}: {e}")

                if verbose:
                        print(f"✓ CSP训练完成: {pair_key} "
                              f"(试次: {len(trials_a)}+{len(trials_b)})")
        if verbose:
            print("-" * 50)
    
    
    def _compute_feature_matrix(self, eeg_epochs_array):
        """
        步骤2: 对每个CSP模型提取log-方差特征并拼接
        
        对于单个类别对:
        - CSP输出 2 * csp_n_components 个成分
        - 前 csp_n_components 个成分 → 对class_a方差最大
        - 后 csp_n_components 个成分 → 对class_b方差最大
        - 对每个成分的方差取log → 得到 2 * csp_n_components 维特征向量
        
        最终: 拼接所有类别对的特征
        """
        if not self.pairwise_csp_models:
            raise RuntimeError("CSP模型为空，请先训练")
        
        feature_blocks = []
        
        for csp_model in self.pairwise_csp_models.values():
            # CSP空间滤波
            csp_output = csp_model.transform(eeg_epochs_array)
            # output shape: (n_trials, csp_n_components * 2)
            
            # 方差特征
            if csp_output.ndim == 3:
                # 备选路径: 如果输出是 (n_trials, n_filters, n_times)
                variance = np.var(csp_output, axis=-1)
            else:
                # 主路径: (n_trials, n_filters)
                variance = csp_output
            
            # log变换
            if self.log_transform:
                variance = np.log(np.maximum(variance, 1e-10))
            
            # 拼接前后两部分特征
            n = self.csp_n_components
            pair_features = np.concatenate([
                variance[:, :n],    # class_a相关特征
                variance[:, -n:]    # class_b相关特征
            ], axis=1)
            
            feature_blocks.append(pair_features)
        
        # 拼接所有类别对
        return np.concatenate(feature_blocks, axis=1)
    
    
    def _apply_lda(self, feature_matrix, event_labels, verbose):
        """
        步骤3: LDA降维
        
        原理: 寻找最能区分类别的投影方向
        - 最多能降到 n_classes - 1 维
        - 保持类别间距离最大，类别内距离最小
        """
        # 计算实际可用的最大维度
        max_dim = min(
            len(self.class_labels) - 1,   # LDA理论上限
            feature_matrix.shape[1]        # 当前特征维度
        )
        actual_dim = min(self.lda_n_components, max_dim)
        
        if actual_dim < self.lda_n_components:
            warnings.warn(f"LDA维度从{self.lda_n_components}调整为{actual_dim}")
        
        if actual_dim <= 0:
            if verbose:
                print("⊗ 跳过LDA (维度不足)")
            return feature_matrix
        
        # 训练LDA
        self.lda_projection = LinearDiscriminantAnalysis(
            n_components=actual_dim,
            solver='eigen', # 默认用特征值分解 还有'svd'奇异值分解法
            shrinkage='auto'
        )
        reduced = self.lda_projection.fit_transform(feature_matrix, event_labels)
        
        if verbose:
            print(f"✓ LDA降维: 【{feature_matrix.shape[1]} → {actual_dim}】")
        
        return reduced
    
    
    # ==================================================================
    # 4. 辅助方法：验证、可视化、评估
    # ==================================================================
    
    def _validate_inputs(self, X, y):
        """输入验证"""
        if X.ndim != 3:
            raise ValueError(f"期望3维数组 (trials, channels, times)，得到{X.ndim}维")
        if len(X) != len(y):
            raise ValueError(f"试次数量({len(X)})与标签数量({len(y)})不匹配")
    
    def _print_training_info(self, X):
        """打印训练信息"""
        n_classes = len(self.class_labels)
        n_pairs = n_classes * (n_classes - 1) // 2
        print("=" * 50)
        print(f"OVO-CSP 特征提取器训练")
        print(f"  数据维度: {X.shape}")
        print(f"  类别数量: {n_classes}")
        print(f"  类别对数量: {n_pairs}")
        print(f"  每对特征维度: {self.csp_n_components} * 2 = {self.csp_n_components * 2}")
        print("-" * 50)
    
    def get_spatial_patterns(self, pair_key):
        """
        获取指定类别对的CSP空间模式(用于地形图可视化)
        
        CSP的空间模式 = 协方差矩阵 × 空间滤波器
        反映了源空间的激活模式
        """
        if pair_key not in self.pairwise_csp_models:
            raise KeyError(f"类别对 {pair_key} 未找到")
        return self.pairwise_csp_models[pair_key].patterns_
    
    def get_feature_importance(self):
        """获取特征重要性 (仅当使用LDA时有效)"""
        if self.lda_projection is None:
            raise RuntimeError("未使用LDA，无法获取特征重要性")
        importances = np.abs(self.lda_projection.coef_).mean(axis=0)
        return importances / importances.sum()
    
    def evaluate(self, eeg_epochs_array, event_labels, cv_folds=5):
        """
        交叉验证评估特征质量
        
        使用SVM作为分类器，评估提取特征的分类性能
        """
        from sklearn.svm import SVC
        
        features = self.transform(eeg_epochs_array)
        clf = SVC(kernel='rbf', random_state=self.random_state)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                            random_state=self.random_state)
        scores = cross_val_score(clf, features, event_labels, cv=cv, n_jobs=-1)
        
        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'feature_dim': features.shape[1]
        }
    
    
    # ==================================================================
    # 5. 模型持久化
    # ==================================================================
    
    def save(self, filepath):
        """保存模型"""
        state = {
            'config': {
                'csp_n_components': self.csp_n_components,
                'csp_reg': self.csp_reg,
                'log_transform': self.log_transform,
                'normalize_features': self.normalize_features,
                'lda_n_components': self.lda_n_components,
                'random_state': self.random_state,
            },
            'models': {
                'pairwise_csp': self.pairwise_csp_models,
                'scaler': self.scaler,
                'lda': self.lda_projection,
            },
            'metadata': {
                'class_labels': self.class_labels,
                'is_fitted': self.is_fitted,
            }
        }
        joblib.dump(state, filepath)
        # print(f"模型已保存: {filepath}")
    
    @staticmethod
    def load(filepath):
        """加载模型"""
        state = joblib.load(filepath)
        cfg = state['config']
        mdl = state['models']
        meta = state['metadata']
        
        extractor = OVOCspFeatureExtractor(**cfg)
        extractor.pairwise_csp_models = mdl['pairwise_csp']
        extractor.scaler = mdl['scaler']
        extractor.lda_projection = mdl['lda']
        extractor.class_labels = meta['class_labels']
        extractor.is_fitted = meta['is_fitted']
        
        print(f"模型已加载: {filepath}")
        return extractor
    
    def __repr__(self):
        status = "已训练" if self.is_fitted else "未训练"
        n_pairs = len(self.pairwise_csp_models)
        return (f"OVOCspFeatureExtractor({status}, "
                f"类别对数={n_pairs}, "
                f"CSP成分={self.csp_n_components}, "
                f"正则化={self.csp_reg})")


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    
    # 创建特征提取器
    extractor = OVOCspFeatureExtractor(
        csp_n_components=4,
        csp_reg='ledoit_wolf'
    )
    
    print("\n使用方法:")
    print("  # 1. 从epochs提取数据")
    print("  X = epochs.get_data()")
    print("  y = epochs.events[:, -1]")
    print()
    print("  # 2. 训练并提取特征")
    print("  features = extractor.fit_transform(X, y)")
    print()
    print("  # 3. 用新数据提取特征")
    print("  X_test = test_epochs.get_data()")
    print("  test_features = extractor.transform(X_test)")
    print()
    print("  # 4. 保存/加载模型")
    print("  extractor.save('csp_model.joblib')")
    print("  extractor = OVOCspFeatureExtractor.load('csp_model.joblib')")