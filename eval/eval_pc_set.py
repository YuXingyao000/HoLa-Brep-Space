import numpy as np


def evaluate_uniformity_nnd(points):
    """
    Evaluate point cloud uniformity using Nearest Neighbor Distance (NND)
    Args:
        points: numpy array of shape (N, 3)
    Returns:
        dict containing NND statistics
    """
    # 1. 计算每个点到其最近邻的距离
    diff = points[:, None, :] - points[None, :, :]  # (N, N, 3)
    distances = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)

    # 将自身距离设为无穷大
    np.fill_diagonal(distances, np.inf)

    # 获取每个点的最近邻距离
    min_distances = np.min(distances, axis=1)  # (N,)

    # 2. 计算统计指标
    metrics = {
        'mean_nnd': np.mean(min_distances),
        'std_nnd' : np.std(min_distances),
        'cv_nnd'  : np.std(min_distances) / np.mean(min_distances),  # 变异系数
        'min_nnd' : np.min(min_distances),
        'max_nnd' : np.max(min_distances),

        # Clark-Evans R统计量: R = 实际平均最近邻距离 / 期望平均最近邻距离
        # R接近1表示随机分布，R<1表示聚集，R>1表示均匀
        'density' : len(points) / np.prod(np.max(points, axis=0) - np.min(points, axis=0)),
    }

    # 计算Clark-Evans R统计量
    expected_mean_dist = 0.5 / np.sqrt(metrics['density'])
    metrics['clark_evans_r'] = metrics['mean_nnd'] / expected_mean_dist

    # 3. 计算直方图数据（可用于可视化）
    hist, bins = np.histogram(min_distances, bins='auto', density=True)
    metrics['hist_values'] = hist
    metrics['hist_bins'] = bins

    return metrics
