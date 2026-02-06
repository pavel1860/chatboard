
import numpy as np


def rsf_fussion(res, top_k, alpha = 0.5):

    if type(alpha) == float:
        alpha_list = [alpha, 1 - alpha]
    elif type(alpha) == list:
        alpha_list = alpha

    assert len(alpha_list) == len(res)

    def normalize(scores):
        scores_arr = np.array(scores)    
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        if std == 0:
            std = 0.0001
        return (scores_arr - (mean - 3 * std)) / (6 * std)

    all_normlized_points = []
    for points in res:
        normlized_points = {}
        norm_scores = normalize([p.score for p in points])
        for i, p in enumerate(points):
            pc = p.model_copy()
            pc.score = norm_scores[i]
            normlized_points[pc.id]=pc
        all_normlized_points.append(normlized_points)


    def get_score(idx, pid):
        point = all_normlized_points[idx].get(pid, None)
        if point is None:
            return 0
        return point.score

    visited = set()

    fussed_points = []
    for vec_idx, points in enumerate(all_normlized_points):
        for p in points.values():
            if p.id in visited:
                continue
            pc = p.model_copy()
            score = 0
            for i, a in enumerate(alpha_list):
                score += a * get_score(i, p.id)
            pc.score = score
            visited.add(pc.id)
            fussed_points.append(pc)

    sorted_points = sorted(fussed_points, key=lambda x: x.score, reverse=True) 
    return sorted_points[:top_k]