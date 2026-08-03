"""
src/scoring_model/scorer.py

특징 벡터와 ReferenceStats(정상 사례 통계)를 받아 Mahalanobis distance를 계산하고,
이를 0~100점 스코어로 변환한다. 어떤 관절/각도가 감점에 가장 크게 기여했는지도
함께 계산해 "무릎 가동범위가 부족합니다" 같은 피드백에 쓸 수 있게 한다.
"""

from __future__ import annotations

import re

import numpy as np

from .reference_stats import ReferenceStats

# -----------------------------
# 특징 이름 -> 자연어 변환
# -----------------------------

JOINT_LABELS = {
    "knee": "무릎",
    "elbow": "팔꿈치",
    "hip": "허리(엉덩이)",
    "shoulder": "어깨",
}

STAT_LABELS = {
    "min_angle": "최저 각도(가장 굽혀졌을 때)",
    "max_angle": "최고 각도(가장 펴졌을 때)",
    "rom": "가동범위(ROM)",
}

_BIN_PATTERN = re.compile(r"^(?P<joint>[a-z]+)_bin(?P<bin>\d{2})$")
_STAT_PATTERN = re.compile(r"^(?P<stat>min_angle|max_angle|rom)_(?P<joint>[a-z]+)$")

# 3분할(기본값)일 때는 %보다 동작 단계 이름이 더 직관적이다 — 대부분의 운동(스쿼트/
# 팔굽혀펴기 등)이 "내려갈 때 -> 최저점에서 버틸 때 -> 올라갈 때"의 3단계 구조라서.
_THREE_BIN_LABELS = ["하강 구간(내려갈 때)", "최저점 구간(버틸 때)", "상승 구간(올라갈 때)"]


def describe_feature_name(feature_name: str, num_phase_bins: int = 3) -> str:
    """
    특징 이름(예: "min_angle_knee", "knee_bin01")을 사람이 읽을 수 있는 문장으로 변환.

    phase-bin 특징(`{joint}_binNN`)은 feature_extraction.extract_rep_features()가
    rep 전체를 num_phase_bins개 구간으로 나눈 것 중 몇 번째인지를 언어화한다.
    **num_phase_bins는 그 특징을 실제로 추출할 때 쓴 값과 반드시 일치해야 한다**
    (다르면 잘못된 구간으로 언어화된다 — bin 인덱스만으로는 전체 구간 수를 알 수 없음).
    num_phase_bins=3(기본값)이면 %구간 대신 "하강/최저점/상승" 같은 의미 있는 동작
    단계 이름을 쓰고, 그 외 개수면 %구간(예: bin01/10bins -> "10~20%")으로 표시한다.

    알려진 패턴에 안 걸리면(예: reliable_side_* 같은 메타데이터) 이름을 그대로 반환한다.
    """
    m = _BIN_PATTERN.match(feature_name)
    if m:
        joint = m.group("joint")
        bin_idx = int(m.group("bin"))
        joint_label = JOINT_LABELS.get(joint, joint)

        if num_phase_bins == 3 and bin_idx < len(_THREE_BIN_LABELS):
            phase_label = _THREE_BIN_LABELS[bin_idx]
            return f"{joint_label} 각도 ({phase_label})"

        bin_width = 100.0 / num_phase_bins
        start_pct, end_pct = bin_idx * bin_width, (bin_idx + 1) * bin_width
        return f"{joint_label} 각도 (rep 진행 {start_pct:.0f}~{end_pct:.0f}% 구간)"

    m = _STAT_PATTERN.match(feature_name)
    if m:
        stat_label = STAT_LABELS[m.group("stat")]
        joint_label = JOINT_LABELS.get(m.group("joint"), m.group("joint"))
        return f"{joint_label} {stat_label}"

    return feature_name


def format_issue(feature_name: str, z_score: float, num_phase_bins: int = 3) -> str:
    """
    (특징 이름, z-score) 한 쌍을 "무릎 각도 (최저점 구간(버틸 때)): 평균보다 작음 (z=-2.1)"
    같은 완성된 문장으로 변환한다. num_phase_bins는 describe_feature_name과 동일하게,
    실제 특징 추출 시 쓴 bin 개수와 일치해야 한다.
    """
    description = describe_feature_name(feature_name, num_phase_bins=num_phase_bins)
    direction = "평균보다 큼" if z_score > 0 else "평균보다 작음"
    return f"{description}: {direction} (z={z_score:+.2f})"


def mahalanobis_distance(feature_vector: np.ndarray, stats: ReferenceStats) -> float:
    """정상 사례 분포 기준으로 feature_vector까지의 Mahalanobis distance."""
    diff = feature_vector - stats.mean
    return float(np.sqrt(diff @ stats.inv_cov @ diff))


def calibrate_decay_scale(
    distance: float, stats: ReferenceStats, target_score: float, inside_percentile: float = 95.0
) -> float:
    """
    "이 정도로 벗어난 rep은 이 정도 점수였으면 좋겠다"를 그대로 넣으면 맞는 decay_scale을 역산.

    예) 실제로 겪은 rep의 distance가 5.89였는데 그게 50점 정도였으면 좋겠다면:
        decay_scale = calibrate_decay_scale(5.89, stats, target_score=50.0)
    이렇게 구한 값을 이후 score_rep()/distance_to_score() 호출 시 decay_scale=... 로 넘기면 된다.

    Args:
        distance: 기준으로 삼을 실제 Mahalanobis distance (예: 예전에 0점 나왔던 그 rep의 distance)
        stats: 그 rep을 채점할 때 썼던 것과 동일한 ReferenceStats
        target_score: 그 distance에서 나왔으면 하는 목표 점수 (0~100, 100 미만이어야 함)
        inside_percentile: distance_to_score와 동일한 값을 넣어야 한다 (기본 85)

    Returns:
        decay_scale (distance_to_score에 그대로 넘길 값)
    """
    if not (0 < target_score < 100):
        raise ValueError("target_score는 0과 100 사이(100 미만)여야 합니다.")

    threshold = float(np.percentile(stats.reference_distances, inside_percentile))
    if distance <= threshold:
        raise ValueError(
            f"distance({distance})가 이미 임계값({threshold:.3f}) 이내라 감쇠 자체가 적용되지 않고 "
            "무조건 100점입니다. decay_scale로 조절할 수 있는 구간이 아닙니다."
        )

    spread = float(np.std(stats.reference_distances)) + 1e-6
    excess = (distance - threshold) / spread
    return float(excess / np.log(100.0 / target_score))


def distance_to_score(
    distance: float,
    stats: ReferenceStats,
    method: str = "threshold",
    inside_percentile: float = 90.0,
    decay_scale: float = 3.0,
) -> float:
    """
    Mahalanobis distance를 0~100점으로 변환.

    method="percentile": 기준 집단(stats.reference_distances) 안에서의 상대 순위로 점수를 매긴다.
        문제는, 학습에 쓴 데이터가 전부 "모범 사례"인데도 이 방식은 그 모범 사례들끼리
        상대 순위를 매겨버려서, 완전히 정상적인 폼이어도 그저 기준 집단의 중앙값 근처에
        있다는 이유만으로 50점 근처가 나온다 (교과서적인 자세인데도 만점을 못 받는 셈).
        기준 집단의 최댓값을 넘어서면 무조건 0점으로 포화되는 문제도 있다.

    method="threshold"(기본): 학습 데이터가 전부 정상 사례라는 전제를 그대로 살려서,
        "기준 집단이 보여준 자연스러운 변동 범위 안"이면 그냥 100점을 준다. 그 범위를
        벗어나야만 감점이 시작되고, 벗어난 정도(스프레드 단위)에 비례해 완만하게 줄어든다.
        경계는 max(reference_distances)가 아니라 inside_percentile(기본 85%)로 잡는다 —
        기준 데이터 안에 노이즈 섞인 이상치가 하나라도 있으면(실제로 겪었던 rep 과다분절
        사례처럼) max를 그대로 쓸 경우 그 이상치 하나가 "만점 구간"의 크기를 왜곡시키기
        때문에, 상위 몇 % 정도는 이상치로 보고 무시하는 편이 더 안정적이다.

        inside_percentile을 95 -> 85로 낮췄다 (실측으로 확인된 문제: pushup처럼 특징이
        많고(52개) 사람 수도 다양하면 reference_distances 자체의 분산이 커져서,
        95%ile 기준 threshold가 너무 관대해져 폼이 명백히 나쁜 rep도 threshold 안쪽에
        들어가 100점이 나오는 사례가 실제로 있었다. 85%로 낮추면 "만점 구간"이 좁아져서
        이런 경우를 더 잘 잡아내지만, 정상적인 개인차도 더 자주 감점 구간으로 넘어갈 수
        있다는 트레이드오프가 있다 — 실제 데이터로 good/bad 분리도를 재검증할 것.

        decay_scale: 경계를 넘은 뒤 감쇠 속도. `score = 100 * exp(-초과분 / decay_scale)`.
        값이 클수록 완만해진다. decay_scale=1(첫 시도)이었을 때는 경계를 스프레드 1배만
        넘어도 100 -> 37점으로 너무 가파르게 떨어져서, 기본값을 3으로 완만하게 바꿨다
        (경계+1스프레드에서 100->72점, +3스프레드에서 37점 정도로 덜 급격하게 감).
    """
    if method == "percentile":
        percentile = float(np.mean(stats.reference_distances <= distance)) * 100
        return float(np.clip(100.0 - percentile, 0.0, 100.0))

    if method != "threshold":
        raise ValueError(f"알 수 없는 method: {method} (percentile/threshold 중 하나)")

    threshold = float(np.percentile(stats.reference_distances, inside_percentile))
    if distance <= threshold:
        return 100.0

    spread = float(np.std(stats.reference_distances)) + 1e-6
    excess = (distance - threshold) / spread  # 경계를 몇 스프레드만큼 넘었는지
    return float(100.0 * np.exp(-excess / decay_scale))


def per_feature_contribution(feature_vector: np.ndarray, stats: ReferenceStats) -> dict[str, float]:
    """
    각 특징이 평균에서 얼마나 벗어났는지(표준편차 단위 z-score)를 계산.
    공분산 간 상호작용까지는 반영하지 않지만, "어느 각도가 문제인지" 피드백용으로 충분히 유용하다.

    Returns:
        {feature_name: z-score} — 절댓값이 클수록 그 특징이 크게 벗어났다는 뜻
    """
    std = np.sqrt(np.diag(stats.cov))
    std = np.where(std < 1e-8, 1e-8, std)
    z_scores = (feature_vector - stats.mean) / std
    return dict(zip(stats.feature_names, z_scores.tolist()))


def list_outliers(contributions: dict[str, float], z_threshold: float = 1.5) -> list[tuple[str, float]]:
    """
    per_feature_contribution() 결과에서, |z-score|가 z_threshold를 넘는 특징만 전부 골라
    심한 순서대로 정렬해서 반환한다. top_k_feedback(상위 몇 개 고정)과 달리, "이상치로
    볼 만한 게 몇 개인지 자체가 rep마다 다를 수 있다"는 걸 그대로 반영한 rep별 이상치 목록이다
    (전부 정상이면 빈 리스트가 나올 수 있고, 문제가 많으면 여러 개가 나올 수 있다).
    """
    outliers = [(name, z) for name, z in contributions.items() if abs(z) >= z_threshold]
    return sorted(outliers, key=lambda kv: abs(kv[1]), reverse=True)


def score_rep(
    feature_vector: np.ndarray,
    stats: ReferenceStats,
    top_k_feedback: int = 2,
    decay_scale: float = 3.0,
    outlier_z_threshold: float = 1.5,
) -> dict:
    """
    rep 1개를 채점하는 엔드투엔드 함수.

    decay_scale: distance_to_score()에 그대로 전달됨. 새 영상의 특정 rep을 기준으로
    calibrate_decay_scale()로 역산한 값을 넣으면, 그 rep이 원하는 점수가 나오도록
    감쇠 속도를 조정할 수 있다 (score_video()에서 이걸 그대로 지원한다).

    outlier_z_threshold: list_outliers()에 그대로 전달되는 이상치 판정 기준.

    Returns:
        {
            "score": 0~100,
            "distance": Mahalanobis distance,
            "feature_contributions": {feature_name: z-score, ...},
            "top_issues": [(feature_name, z-score), ...] 절댓값 기준 상위 top_k_feedback개
            "outliers": [(feature_name, z-score), ...] |z|>=outlier_z_threshold인 것 전부
                (top_issues와 달리 개수가 rep마다 다를 수 있음 — 문제 없으면 빈 리스트)
            "outlier_messages": ["무릎 각도 (rep 진행 30~40% 구간): 평균보다 작음 (z=-2.10)", ...]
                outliers를 그대로 자연어 문장으로 바꾼 것
        }
    """
    distance = mahalanobis_distance(feature_vector, stats)
    score = distance_to_score(distance, stats, decay_scale=decay_scale)
    contributions = per_feature_contribution(feature_vector, stats)

    top_issues = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k_feedback]
    outliers = list_outliers(contributions, z_threshold=outlier_z_threshold)
    outlier_messages = [format_issue(name, z) for name, z in outliers]

    return {
        "score": score,
        "distance": distance,
        "feature_contributions": contributions,
        "top_issues": top_issues,
        "outliers": outliers,
        "outlier_messages": outlier_messages,
    }
