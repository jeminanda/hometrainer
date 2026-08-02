import numpy as np
from scipy.interpolate import interp1d

def normalize_phase(rep_data: np.ndarray, target_length: int = 100) -> np.ndarray:
    """
    가변 길이의 1회 Rep 데이터를 고정 길이(예: 100 프레임)로 선형 보간 정규화합니다.
    
    rep_data: shape (N_frames, num_features)
    """
    current_length = rep_data.shape[0]
    if current_length < 2:
        return np.zeros((target_length, rep_data.shape[1]))
        
    original_timeline = np.linspace(0, 1, current_length)
    target_timeline = np.linspace(0, 1, target_length)
    
    interpolator = interp1d(original_timeline, rep_data, axis=0, kind='linear')
    normalized_data = interpolator(target_timeline)
    
    return normalized_data


# -----------------------------
# DTW 기반 위상 정규화
# -----------------------------
#
# 기존 normalize_phase()는 프레임 인덱스를 그냥 균등하게 0~100%로 늘리는 선형 보간이라,
# "바닥에서 오래 머무는" 것처럼 사람마다 다른 템포(속도 프로파일)를 무시하고 왜곡시킨다
# (실측 사례: 정상적으로 느리게 올라오는 스쿼트가, 빠른 템포의 기준 영상들과 선형으로
# 비교되면서 초반 구간이 이상치로 잘못 잡힘). DTW(Dynamic Time Warping)는 프레임을
# 균등하게 늘리는 대신, "이 rep의 각도 궤적을 모범 사례 평균 궤적과 모양이 가장
# 비슷하게 맞추려면 어느 프레임을 어디에 대응시켜야 하는가"를 찾아서 정렬한다.
# 템포가 달라도 "바닥 구간 오래 머문 부분"이 템플릿의 "바닥 구간"에 그대로 대응되므로
# 템포 차이 자체가 이상치로 오인되지 않는다.

def build_template(reference_angle_series_list: list[np.ndarray], target_length: int = 100) -> np.ndarray:
    """
    여러 rep의 원본(가변 길이) 대표 각도 시계열로부터 DTW 정렬에 쓸 평균 템플릿을 만든다.

    지금 가진 모범 사례들을 기준으로 평균 궤적을 만드는 것이라, 먼저 기존
    normalize_phase()(선형 보간)로 각 rep을 target_length로 맞춘 뒤 평균을 낸다
    (템플릿을 만들려면 어차피 한 번은 길이를 맞춰야 하므로, 여기서는 기존 방식을
    "부트스트랩"용으로 재사용한다 — 이후 실제 DTW 정렬은 이 평균 템플릿을 기준으로
    이뤄지므로, 부트스트랩 단계의 선형 보간 왜곡은 평균을 내는 과정에서 상당히 희석된다).

    Args:
        reference_angle_series_list: [(T_1,), (T_2,), ...] 각 rep의 원본 각도 시계열
        target_length: 템플릿 길이 (rep_sequences의 target_length와 동일하게 맞출 것)

    Returns:
        (target_length,) 평균 템플릿 각도 시계열
    """
    if len(reference_angle_series_list) == 0:
        raise ValueError("템플릿을 만들 reference 각도 시계열이 하나도 없습니다.")

    aligned = np.stack(
        [normalize_phase(series[:, None], target_length=target_length)[:, 0] for series in reference_angle_series_list],
        axis=0,
    )  # (N_reps, target_length)
    return aligned.mean(axis=0)


def _dtw_align(query: np.ndarray, template: np.ndarray) -> list[list[int]]:
    """
    query(rep 1개의 원본 각도 시계열, 가변 길이)를 template(고정 길이 평균 궤적)에
    DTW로 정렬해서, template의 각 인덱스가 query의 어느 인덱스(들)에 대응되는지 반환한다.

    표준 DP 기반 DTW (경로에 대각선/수평/수직 이동 모두 허용, 유클리드 거리 비용).
    rep 길이가 보통 수십~수백 프레임 수준이라 O(len(query) x len(template)) DP로 충분히 빠르다.

    Returns:
        길이 len(template)인 리스트. mapping[j] = template의 j번째 위치에 대응되는
        query 프레임 인덱스들의 리스트 (여러 프레임이 한 template 위치에 몰릴 수 있음).
    """
    n, m = len(query), len(template)
    if n == 0:
        raise ValueError("query가 비어 있습니다.")

    cost = np.abs(query[:, None] - template[None, :])  # (n, m)

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # 역추적 (끝점에서 시작점까지)
    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dtw[i - 1, j - 1], i - 1, j - 1),
            (dtw[i - 1, j], i - 1, j),
            (dtw[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda c: c[0])
    path.reverse()

    mapping: list[list[int]] = [[] for _ in range(m)]
    for query_idx, template_idx in path:
        mapping[template_idx].append(query_idx)

    # DTW 경로의 경계 조건(반드시 (0,0)에서 시작해 (n-1,m-1)에서 끝남)상 모든 template
    # 인덱스가 최소 1개의 query 프레임과 대응되지만, 혹시 모를 빈 경우를 대비한 안전장치.
    for j in range(m):
        if not mapping[j]:
            mapping[j] = [min(j, n - 1)]

    return mapping


def normalize_phase_dtw(rep_sequence: np.ndarray, rep_angle_series: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    DTW로 rep_sequence(키포인트 원본, 가변 길이)를 template 길이에 맞춰 위상 정규화.

    normalize_phase()가 프레임 인덱스를 균등하게 늘리는 것과 달리, rep_angle_series
    (이 rep의 원본 각도 시계열, rep_sequence와 같은 길이)를 template과 DTW로 정렬해서
    나온 대응 관계를 그대로 rep_sequence(33관절 전체)에 적용한다. 한 template 위치에
    여러 원본 프레임이 대응되면 그 프레임들의 좌표를 평균 낸다.

    Args:
        rep_sequence: (T_rep, J, C) 정규화된(좌표계는 이미 처리됨) 원본 길이 rep 시퀀스
        rep_angle_series: (T_rep,) 이 rep의 원본 대표 각도 시계열 (rep_slicer가 쓴 것과 동일한 각도)
        template: (target_length,) build_template()으로 만든 평균 템플릿

    Returns:
        (target_length, J, C) DTW로 위상 정규화된 시퀀스
    """
    T_rep = rep_sequence.shape[0]
    if T_rep != len(rep_angle_series):
        raise ValueError(
            f"rep_sequence 길이({T_rep})와 rep_angle_series 길이({len(rep_angle_series)})가 다릅니다."
        )

    if T_rep < 2:
        return np.repeat(rep_sequence, len(template), axis=0)

    mapping = _dtw_align(rep_angle_series, template)

    target_length = len(template)
    J, C = rep_sequence.shape[1], rep_sequence.shape[2]
    warped = np.empty((target_length, J, C), dtype=rep_sequence.dtype)
    for j, indices in enumerate(mapping):
        warped[j] = rep_sequence[indices].mean(axis=0)

    return warped