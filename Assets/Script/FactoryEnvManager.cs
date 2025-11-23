using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>
/// 공장 전체의 Spawner / Tunnel을 스캔해서
/// - nodeId 기준으로 상태(State, Q, Capacity)를 모으고
/// - nodeId 기준 그래프(인접 리스트)를 만든다.
/// + 터널의 FAULT 진입/탈출을 감지해서 고장 리워드(테스트용)를 관리한다.
/// + PDF 수식 기반의 "글로벌 보상 R_total"을 계산해 테스트 로그를 출력한다.
/// + (추가) 의사결정 시점마다 관찰 윈도우 T 동안 PL/QD/FT/BT를 샘플링해서
///         윈도우 단위 리워드를 계산할 수 있다.
/// 나중에 RL / Python 브릿지에서 이 매니저만 바라보면 됨.
/// </summary>
public class FactoryEnvManager : MonoBehaviour
{
    // ==== Singleton (편의용) ====
    public static FactoryEnvManager Instance { get; private set; }

    [Header("Scene References (비워두면 자동 찾기)")]
    public ProductSpawner[] spawners;
    public TunnelController[] tunnels;

    // nodeId -> NodeData
    private Dictionary<int, NodeData> nodes = new Dictionary<int, NodeData>();

    // nodeId -> 나가는 child nodeId 리스트 (그래프 인접 리스트)
    private Dictionary<int, List<int>> adjacency = new Dictionary<int, List<int>>();

    // 각 터널의 "이전 프레임 상태"를 기억해서 Fault 진입/탈출을 감지
    private Dictionary<TunnelController, TunnelController.TunnelState> _lastTunnelStates
        = new Dictionary<TunnelController, TunnelController.TunnelState>();

    // ===== 테스트용 고장 리워드 관리 (Fault별 점수) =====
    [Header("Fault Reward (테스트용, per-tunnel)")]
    [Tooltip("고장 발생 시 부여할 최소 리워드")]
    public float minFaultReward = 1f;

    [Tooltip("고장 발생 시 부여할 최대 리워드")]
    public float maxFaultReward = 5f;

    // 고장난 터널 -> 리워드 점수
    private Dictionary<TunnelController, float> faultRewards
        = new Dictionary<TunnelController, float>();

    // ===== 글로벌 리워드 (PDF 수식 기반) =====
    //
    // 슬라이드의 개념:
    //   PL(T) : 생산량
    //   QD(T) : 큐 길이 (혼잡도)
    //   FT(T) : 고장 시간
    //   BT(T) : 라인 블로킹 시간
    //   EC(T) : 에너지
    //   RO(T) : 로봇 운용 비용
    //
    //   R_total = w1 * PL~ - w2 * QD~ - w3 * FT~ - w4 * BT~ - w5 * EC~ - w6 * RO~
    //
    // 여기서는 단순화 버전으로:
    //   - PL : sink 터널의 throughput (관찰 윈도우에서 delta count 사용 가능)
    //   - QD : 전체 큐 길이 합
    //   - FT : FAULT 터널 개수
    //   - BT : HOLD + HALF_HOLD 터널 개수
    //   - EC, RO : 지금은 0 (나중에 로봇 이동량/수리 횟수와 연결 가능)
    //
    [Header("RL Reward (Global, Next-week Formula / Test)")]
    [Tooltip("R_total = w1*PL~ - w2*QD~ - w3*FT~ - w4*BT~ - w5*EC~ - w6*RO~ 을 주기적으로 로그 출력할지 여부")]
    public bool debugLogGlobalReward = true;

    [Tooltip("글로벌 리워드 로그 주기(초, 즉시형 인스턴트 리워드)")]
    public float globalRewardLogInterval = 1f;

    private float _nextGlobalRewardLogTime = 0f;

    [Header("Reward Weights (w1~w6)")]
    [Tooltip("생산량 PL~의 가중치 (좋은 항, +)")]
    public float w1_PL = 1f;

    [Tooltip("큐 길이 QD~의 가중치 (나쁜 항, -)")]
    public float w2_QD = 1f;

    [Tooltip("고장 FT~의 가중치 (나쁜 항, -)")]
    public float w3_FT = 1f;

    [Tooltip("블로킹 BT~의 가중치 (나쁜 항, -)")]
    public float w4_BT = 1f;

    [Tooltip("에너지 EC~의 가중치 (나쁜 항, -)")]
    public float w5_EC = 0f; // 아직 미사용이므로 0으로 시작

    [Tooltip("로봇 운용비 RO~의 가중치 (나쁜 항, -)")]
    public float w6_RO = 0f; // 아직 미사용이므로 0으로 시작

    [Header("Reward Normalizers (max 값 가정)")]
    [Tooltip("PL 정규화용 최대값 (예: 시간 T 동안 가능한 최대 생산량)")]
    public float maxPL = 1f;

    [Tooltip("QD 정규화용 최대값 (예: 모든 큐가 풀로 찬 상태의 합)")]
    public float maxQD = 10f;

    [Tooltip("FT 정규화용 최대값 (예: 터널 수 또는 시간 누적 등)")]
    public float maxFT = 5f;

    [Tooltip("BT 정규화용 최대값 (예: 최악 블로킹 상태 기준)")]
    public float maxBT = 5f;

    [Tooltip("EC 정규화용 최대값 (에너지)")]
    public float maxEC = 1f;

    [Tooltip("RO 정규화용 최대값 (로봇 운용비)")]
    public float maxRO = 1f;

    // 최근에 계산된 글로벌 리워드 값
    private float _lastGlobalReward = 0f;

    // ===== 관찰 윈도우 기반 리워드 (R_t for one decision) =====
    [Header("RL Observation Window (per decision)")]
    [Tooltip("의사결정마다 T초 동안 PL/QD/FT/BT를 관찰해 윈도우 리워드를 계산할지 여부")]
    public bool useObservationWindow = false;

    [Tooltip("관찰 윈도우 길이 T (seconds)")]
    public float observationWindow = 8f;

    bool _isObserving = false;
    float _obsEndTime;

    // 시간 평균을 위한 누적값
    float _sumQD, _sumFT, _sumBT, _sumEC, _sumRO;
    int _sampleCount;

    // sink 터널들의 시작 시점 throughput 카운트
    Dictionary<TunnelController, int> _sinkStartCounts
        = new Dictionary<TunnelController, int>();

    [System.Serializable]
    public class NodeData
    {
        public int nodeId;
        public string name;

        public bool isSpawner;
        public ProductSpawner spawner;        // isSpawner == true 일 때
        public TunnelController tunnel;       // isSpawner == false 일 때

        // Tunnel인 경우에만 유효
        public TunnelController.TunnelState tunnelState;
        public int queueCount;
        public int queueCapacity;
    }

    // 외부에서 읽기용
    public IReadOnlyDictionary<int, NodeData> Nodes => nodes;
    public IReadOnlyDictionary<int, List<int>> Adjacency => adjacency;

    [Header("Debug 옵션")]
    [Tooltip("Awake 시 한 번 그래프 구조를 로그로 출력")]
    public bool debugLogOnBuild = true;

    [Header("Compact State Debug (한 줄 요약 로그)")]
    [Tooltip("true면 일정 주기로 전체 노드 상태를 한 줄로 출력")]
    public bool debugCompactState = false;

    [Tooltip("Compact 상태 로그 주기(초)")]
    public float debugCompactInterval = 1f;

    private float _nextCompactLogTime = 0f;

    void Awake()
    {
        // Singleton 세팅
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("[FactoryEnvManager] 이미 인스턴스가 존재해서 두 번째 인스턴스를 제거합니다.");
            Destroy(this);
            return;
        }
        Instance = this;

        // 씬에서 자동 스캔 (인스펙터에서 수동 지정해도 됨)
        if (spawners == null || spawners.Length == 0)
            spawners = FindObjectsOfType<ProductSpawner>();

        if (tunnels == null || tunnels.Length == 0)
            tunnels = FindObjectsOfType<TunnelController>();

        BuildNodeIndex();
        BuildGraphEdges();

        if (debugLogOnBuild)
        {
            DumpGraphToLog();
        }

        _nextCompactLogTime = Time.time + debugCompactInterval;
        _nextGlobalRewardLogTime = Time.time + globalRewardLogInterval;
    }

    void Update()
    {
        // 매 프레임마다 상태만 갱신
        UpdateNodeStates();

        // 한 줄 compact 로그
        if (debugCompactState && Time.time >= _nextCompactLogTime)
        {
            DumpCompactStatesToLog();
            _nextCompactLogTime = Time.time + Mathf.Max(0.1f, debugCompactInterval);
        }

        // === 관찰 윈도우 T 처리 (의사결정 기반 리워드) ===
        if (useObservationWindow && _isObserving)
        {
            SampleForObservation();

            if (Time.time >= _obsEndTime)
            {
                FinishObservationAndComputeReward();
            }
        }

        // === PDF 수식 기반 "즉시형" 글로벌 리워드 로그 (선택) ===
        if (debugLogGlobalReward && Time.time >= _nextGlobalRewardLogTime)
        {
            float plT, qdT, ftT, btT, ecT, roT;
            float plN, qdN, ftN, btN, ecN, roN;

            float r = ComputeGlobalReward(
                out plT, out qdT, out ftT, out btT, out ecT, out roT,
                out plN, out qdN, out ftN, out btN, out ecN, out roN
            );

            _lastGlobalReward = r;

            Debug.Log(
                $"[FactoryReward(instant)] R_total={r:F3} " +
                $"(PL={plT:F2}, QD={qdT:F2}, FT={ftT}, BT={btT}, EC={ecT:F2}, RO={roT:F2} | " +
                $"PL~={plN:F2}, QD~={qdN:F2}, FT~={ftN:F2}, BT~={btN:F2}, EC~={ecN:F2}, RO~={roN:F2})"
            );

            _nextGlobalRewardLogTime = Time.time + Mathf.Max(0.1f, globalRewardLogInterval);
        }
    }

    // ===================== 관찰 윈도우 API =====================

    /// <summary>
    /// 로봇이 "다음 수리 대상"을 결정하는 시점에 한번 호출해주면 됨.
    /// observationWindow 동안 QD/FT/BT를 샘플링하고,
    /// sink 터널의 throughput delta로 PL(T)를 계산한다.
    /// </summary>
    public void BeginRewardObservation()
    {
        if (!useObservationWindow)
            return;

        _isObserving = true;
        _obsEndTime = Time.time + observationWindow;

        _sumQD = _sumFT = _sumBT = _sumEC = _sumRO = 0f;
        _sampleCount = 0;
        _sinkStartCounts.Clear();

        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;

                // TunnelController에 isSink, totalExitedCount가 있다고 가정
                if (t.isSink)
                    _sinkStartCounts[t] = t.totalExitedCount;
            }
        }
    }

    /// <summary>
    /// 관찰 윈도우 중 매 프레임 호출되어,
    /// QD/FT/BT를 time-average를 위해 누적한다.
    /// </summary>
    void SampleForObservation()
    {
        if (tunnels == null) return;

        int totalQD = 0;
        int faultCount = 0;
        int blockCount = 0;

        foreach (var t in tunnels)
        {
            if (t == null) continue;

            if (t.queue != null)
                totalQD += t.queue.Count;

            if (t.IsFault)
                faultCount++;

            if (t.IsHold || t.IsHalfHold)
                blockCount++;
        }

        _sumQD += totalQD;
        _sumFT += faultCount;
        _sumBT += blockCount;
        // EC/RO는 아직 0으로 둔 상태
        _sampleCount++;
    }

    /// <summary>
    /// 관찰 윈도우가 끝났을 때 호출.
    /// 평균 QD/FT/BT와 sink throughput delta로 PL(T)을 계산하고,
    /// 글로벌 리워드를 한 번 로그로 출력한다.
    /// </summary>
    void FinishObservationAndComputeReward()
    {
        _isObserving = false;

        if (_sampleCount <= 0)
            return;

        // 관찰 윈도우 동안의 시간 평균
        float avgQD = _sumQD / _sampleCount;
        float avgFT = _sumFT / _sampleCount;
        float avgBT = _sumBT / _sampleCount;
        float ecT = 0f;
        float roT = 0f;

        // PL(T): sink 터널들의 throughput 증가량 합
        int totalDeltaExit = 0;
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null || !t.isSink) continue;

                int startCount = 0;
                _sinkStartCounts.TryGetValue(t, out startCount);
                int delta = t.totalExitedCount - startCount;
                if (delta > 0)
                    totalDeltaExit += delta;
            }
        }
        float plT = totalDeltaExit;

        float plN, qdN, ftN, btN, ecN, roN;
        float r = ComputeGlobalRewardFromValues(
            plT, avgQD, avgFT, avgBT, ecT, roT,
            out plN, out qdN, out ftN, out btN, out ecN, out roN
        );

        _lastGlobalReward = r;

        if (debugLogGlobalReward)
        {
            Debug.Log(
                $"[FactoryReward(window)] R={r:F3} | " +
                $"PL(T)={plT:F2}, QD_avg={avgQD:F2}, FT_avg={avgFT:F2}, BT_avg={avgBT:F2} | " +
                $"PL~={plN:F2}, QD~={qdN:F2}, FT~={ftN:F2}, BT~={btN:F2}, EC~={ecN:F2}, RO~={roN:F2}"
            );
        }
    }

    // ===================== 노드 인덱스 =====================

    void BuildNodeIndex()
    {
        nodes.Clear();
        _lastTunnelStates.Clear();
        faultRewards.Clear();

        // 1) Spawner → 노드 등록
        if (spawners != null)
        {
            foreach (var sp in spawners)
            {
                if (sp == null) continue;

                int id = sp.nodeId;  // ProductSpawner에 public int nodeId
                if (id < 0)
                {
                    Debug.LogWarning($"[FactoryEnvManager] Spawner '{sp.name}' 의 nodeId가 설정되지 않음 (<0). 그래프에서 제외.");
                    continue;
                }

                if (nodes.ContainsKey(id))
                {
                    Debug.LogWarning($"[FactoryEnvManager] nodeId={id} 중복! (Spawner '{sp.name}')");
                    continue;
                }

                NodeData data = new NodeData
                {
                    nodeId = id,
                    name = sp.name,
                    isSpawner = true,
                    spawner = sp,
                    tunnel = null,
                    tunnelState = TunnelController.TunnelState.RUN,
                    queueCount = 0,
                    queueCapacity = 0
                };

                nodes.Add(id, data);
            }
        }

        // 2) Tunnel → 노드 등록
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;

                int id = t.nodeId;   // TunnelController에 public int nodeId
                if (id < 0)
                {
                    Debug.LogWarning($"[FactoryEnvManager] Tunnel '{t.name}' 의 nodeId가 설정되지 않음 (<0). 그래프에서 제외.");
                    continue;
                }

                if (nodes.ContainsKey(id))
                {
                    Debug.LogWarning($"[FactoryEnvManager] nodeId={id} 중복! (Tunnel '{t.name}')");
                    continue;
                }

                int qCount = 0;
                int qCap = 0;
                if (t.queue != null)
                {
                    qCount = t.queue.Count;
                    qCap = t.queue.Capacity;
                }

                NodeData data = new NodeData
                {
                    nodeId = id,
                    name = t.name,
                    isSpawner = false,
                    spawner = null,
                    tunnel = t,
                    tunnelState = t.State,
                    queueCount = qCount,
                    queueCapacity = qCap
                };

                nodes.Add(id, data);

                // 터널의 초기 상태를 "이전 상태" 딕셔너리에 저장
                _lastTunnelStates[t] = t.State;
            }
        }
    }

    // ===================== 그래프 간선 빌드 =====================

    void BuildGraphEdges()
    {
        adjacency.Clear();

        // 1) Spawner: nodeId -> firstTunnels[].nodeId
        if (spawners != null)
        {
            foreach (var sp in spawners)
            {
                if (sp == null) continue;
                int fromId = sp.nodeId;
                if (fromId < 0) continue;
                if (!nodes.ContainsKey(fromId)) continue;

                if (!adjacency.TryGetValue(fromId, out var list))
                {
                    list = new List<int>();
                    adjacency.Add(fromId, list);
                }

                if (sp.firstTunnels != null)
                {
                    foreach (var t in sp.firstTunnels)
                    {
                        if (t == null) continue;
                        int toId = t.nodeId;
                        if (toId < 0) continue;
                        if (!nodes.ContainsKey(toId)) continue;

                        if (!list.Contains(toId))
                            list.Add(toId);
                    }
                }
            }
        }

        // 2) Tunnel: nodeId -> nextTunnelsForGraph[]
        if (tunnels != null)
        {
            foreach (var t in tunnels)
            {
                if (t == null) continue;
                int fromId = t.nodeId;
                if (fromId < 0) continue;
                if (!nodes.ContainsKey(fromId)) continue;

                if (!adjacency.TryGetValue(fromId, out var list))
                {
                    list = new List<int>();
                    adjacency.Add(fromId, list);
                }

                var next = t.nextTunnelsForGraph;  // TunnelController에 public 필드
                if (next == null) continue;

                foreach (var child in next)
                {
                    if (child == null) continue;
                    int toId = child.nodeId;
                    if (toId < 0) continue;
                    if (!nodes.ContainsKey(toId)) continue;

                    if (!list.Contains(toId))
                        list.Add(toId);
                }
            }
        }
    }

    // ===================== 상태 갱신 + 고장 리워드 이벤트 =====================

    void UpdateNodeStates()
    {
        // Tunnel 상태/큐 정보만 주기적으로 업데이트
        if (tunnels == null) return;

        foreach (var t in tunnels)
        {
            if (t == null) continue;
            int id = t.nodeId;
            if (id < 0) continue;
            if (!nodes.TryGetValue(id, out var data)) continue;

            if (!data.isSpawner)
            {
                var currentState = t.State;

                // 이전 상태가 있으면 Fault 진입/탈출 감지
                if (_lastTunnelStates.TryGetValue(t, out var prevState))
                {
                    // 비-FAULT → FAULT : 고장 발생
                    if (prevState != TunnelController.TunnelState.FAULT &&
                        currentState == TunnelController.TunnelState.FAULT)
                    {
                        OnTunnelFailed(t);
                    }
                    // FAULT → 비-FAULT : 수리 완료
                    else if (prevState == TunnelController.TunnelState.FAULT &&
                             currentState != TunnelController.TunnelState.FAULT)
                    {
                        OnTunnelRepaired(t);
                    }
                }

                // 현재 상태를 "이전 상태"로 갱신
                _lastTunnelStates[t] = currentState;

                // NodeData 갱신
                data.tunnelState = currentState;

                if (t.queue != null)
                {
                    data.queueCount = t.queue.Count;
                    data.queueCapacity = t.queue.Capacity;
                }
                else
                {
                    data.queueCount = 0;
                    data.queueCapacity = 0;
                }
            }
        }
    }

    // ----- Fault Reward 내부 처리 (per-tunnel) -----

    void OnTunnelFailed(TunnelController t)
    {
        // 테스트용: 고장마다 랜덤 리워드 부여
        float reward = Random.Range(minFaultReward, maxFaultReward);
        faultRewards[t] = reward;
        // Debug.Log($"[FactoryEnvManager] Tunnel FAILED '{t.name}', reward={reward}");
    }

    void OnTunnelRepaired(TunnelController t)
    {
        if (faultRewards.ContainsKey(t))
        {
            faultRewards.Remove(t);
            // Debug.Log($"[FactoryEnvManager] Tunnel REPAIRED '{t.name}', remove reward entry");
        }
    }

    /// <summary>
    /// 현재 고장난 터널들 중에서 리워드가 가장 큰 터널을 반환.
    /// 없으면 null.
    /// (테스트용 정책: 리워드가 클수록 먼저 수리하러 감)
    /// </summary>
    public TunnelController GetBestFaultyTunnel()
    {
        TunnelController best = null;
        float bestReward = float.NegativeInfinity;

        foreach (var kvp in faultRewards)
        {
            if (kvp.Value > bestReward)
            {
                bestReward = kvp.Value;
                best = kvp.Key;
            }
        }

        return best;
    }

    // ----- 글로벌 리워드 계산 (PDF 수식 버전, 단순화) -----

    /// <summary>
    /// 현재 상태에서 PL(T), QD(T), FT(T), BT(T), EC(T), RO(T)를
    /// 단순하게 추정한다.
    /// </summary>
    void ComputeRawMetrics(
        out float PL, out float QD,
        out float FT, out float BT,
        out float EC, out float RO)
    {
        // PL은 기본적으로 0으로 두고,
        // 실제 throughput은 관찰 윈도우 기반으로 plT에서 계산하는 것을 추천.
        PL = 0f;
        QD = 0f;
        FT = 0f;
        BT = 0f;
        EC = 0f;
        RO = 0f;

        foreach (var kv in nodes)
        {
            var n = kv.Value;
            if (n.isSpawner) continue;

            // 큐 길이 합(단순 QD 근사)
            QD += n.queueCount;

            switch (n.tunnelState)
            {
                case TunnelController.TunnelState.FAULT:
                    FT += 1f;
                    break;
                case TunnelController.TunnelState.HOLD:
                case TunnelController.TunnelState.HALF_HOLD:
                    BT += 1f;
                    break;
            }
        }
    }

    /// <summary>
    /// 주어진 PL/QD/FT/BT/EC/RO 값으로부터
    /// 정규화된 항들을 계산하고, 
    /// R_total = w1*PL~ - w2*QD~ - ... 수식을 적용한다.
    /// (관찰 윈도우 / 즉시형 모두 공용으로 사용)
    /// </summary>
    public float ComputeGlobalRewardFromValues(
        float PL, float QD, float FT, float BT, float EC, float RO,
        out float PL_norm, out float QD_norm,
        out float FT_norm, out float BT_norm,
        out float EC_norm, out float RO_norm)
    {
        PL_norm = (maxPL > 0f) ? Mathf.Clamp01(PL / maxPL) : 0f;
        QD_norm = (maxQD > 0f) ? Mathf.Clamp01(QD / maxQD) : 0f;
        FT_norm = (maxFT > 0f) ? Mathf.Clamp01(FT / maxFT) : 0f;
        BT_norm = (maxBT > 0f) ? Mathf.Clamp01(BT / maxBT) : 0f;
        EC_norm = (maxEC > 0f) ? Mathf.Clamp01(EC / maxEC) : 0f;
        RO_norm = (maxRO > 0f) ? Mathf.Clamp01(RO / maxRO) : 0f;

        float reward =
            + w1_PL * PL_norm
            - w2_QD * QD_norm
            - w3_FT * FT_norm
            - w4_BT * BT_norm
            - w5_EC * EC_norm
            - w6_RO * RO_norm;

        return reward;
    }

    /// <summary>
    /// "현재 시점"의 상태로부터 글로벌 리워드를 계산.
    /// (기존 즉시형 로그용, 관찰 윈도우가 아니라 그냥 스냅샷 기준)
    /// </summary>
    public float ComputeGlobalReward(
        out float PL, out float QD,
        out float FT, out float BT,
        out float EC, out float RO,
        out float PL_norm, out float QD_norm,
        out float FT_norm, out float BT_norm,
        out float EC_norm, out float RO_norm)
    {
        ComputeRawMetrics(out PL, out QD, out FT, out BT, out EC, out RO);
        return ComputeGlobalRewardFromValues(
            PL, QD, FT, BT, EC, RO,
            out PL_norm, out QD_norm,
            out FT_norm, out BT_norm,
            out EC_norm, out RO_norm
        );
    }

    /// <summary>
    /// 최근에 계산된 글로벌 리워드 값을 읽고 싶을 때 사용.
    /// (즉시형/윈도우형 둘 중 마지막으로 계산된 값)
    /// </summary>
    public float GetLastGlobalReward()
    {
        return _lastGlobalReward;
    }

    // ===================== Debug 출력 =====================

    void DumpGraphToLog()
    {
        Debug.Log("===== FactoryEnvManager Graph Dump =====");

        foreach (var pair in adjacency)
        {
            int from = pair.Key;
            string fromName = nodes.TryGetValue(from, out var n) ? n.name : "Unknown";

            var list = pair.Value;
            string targets = "";
            for (int i = 0; i < list.Count; i++)
            {
                int to = list[i];
                string toName = nodes.TryGetValue(to, out var nn) ? nn.name : "Unknown";
                targets += $"{to}({toName})";
                if (i < list.Count - 1) targets += ", ";
            }

            Debug.Log($"{from}({fromName}) -> [{targets}]");
        }
    }

    /// <summary>
    /// 한 줄로 전체 노드 상태를 compact하게 출력
    /// 예: [FactoryCompact] 0(Spawner_S1):0 | 1(Tunnel_A):2 Q=3/5 | ...
    /// 상태 코드: RUN=0, HALF_HOLD=1, HOLD=2, FAULT=3
    /// </summary>
    void DumpCompactStatesToLog()
    {
        if (nodes.Count == 0) return;

        List<int> ids = new List<int>(nodes.Keys);
        ids.Sort();

        StringBuilder sb = new StringBuilder();
        sb.Append("[FactoryCompact] ");

        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if (!nodes.TryGetValue(id, out var n)) continue;

            if (n.isSpawner)
            {
                // Spawner는 상태 코드 0으로 통일
                sb.AppendFormat("{0}({1}):0", n.nodeId, n.name);
            }
            else
            {
                int stateCode = StateToInt(n.tunnelState);
                sb.AppendFormat("{0}({1}):{2} Q={3}/{4}",
                    n.nodeId, n.name, stateCode, n.queueCount, n.queueCapacity);
            }

            if (i < ids.Count - 1)
                sb.Append(" | ");
        }

        Debug.Log(sb.ToString());
    }

    int StateToInt(TunnelController.TunnelState s)
    {
        switch (s)
        {
            case TunnelController.TunnelState.RUN:       return 0;
            case TunnelController.TunnelState.HALF_HOLD: return 1;
            case TunnelController.TunnelState.HOLD:      return 2;
            case TunnelController.TunnelState.FAULT:     return 3;
        }
        return -1;
    }

    // ===================== 외부에서 쓸 수 있는 헬퍼 =====================

    /// <summary>
    /// 특정 nodeId의 상태를 얻는다. 존재하지 않으면 null 반환.
    /// </summary>
    public NodeData GetNode(int nodeId)
    {
        nodes.TryGetValue(nodeId, out var n);
        return n;
    }

    /// <summary>
    /// 특정 nodeId에서 나가는 child nodeId 리스트를 얻는다. 없으면 빈 리스트 반환.
    /// </summary>
    public List<int> GetNeighbors(int nodeId)
    {
        if (adjacency.TryGetValue(nodeId, out var list))
            return list;
        return new List<int>();
    }
}
