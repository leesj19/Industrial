using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// - FactoryEnvManager에서 상태(state 벡터)를 만들고
/// - FactoryEnvManager에서 글로벌 리워드를 읽어서
/// - transition JSON을 만들어내는 에이전트.
///
/// state 벡터 구성:
///   1) 터널 노드들만 nodeId 오름차순으로 정렬해서,
///      각 노드마다 다음 5개 feature를 순서대로 붙임:
///        [ stateIndex, queueCount, queueCapacity, isSinkFlag, supplyCount ]
///        - stateIndex : RUN=0, HALF_HOLD=1, HOLD=2, FAULT=3
///        - queueCount / queueCapacity : 원값 (비율 X)
///        - isSinkFlag : tunnel.isSink ? 1 : 0
///        - supplyCount : isSink이면 totalExitedCount, 아니면 0
///
///   2) 같은 터널 노드 순서로 adjacency matrix (N x N)를 row-major로 플랫하게 추가:
///        edge(i,j) = 1 if i->j edge exists (FactoryEnvManager.Adjacency 기준), else 0
///
/// RL 관점 시점 정의:
///   - RecordAction() : "수리 시작 시점"에 호출 → s_t 스냅샷
///   - FinishStepAndSend() : "수리 끝난 직후"에 호출 → 내부에서
///       BeginRewardObservation() → T초 기다린 뒤
///       GetLastGlobalReward()와 s_{t+1}를 이용해 transition 생성
/// </summary>
public class DqnAgent : MonoBehaviour
{
    [Header("Environment References")]
    public FactoryEnvManager factoryEnv;         // 상태/리워드 정보 소스
    public RepairTaskManager repairTaskManager;  // (선택) 나중에 액션 선택과 연결할 때 사용

    [Header("Network / TCP (optional)")]
    [Tooltip("Python DQN 서버와 통신할 TCP 클라이언트 (없으면 콘솔 로그만)")]
    public DqnTcpClient tcpClient;
    [Tooltip("true면 Transition JSON을 TCP로도 전송")]
    public bool sendTransitionOverTcp = false;

    [Header("Debug 옵션")]
    public bool debugLogs = true;

    [Tooltip("state 벡터 내부 값까지 상세하게 찍을지 여부")]
    public bool logStateVector = true;

    [Tooltip("state 벡터를 찍을 때 앞에서 몇 개까지만 출력할지")]
    public int maxStateElementsToLog = 32;

    [Tooltip("중복 RecordAction 호출 시 스택트레이스를 찍을지 여부")]
    public bool logStackTraceOnDuplicateRecord = true;

    // ===== Transition 버퍼 =====
    float[] lastState;     // s_t (수리 시작 순간의 상태)
    int lastActionId = -1; // "몇 번째 액션"인지 (액션 인덱스 등)
    int lastNodeId = -1;   // 실제로 수리하러 간 nodeId (터널 nodeId)
    bool hasPendingTransition = false;

    // 디버그용 transition 카운터
    int transitionStepCounter = 0;

    // Python 쪽으로 보낼 JSON 구조와 동일한 형태
    [Serializable]
    public class TransitionMessage
    {
        public string type = "transition";
        public int action_id;
        public int node_id;
        public float reward;
        public float[] state_t;
        public float[] state_tp1;
    }

    void Awake()
    {
        if (factoryEnv == null)
        {
            factoryEnv = FactoryEnvManager.Instance; // FactoryEnvManager.Singleton 사용
        }
    }

    // =========================
    //  외부에서 호출할 API
    // =========================

    /// <summary>
    /// "수리를 시작하는 순간"에 한 번 호출.
    /// - 현재 상태 s_t를 저장하고
    /// - 어떤 액션/노드를 선택했는지 기억해둔다.
    /// (실제 호출 위치는 RepairTaskManager.CoRepairCurrentTarget 시작부)
    /// </summary>
    public void RecordAction(int actionId, int nodeId)
    {
        if (factoryEnv == null)
        {
            Debug.LogError("[DqnAgent] FactoryEnvManager 참조가 없습니다.");
            return;
        }

        // *** 중요: 이전 transition이 아직 안 끝났는데 다시 RecordAction이 호출되는 경우 감지 ***
        if (hasPendingTransition)
        {
            Debug.LogWarning(
                $"[DqnAgent] RecordAction이 이전 transition이 끝나기 전에 다시 호출됨. " +
                $"(prevActionId={lastActionId}, prevNodeId={lastNodeId} → newActionId={actionId}, newNodeId={nodeId})"
            );

            if (logStackTraceOnDuplicateRecord)
            {
                var st = new StackTrace(true);
                Debug.Log($"[DqnAgent] 중복 RecordAction 호출 스택트레이스:\n{st}");
            }
            // 여기서는 가장 최근 액션 기준으로 덮어쓰는 쪽을 택함.
        }

        // === s_t 스냅샷 ===
        lastState = BuildStateVector(); // s_t
        lastActionId = actionId;
        lastNodeId = nodeId;
        hasPendingTransition = true;

        if (debugLogs)
        {
            Debug.Log($"[DqnAgent] RecordAction (수리 시작): actionId={actionId}, nodeId={nodeId}, state_dim={lastState?.Length ?? 0}");
            if (logStateVector)
            {
                DebugLogState("s_t", lastState);
            }
        }

        // 관찰 윈도우는 "수리 끝난 직후"에 시작할 것이라 여기서는 호출하지 않음.
    }

    /// <summary>
    /// "수리가 끝난 직후"에 호출.
    /// - 여기서 관찰 윈도우 T를 시작하고
    /// - T초 지난 뒤 FactoryEnvManager가 계산한 윈도우 리워드와
    ///   s_{t+1} 상태를 읽어 transition을 완성한다.
    /// (실제 호출 위치는 RepairTaskManager.CoRepairCurrentTarget 끝부분)
    /// </summary>
    public void FinishStepAndSend()
    {
        if (!hasPendingTransition)
            return;

        if (factoryEnv == null)
        {
            Debug.LogError("[DqnAgent] FactoryEnvManager 참조가 없습니다.");
            return;
        }

        // 관찰 윈도우를 사용하는 경우: 여기서 시작 신호
        if (factoryEnv.useObservationWindow)
        {
            factoryEnv.BeginRewardObservation();
            if (debugLogs)
            {
                Debug.Log($"[DqnAgent] BeginRewardObservation() 호출 - window T={factoryEnv.observationWindow:F2}s");
            }
        }

        // 실제 전송은 코루틴에서 window T만큼 지난 뒤 수행
        StartCoroutine(CoWaitWindowAndSend());
    }

    /// <summary>
    /// 관찰 윈도우 T초를 기다린 뒤,
    /// FactoryEnvManager가 계산한 윈도우 리워드를 읽고
    /// s_{t+1}을 스냅샷해서 transition을 생성/전송한다.
    /// </summary>
    IEnumerator CoWaitWindowAndSend()
    {
        float waitT = 0f;

        if (factoryEnv.useObservationWindow)
        {
            waitT = Mathf.Max(0f, factoryEnv.observationWindow);
        }

        if (waitT > 0f)
        {
            float endTime = Time.time + waitT;
            while (Time.time < endTime)
            {
                yield return null;
            }
        }
        else
        {
            // 윈도우 기능을 끈 경우 최소 한 프레임은 양보
            yield return null;
        }

        // === 리워드 계산 ===
        float reward;
        if (factoryEnv.useObservationWindow)
        {
            // 관찰 윈도우 기반 마지막 글로벌 리워드 사용
            reward = factoryEnv.GetLastGlobalReward();
        }
        else
        {
            // fallback: 즉시형 리워드
            reward = GetInstantReward();
        }

        // === s_{t+1} 상태 스냅샷 ===
        float[] nextState = BuildStateVector();

        transitionStepCounter++;

        var msg = new TransitionMessage
        {
            action_id = lastActionId,
            node_id = lastNodeId,
            reward = reward,
            state_t = lastState,
            state_tp1 = nextState
        };

        string json = JsonUtility.ToJson(msg);

        if (debugLogs)
        {
            Debug.Log(
                $"[DqnAgent] FinishStepAndSend(step={transitionStepCounter}) " +
                $"actionId={lastActionId}, nodeId={lastNodeId}, reward={reward:F3}"
            );
            Debug.Log(
                $"[DqnAgent] state_t_dim={lastState?.Length ?? 0}, " +
                $"state_tp1_dim={nextState?.Length ?? 0}"
            );

            if (logStateVector)
            {
                DebugLogState("s_tp1", nextState);
            }

            // 노드별 상태 요약도 같이 찍기 (큐/상태 확인용)
            DebugLogNodeSnapshot();

            // JSON 구조까지 확인하고 싶으면:
            Debug.Log($"[DqnAgent] Transition JSON (preview): {json}");
        }

        if (sendTransitionOverTcp && tcpClient != null)
        {
            try
            {
                tcpClient.SendJsonLine(json);
            }
            catch (Exception e)
            {
                Debug.LogError($"[DqnAgent] TCP 전송 중 예외 발생: {e}");
            }
        }

        hasPendingTransition = false;
    }

    // =========================
    //   State / Reward 생성부
    // =========================

    /// <summary>
    /// FactoryEnvManager.Nodes / Adjacency / TunnelController 정보를 이용해
    /// state 벡터를 구성.
    ///
    /// 1) 터널 노드들만 nodeId 오름차순으로 정렬해서,
    ///    각 노드마다 다음 5개 feature를 붙임:
    ///       [ stateIndex, queueCount, queueCapacity, isSinkFlag, supplyCount ]
    /// 2) 같은 터널 순서로 adjacency matrix (N x N, row-major)를 플랫하게 이어붙임.
    /// </summary>
    float[] BuildStateVector()
    {
        if (factoryEnv == null || factoryEnv.Nodes == null)
            return Array.Empty<float>();

        var nodesDict = factoryEnv.Nodes; // nodeId -> NodeData 맵

        // 1) 터널 노드만 추려서 nodeId 오름차순 정렬
        List<int> tunnelNodeIds = new List<int>();
        foreach (var kv in nodesDict)
        {
            if (!kv.Value.isSpawner)
            {
                tunnelNodeIds.Add(kv.Key);
            }
        }

        tunnelNodeIds.Sort();
        int n = tunnelNodeIds.Count;

        // nodeId -> index (0..n-1) 매핑
        Dictionary<int, int> indexOfNode = new Dictionary<int, int>(n);
        for (int i = 0; i < n; i++)
        {
            indexOfNode[tunnelNodeIds[i]] = i;
        }

        List<float> features = new List<float>();

        // ----- (1) per-node feature 5개씩 -----
        for (int i = 0; i < n; i++)
        {
            int nodeId = tunnelNodeIds[i];
            if (!nodesDict.TryGetValue(nodeId, out var data))
                continue;

            // 상태 인덱스: RUN=0, HALF_HOLD=1, HOLD=2, FAULT=3
            int stateIndex = 0;
            switch (data.tunnelState)
            {
                case TunnelController.TunnelState.HALF_HOLD:
                    stateIndex = 1;
                    break;
                case TunnelController.TunnelState.HOLD:
                    stateIndex = 2;
                    break;
                case TunnelController.TunnelState.FAULT:
                    stateIndex = 3;
                    break;
                case TunnelController.TunnelState.RUN:
                default:
                    stateIndex = 0;
                    break;
            }

            int qCount = data.queueCount;
            int qCap   = data.queueCapacity;

            bool isSink = false;
            int supplyCount = 0;

            if (data.tunnel != null)
            {
                isSink = data.tunnel.isSink;
                if (isSink)
                {
                    // 지금까지 배출된 제품 개수 (sink throughput)
                    supplyCount = data.tunnel.totalExitedCount;
                }
            }

            features.Add((float)stateIndex);   // state index
            features.Add((float)qCount);       // queue count (원값)
            features.Add((float)qCap);         // queue capacity (원값)
            features.Add(isSink ? 1f : 0f);    // isSink flag
            features.Add((float)supplyCount);  // 공급량/throughput (sink만 의미 있음)
        }

        // ----- (2) adjacency matrix (N x N, row-major) -----
        var adjacency = factoryEnv.Adjacency; // nodeId 기반 인접 리스트

        for (int i = 0; i < n; i++)
        {
            int fromId = tunnelNodeIds[i];

            List<int> children = null;
            if (adjacency != null)
            {
                adjacency.TryGetValue(fromId, out children);
            }

            for (int j = 0; j < n; j++)
            {
                int toId = tunnelNodeIds[j];
                float edge = 0f;

                if (children != null && children.Count > 0)
                {
                    // 간단하게 Contains 체크 (터널 수가 많지 않으니 성능 문제 거의 없음)
                    if (children.Contains(toId))
                        edge = 1f;
                }

                features.Add(edge);
            }
        }

        return features.ToArray();
    }

    /// <summary>
    /// "지금 이 순간"의 글로벌 리워드 계산.
    /// (윈도우 기능을 끈 경우 fallback 용도로만 사용)
    /// </summary>
    float GetInstantReward()
    {
        float PL, QD, FT, BT, EC, RO;
        float PLn, QDn, FTn, BTn, ECn, ROn;

        float r = factoryEnv.ComputeGlobalReward(
            out PL, out QD, out FT, out BT, out EC, out RO,
            out PLn, out QDn, out FTn, out BTn, out ECn, out ROn
        );

        if (debugLogs)
        {
            Debug.Log($"[DqnAgent] Instant Reward={r:F3} (PL={PL:F1}, QD={QD:F1}, FT={FT}, BT={BT})");
        }

        return r;
    }

    // =========================
    //      Debug Helper들
    // =========================

    /// <summary>
    /// state 벡터 앞부분만 잘라서 보기 좋게 출력.
    /// </summary>
    void DebugLogState(string label, float[] state)
    {
        if (state == null)
        {
            Debug.Log($"[DqnAgent] {label}: null");
            return;
        }

        int len = state.Length;
        int n = Mathf.Min(len, Mathf.Max(1, maxStateElementsToLog));

        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendFormat("[DqnAgent] {0} (len={1}) first {2} elems: [", label, len, n);
        for (int i = 0; i < n; i++)
        {
            sb.Append(state[i].ToString("0.000"));
            if (i < n - 1) sb.Append(", ");
        }
        if (len > n) sb.Append(" ...");
        sb.Append("]");

        Debug.Log(sb.ToString());
    }

    /// <summary>
    /// 각 노드별 nodeId / 이름 / 큐길이 / 용량 / 상태를 한 줄씩 찍어줌.
    /// 상태/리워드가 이상할 때 실제 파이프라인 상황 확인용.
    /// </summary>
    void DebugLogNodeSnapshot()
    {
        if (factoryEnv == null || factoryEnv.Nodes == null)
            return;

        var nodesDict = factoryEnv.Nodes;
        List<int> nodeIds = new List<int>(nodesDict.Keys);
        nodeIds.Sort();

        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendLine("[DqnAgent] Node snapshot --------------------");

        foreach (int id in nodeIds)
        {
            if (!nodesDict.TryGetValue(id, out var data))
                continue;

            if (data.isSpawner)
            {
                sb.AppendFormat("  [Spawner] id={0}, name={1}\n", id, data.name);
                continue;
            }

            float cap = Mathf.Max(1, data.queueCapacity);
            float fillRatio = Mathf.Clamp01((float)data.queueCount / cap);

            bool isSink = (data.tunnel != null && data.tunnel.isSink);
            int supplyCount = (isSink && data.tunnel != null) ? data.tunnel.totalExitedCount : 0;

            sb.AppendFormat(
                "  [Tunnel] id={0}, name={1}, Q={2}/{3} (fill={4:0.00}), State={5}, isSink={6}, supply={7}\n",
                id, data.name, data.queueCount, data.queueCapacity, fillRatio,
                data.tunnelState, isSink ? 1 : 0, supplyCount
            );
        }

        Debug.Log(sb.ToString());
    }
}
