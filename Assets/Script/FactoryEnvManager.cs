using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>
/// 공장 전체의 Spawner / Tunnel을 스캔해서
/// - nodeId 기준으로 상태(State, Q, Capacity)를 모으고
/// - nodeId 기준 그래프(인접 리스트)를 만든다.
/// 나중에 RL / Python 브릿지에서 이 매니저만 바라보면 됨.
/// </summary>
public class FactoryEnvManager : MonoBehaviour
{
    [Header("Scene References (비워두면 자동 찾기)")]
    public ProductSpawner[] spawners;
    public TunnelController[] tunnels;

    // nodeId -> NodeData
    private Dictionary<int, NodeData> nodes = new Dictionary<int, NodeData>();

    // nodeId -> 나가는 child nodeId 리스트 (그래프 인접 리스트)
    private Dictionary<int, List<int>> adjacency = new Dictionary<int, List<int>>();

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
    }

    // ===================== 노드 인덱스 =====================

    void BuildNodeIndex()
    {
        nodes.Clear();

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
        //    (DownstreamTunnels/branchChildren는 HOLD 전파용이고,
        //     그래프는 인스펙터의 nextTunnelsForGraph만 사용)
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

                var next = t.nextTunnelsForGraph;  // <-- TunnelController에 public 필드
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

    // ===================== 상태 갱신 =====================

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
                data.tunnelState = t.State;

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
