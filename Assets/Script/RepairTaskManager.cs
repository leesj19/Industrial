using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

using Random = UnityEngine.Random;

public class RepairTaskManager : MonoBehaviour
{
    [Header("Debug 옵션")]
    [Tooltip("로봇 이동 / 수리 상태 디버그 로그")]
    public bool debugRobotFlow = true;
    [Header("로봇 & 대상들")]
    public AStarAgent robot;          // 수리 로봇 (A* 에이전트)
    public List<RepairSite> sites;    // 씬에서 등록할 RepairSite 목록

    [Header("DQN Agent (optional)")]
    [Tooltip("DQN 학습용 transition + 액션 선택을 위한 에이전트 (없으면 DQN 연동 안 함)")]
    public DqnAgent dqnAgent;

    [Header("기본 선택 정책 (DQN 실패 시 사용)")]
    [Tooltip("여러 고장 사이트가 있을 때 무작위로 선택할지 여부 (false면 리스트 순서대로)")]
    public bool chooseRandomWhenMultiple = true;

    [Header("수리 설정")]
    [Tooltip("수리 구역 도착 후 실제 수리에 걸리는 시간(초)")]
    public float repairDuration = 5f;

    [Header("DQN 액션 선택 설정")]
    [Tooltip("true면 가능한 경우 DQN으로 다음 수리 대상 선택")]
    public bool useDqnSelection = true;

    [Range(0f, 1f)]
    [Tooltip("ε-greedy 탐색 비율 (Python 쪽으로 전달)")]
    public float epsilon = 0.3f;

    [Tooltip("ε 최소값")]
    public float epsilonMin = 0.05f;

    [Tooltip("수리 한 번 끝날 때마다 ε *= epsilonDecay")]
    public float epsilonDecay = 0.999f;

    RepairSite currentTarget;
    readonly List<RepairSite> pendingSites = new List<RepairSite>();
    bool robotBusy = false;

    void Awake()
    {
        if (robot != null)
        {
            robot.OnPathFinished += HandleRobotArrived;
        }
    }

    void OnDestroy()
    {
        if (robot != null)
        {
            robot.OnPathFinished -= HandleRobotArrived;
        }
    }

    void Update()
    {
        if (robot == null || sites == null) return;

        // 1) 수리가 필요한 사이트들 스캔해서 pending 큐에 쌓기
        ScanSites();

        // 2) 로봇이 놀고 있으면 다음 작업 배정
        TryAssignNextTask();
    }

    /// <summary>
    /// 전체 RepairSite를 돌면서 NeedsRepair == true 인 곳을 pending 큐에 추가.
    /// 이미 isQueued=true 인 사이트는 중복 추가하지 않음.
    /// </summary>
    void ScanSites()
    {
        foreach (var s in sites)
        {
            if (s == null) continue;

            // 이미 큐에 들어간 애면 스킵
            if (s.isQueued) continue;

            if (s.NeedsRepair)
            {
                s.isQueued = true;
                pendingSites.Add(s);
            }
        }
    }

    /// <summary>
    /// 로봇이 비어 있고, pendingSites에 수리할 곳이 있으면
    /// DQN으로 한 곳을 선택하거나 (가능하다면),
    /// 실패 시 기본 랜덤/순차 정책으로 하나 골라서 이동 명령.
    /// </summary>
    void TryAssignNextTask()
    {
        if (robotBusy) return;
        if (pendingSites.Count == 0) return;

        bool canUseDqn =
            useDqnSelection &&
            dqnAgent != null &&
            dqnAgent.tcpClient != null &&
            dqnAgent.tcpClient.IsConnected;

        if (canUseDqn)
        {
            // ---- DQN 후보 nodeId 리스트 구성 ----
            List<int> candidateNodeIds = new List<int>();
            foreach (var s in pendingSites)
            {
                if (s != null && s.tunnel != null)
                {
                    candidateNodeIds.Add(s.tunnel.nodeId);
                }
            }

            if (candidateNodeIds.Count == 0)
            {
                // 안전 장치: 터널 없는 사이트들이라면 그냥 기본 정책으로
                if (dqnAgent.debugLogs)
                    Debug.LogWarning("[RepairTaskManager] DQN candidateNodeIds 비어있음 → 기본 정책 사용");
                PickAndAssignLocalPolicy();
                return;
            }

            // 두 번 배정되는 것 방지
            robotBusy = true;

            if (dqnAgent.debugLogs)
            {
                Debug.Log($"[RepairTaskManager] DQN 액션 요청: candidates=[{string.Join(",", candidateNodeIds)}], eps={epsilon:F3}");
            }

            // DQN 에게 액션 요청 (코루틴)
            StartCoroutine(dqnAgent.CoRequestActionAndPickNode(
                candidateNodeIds,
                epsilon,
                (chosenNodeId, isRandomFromEps) =>
                {
                    // 이 콜백은 메인 스레드에서 실행됨

                    // ε decay (성공/실패와 상관없이 한 스텝 끝났다고 보고 감소)
                    epsilon = Mathf.Max(epsilonMin, epsilon * epsilonDecay);

                    // chosenNodeId가 비정상이면 fallback
                    if (chosenNodeId < 0)
                    {
                        if (dqnAgent.debugLogs)
                            Debug.LogWarning("[RepairTaskManager] chosenNodeId < 0 → 기본 정책 fallback");
                        robotBusy = false;
                        PickAndAssignLocalPolicy();
                        return;
                    }

                    // chosenNodeId에 대응하는 RepairSite 찾기
                    currentTarget = null;
                    for (int i = 0; i < pendingSites.Count; i++)
                    {
                        var s = pendingSites[i];
                        if (s != null && s.tunnel != null && s.tunnel.nodeId == chosenNodeId)
                        {
                            currentTarget = s;
                            pendingSites.RemoveAt(i);
                            break;
                        }
                    }

                    if (currentTarget == null)
                    {
                        if (dqnAgent.debugLogs)
                            Debug.LogWarning($"[RepairTaskManager] chosenNodeId={chosenNodeId} 에 해당하는 pending site 없음 → 기본 정책 fallback");
                        robotBusy = false;
                        PickAndAssignLocalPolicy();
                        return;
                    }

                    if (dqnAgent.debugLogs)
                    {
                        Debug.Log($"[RepairTaskManager] DQN 선택 nodeId={chosenNodeId}, epsRandom={isRandomFromEps}");
                    }

                    // 로봇 이동 시작
                    MoveRobotToCurrentTarget();
                }));
        }
        else
        {
            // DQN 사용 불가 → 기존 정책
            PickAndAssignLocalPolicy();
        }
    }

    /// <summary>
    /// DQN을 쓰지 못할 때 사용하는 기존 랜덤/순차 정책.
    /// </summary>
    void PickAndAssignLocalPolicy()
    {
        if (pendingSites.Count == 0)
        {
            robotBusy = false;
            return;
        }

        int idx = 0;
        if (chooseRandomWhenMultiple && pendingSites.Count > 1)
        {
            idx = Random.Range(0, pendingSites.Count);
        }

        currentTarget = pendingSites[idx];
        pendingSites.RemoveAt(idx);

        if (currentTarget == null)
        {
            robotBusy = false;
            return;
        }

        if (dqnAgent != null && dqnAgent.debugLogs)
        {
            if (currentTarget.tunnel != null)
            {
                Debug.Log($"[RepairTaskManager] Local policy로 nodeId={currentTarget.tunnel.nodeId} 선택");
            }
            else
            {
                Debug.Log("[RepairTaskManager] Local policy 선택 (tunnel null)");
            }
        }

        robot.SetTarget(currentTarget.RepairPoint, true);
        robotBusy = true;
    }

    /// <summary>
    /// AStarAgent가 현재 target까지 경로를 모두 따라간 뒤 호출되는 콜백.
    /// 여기서 수리 코루틴 시작.
    /// </summary>
    void HandleRobotArrived()
    {
        if (currentTarget != null)
        {
            StartCoroutine(CoRepairCurrentTarget(currentTarget));
        }
        else
        {
            robotBusy = false;
            TryAssignNextTask();
        }
    }
    void MoveRobotToCurrentTarget()
    {
        if (robot == null || currentTarget == null || currentTarget.RepairPoint == null)
        {
            robotBusy = false;
            return;
        }

        // ① 로봇 위치와 목표 위치 거리 계산 (y축은 무시)
        Vector3 robotPos = robot.transform.position;
        Vector3 targetPos = currentTarget.RepairPoint.position;
        robotPos.y = 0f;
        targetPos.y = 0f;

        float dist = Vector3.Distance(robotPos, targetPos);

        int nodeId = (currentTarget.tunnel != null) ? currentTarget.tunnel.nodeId : -1;

        if (debugRobotFlow)
        {
            Debug.Log($"[RepairTaskManager] MoveRobotToCurrentTarget: nodeId={nodeId}, dist={dist:F3}");
        }

        // 🔹 1) "완전 같은 자리"면 그냥 바로 수리 시작
        const float immediateRepairThreshold = 0.05f;   // 거의 같은 위치(5cm 이내)라고 보는 기준

        if (dist <= immediateRepairThreshold)
        {
            if (debugRobotFlow)
            {
                Debug.Log($"[RepairTaskManager] 로봇이 이미 터널 {nodeId} 수리 위치에 있음 → 즉시 재수리 시작");
            }

            robotBusy = true;   // 이 수리 동안은 바쁘다고 표시
            StartCoroutine(CoRepairCurrentTarget(currentTarget));
            return;
        }

        // 🔹 2) 조금 떨어져 있긴 하지만, 거의 근처라면 도착으로 처리 (선택사항)
        const float arriveThreshold = 0.6f;  // 필요에 따라 0.5~1.0 사이로 조절

        if (dist <= arriveThreshold)
        {
            if (debugRobotFlow)
            {
                Debug.Log($"[RepairTaskManager] 로봇이 터널 {nodeId} 근처({dist:F3}) → HandleRobotArrived()로 처리");
            }

            HandleRobotArrived();
            return;
        }

        // 🔹 3) 그 외에는 정상적으로 A* 경로 따라 이동
        if (debugRobotFlow)
        {
            Debug.Log("[RepairTaskManager] 로봇에 SetTarget 호출 (A* 이동)");
        }

        robot.SetTarget(currentTarget.RepairPoint, true);
        robotBusy = true;
    }




    /// <summary>
    /// 로봇이 도착한 뒤 repairDuration 만큼 기다렸다가 실제 수리 실행.
    /// 이 동안 RepairSite의 게이지(visual)를 채운다.
    /// </summary>
    IEnumerator CoRepairCurrentTarget(RepairSite site)
    {
        float wait = Mathf.Max(0f, repairDuration);

        if (site != null)
        {
            // 🔹 수리 시작: 이제는 DQN 상태 기록 안 함
            site.BeginRepairVisual();
        }

        float elapsed = 0f;
        while (elapsed < wait)
        {
            elapsed += Time.deltaTime;

            if (site != null && wait > 0f)
            {
                float progress = Mathf.Clamp01(elapsed / wait);
                site.UpdateRepairVisual(progress);
            }

            yield return null;
        }

        if (site != null)
        {
            // 🔹 실제 수리 완료 시점
            site.OnRepaired();
            site.EndRepairVisual();
            site.isQueued = false;

            // ==== DQN 연동: "수리 끝난 직후"에서 s_t 기록 + 윈도우 시작 ====
            if (dqnAgent != null && site.tunnel != null)
            {
                int nodeId = site.tunnel.nodeId;
                int actionId = nodeId;   // 현재는 nodeId를 액션 ID처럼 사용

                // 여기서 s_t 스냅샷
                dqnAgent.RecordAction(actionId, nodeId);
                // 그리고 관찰 윈도우 시작 + T초 뒤 s_{t+1}, r_t 전송
                dqnAgent.FinishStepAndSend();
            }
        }

        currentTarget = null;
        robotBusy = false;

        // 수리 끝난 뒤, 아직 처리 안 한 다른 고장이 있으면 바로 다음 목적지 배정
        TryAssignNextTask();
    }

}