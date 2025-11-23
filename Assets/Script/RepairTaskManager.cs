using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class RepairTaskManager : MonoBehaviour
{
    [Header("로봇 & 대상들")]
    public AStarAgent robot;          // 수리 로봇 (A* 에이전트)
    public List<RepairSite> sites;    // 씬에서 등록할 RepairSite 목록

    [Header("선택 정책")]
    [Tooltip("여러 고장 사이트가 있을 때 무작위로 선택할지 여부 (false면 리스트 순서대로)")]
    public bool chooseRandomWhenMultiple = true;

    [Header("수리 설정")]
    [Tooltip("수리 구역 도착 후 실제 수리에 걸리는 시간(초)")]
    public float repairDuration = 5f;

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
    /// 로봇이 비어 있고, pendingSites에 수리할 곳이 있으면 하나 골라서 이동 명령.
    /// </summary>
    void TryAssignNextTask()
    {
        if (robotBusy) return;
        if (pendingSites.Count == 0) return;

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

        // 로봇의 목표를 이 RepairSite의 RepairPoint로 설정
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

    /// <summary>
    /// 로봇이 도착한 뒤 repairDuration 만큼 기다렸다가 실제 수리 실행.
    /// 이 동안 RepairSite의 게이지(visual)를 채운다.
    /// </summary>
    IEnumerator CoRepairCurrentTarget(RepairSite site)
    {
        float wait = Mathf.Max(0f, repairDuration);

        if (site != null)
        {
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
            site.OnRepaired();
            site.EndRepairVisual();
        }

        currentTarget = null;
        robotBusy = false;

        // 수리 끝난 뒤, 아직 처리 안 한 다른 고장이 있으면 바로 다음 목적지 배정
        TryAssignNextTask();
    }
}
