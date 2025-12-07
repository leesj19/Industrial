using System.Collections.Generic;
using UnityEngine;

/// <summary>
///  - 15개 터널 중 3개를 랜덤으로 골라 동시에 고장(FORCE FAIL) 시킨다.
///  - 로봇이 3개 전부 수리할 때까지 기다린다.
///  - 3개가 모두 수리되면 10초 대기 후 다시 3개 랜덤 고장.
///  - 이걸 무한 반복.
/// </summary>
public class RandomFailureScenario : MonoBehaviour
{
    [Header("References")]
    public FactoryEnvManager factoryEnv;   // 인스펙터에서 할당 (없으면 자동 탐색)

    [Header("Scenario Settings")]
    [Tooltip("한 시나리오에서 동시에 고장낼 터널 개수")]
    public int faultsPerScenario = 3;

    [Tooltip("모든 고장이 수리된 후 다음 시나리오까지 대기 시간 (초)")]
    public float delayBetweenScenarios = 10f;

    [Header("Debug")]
    public bool debugLogs = true;

    // 내부 상태
    private List<TunnelController> allTunnels = new List<TunnelController>();
    private List<TunnelController> currentFaults = new List<TunnelController>();
    private bool scenarioActive = false;
    private float nextScenarioStartTime = 0f;

        void Start()
    {
        if (factoryEnv == null)
        {
            factoryEnv = FactoryEnvManager.Instance;
        }

        if (factoryEnv == null)
        {
            Debug.LogError("[RandomFailureScenario] FactoryEnvManager를 찾을 수 없습니다.");
            enabled = false;
            return;
        }

        // 공장에 등록된 터널 리스트 가져오기 + 이 시나리오 동안은 확률 고장 끄기
        if (factoryEnv.tunnels != null)
        {
            foreach (var t in factoryEnv.tunnels)
            {
                if (t != null && t.nodeId >= 0)
                {
                    // 시나리오 모드에서는 기존 items-based 확률 고장을 잠깐 꺼둔다
                    t.useItemFailure = false;

                    allTunnels.Add(t);
                }
            }
        }

        if (allTunnels.Count == 0)
        {
            Debug.LogError("[RandomFailureScenario] 사용할 터널이 없습니다.");
            enabled = false;
            return;
        }

        // 게임 시작하자마자 첫 시나리오 바로 시작
        StartNewScenario();
    }

    void Update()
    {
        if (!scenarioActive)
        {
            // 시나리오 사이 대기 중
            if (Time.time >= nextScenarioStartTime)
            {
                StartNewScenario();
            }
            return;
        }

        // 시나리오 진행 중이면 "3개가 모두 수리됐는지" 확인
        bool allRepaired = true;

        foreach (var t in currentFaults)
        {
            if (t == null) continue;

            // 아직 FAULT 상태인 터널이 하나라도 있으면 시나리오 계속
            if (t.IsFault)
            {
                allRepaired = false;
                break;
            }
        }

        if (allRepaired)
        {
            // 이 시나리오 종료 → 다음 시나리오 시작 시간 예약
            scenarioActive = false;
            nextScenarioStartTime = Time.time + delayBetweenScenarios;

            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] 모든 시나리오 고장 수리 완료 → {delayBetweenScenarios:F1}s 뒤 다음 시나리오 시작");
            }
        }
    }

    void StartNewScenario()
    {
        if (allTunnels.Count < faultsPerScenario)
        {
            Debug.LogWarning("[RandomFailureScenario] faultsPerScenario가 터널 개수보다 많습니다. 가능한 만큼만 사용합니다.");
        }

        // 이전 시나리오 정보 리셋
        currentFaults.Clear();

        // 랜덤으로 섞기 (Fisher-Yates)
        List<TunnelController> shuffled = new List<TunnelController>(allTunnels);
        int n = shuffled.Count;
        for (int i = 0; i < n; i++)
        {
            int j = Random.Range(i, n);
            var tmp = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = tmp;
        }

        int count = Mathf.Min(faultsPerScenario, shuffled.Count);

        for (int i = 0; i < count; i++)
        {
            TunnelController t = shuffled[i];
            if (t == null) continue;

            // 자동 수리 끄기 (이 시나리오 동안은 로봇만 수리하게)
            t.autoRepairFixedDelay = false;

            // 이미 고장 상태면 한번 수리해두고 다시 고장낼 수도 있음
            if (t.IsFault)
            {
                t.ForceRepair();
            }

            // 이 터널을 고장 상태로 만들기
            t.ForceFail();

            currentFaults.Add(t);

            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] 시나리오 고장 터널 선택: nodeId={t.nodeId}, name={t.name}");
            }
        }

        scenarioActive = true;

        if (debugLogs)
        {
            Debug.Log($"[RandomFailureScenario] 새 시나리오 시작: 고장 터널 {currentFaults.Count}개");
        }
    }
}
