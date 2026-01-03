using System.Collections.Generic;
using UnityEngine;

/// <summary>
///  - 터널 고장 시나리오를 A/B/C 3종류로 운용한다.
///    A: 3개 동시 고장, B: 2개 동시 고장, C: 1개 고장.
///  - 한 "사이클" 안에서 A/B/C가 각각 cycleCountA / B / C 번씩 등장하도록 함.
///    (예: 5,5,5 로 두면 3*5 + 2*5 + 1*5 = 30 step)
///  - 각 시나리오 타입의 순서는 랜덤이지만, 개수는 정확히 맞춘다.
///  - 한 시나리오에서 발생한 FAULT가 모두 수리되면 delayBetweenScenarios 후 다음 시나리오 시작.
///
///  + Clean 시나리오 D:
///    - 일반 시나리오가 cleanEveryNScenarios 번 끝날 때마다 다음은 Clean(D)
///    - Clean 동안: 고장 0개, 스폰 강제 정지, cleanDuration 동안 대기
///    - 종료 후: 스폰 재개 + (추가) Clean 시작 전 스포너 상태 복원
/// </summary>
public class RandomFailureScenario : MonoBehaviour
{
    [Header("References")]
    public FactoryEnvManager factoryEnv;   // 인스펙터에서 할당 (없으면 자동 탐색)

    [Header("Scenario Settings")]
    [Tooltip("A: 3개 고장 시나리오가 한 사이클에서 몇 번 등장할지")]
    public int cycleCountA = 5;   // A: 3 faults
    [Tooltip("B: 2개 고장 시나리오가 한 사이클에서 몇 번 등장할지")]
    public int cycleCountB = 5;   // B: 2 faults
    [Tooltip("C: 1개 고장 시나리오가 한 사이클에서 몇 번 등장할지")]
    public int cycleCountC = 5;   // C: 1 fault

    [Tooltip("모든 고장이 수리된 후 다음 시나리오까지 대기 시간 (초)")]
    public float delayBetweenScenarios = 10f;

    [Header("Clean Scenario (D)")]
    [Tooltip("일반 시나리오가 이 횟수만큼 끝나면 다음 시나리오는 Clean(D)로 실행 (예: 5면 1~5 후 6번째가 Clean)")]
    public int cleanEveryNScenarios = 5;

    [Tooltip("Clean(D) 지속 시간 (초)")]
    public float cleanDuration = 30f;

    [Tooltip("Clean(D) 동안 강제 스폰 정지/재개할 스포너 목록 (비워두면 씬에서 자동 탐색)")]
    public List<ProductSpawner> spawners = new List<ProductSpawner>();

    [Header("Debug")]
    public bool debugLogs = true;

    // 내부 상태
    private List<TunnelController> allTunnels = new List<TunnelController>();
    private List<TunnelController> currentFaults = new List<TunnelController>();
    private bool scenarioActive = false;
    private float nextScenarioStartTime = 0f;

    // 사이클(에피소드 비슷한 단위) 내에서 남은 A/B/C 개수
    private int remainingA;
    private int remainingB;
    private int remainingC;

    // Clean 관련 상태
    private bool cleanActive = false;
    private float cleanEndTime = 0f;
    private int scenariosCompletedSinceLastClean = 0;

    // ✅ (추가) Clean 시작 전 스포너 상태 저장/복원용
    private Dictionary<ProductSpawner, ProductSpawner.SpawnerState> spawnerStateBackup
        = new Dictionary<ProductSpawner, ProductSpawner.SpawnerState>();

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

        // 스포너 자동 수집(인스펙터에서 비워둔 경우)
        if (spawners == null || spawners.Count == 0)
        {
            spawners = new List<ProductSpawner>(FindObjectsOfType<ProductSpawner>());
            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] spawners 자동 탐색: {spawners.Count}개");
            }
        }

        // 첫 사이클 초기화
        ResetCycleCounts();

        // 게임 시작하자마자 첫 시나리오 바로 시작
        StartNewScenario();
    }

    void Update()
    {
        // Clean(D) 진행 중이면 시간만 체크해서 종료 처리
        if (cleanActive)
        {
            if (Time.time >= cleanEndTime)
            {
                EndCleanScenario();
            }
            return;
        }

        if (!scenarioActive)
        {
            // 시나리오 사이 대기 중
            if (Time.time >= nextScenarioStartTime)
            {
                StartNewScenario();
            }
            return;
        }

        // 시나리오 진행 중이면 "모든 고장이 수리됐는지" 확인
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

            // ✅ 일반 시나리오 완료 카운트 증가 (Clean은 제외)
            scenariosCompletedSinceLastClean++;

            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] 시나리오 고장 {currentFaults.Count}개 수리 완료 → {delayBetweenScenarios:F1}s 뒤 다음 시나리오 시작 (sinceClean={scenariosCompletedSinceLastClean})");
            }
        }
    }

    /// <summary>
    /// A/B/C 남은 개수가 모두 0이면 새 사이클로 초기화.
    /// </summary>
    private void ResetCycleCounts()
    {
        remainingA = cycleCountA;
        remainingB = cycleCountB;
        remainingC = cycleCountC;

        if (debugLogs)
        {
            int totalScenarios = remainingA + remainingB + remainingC;
            Debug.Log($"[RandomFailureScenario] 새 사이클 시작: A={remainingA}, B={remainingB}, C={remainingC}, 총 시나리오 수={totalScenarios}");
        }
    }

    /// <summary>
    /// 다음 시나리오에서 몇 개를 고장낼지(A/B/C 중 하나)를 결정한다.
    /// </summary>
    private int DecideFaultCountForNextScenario()
    {
        if (remainingA <= 0 && remainingB <= 0 && remainingC <= 0)
        {
            ResetCycleCounts();
        }

        List<int> candidates = new List<int>();

        if (remainingA > 0) candidates.Add(3);
        if (remainingB > 0) candidates.Add(2);
        if (remainingC > 0) candidates.Add(1);

        if (candidates.Count == 0)
        {
            Debug.LogWarning("[RandomFailureScenario] 남은 시나리오 타입이 없습니다. 기본값 1개 고장 사용.");
            return 1;
        }

        int idx = Random.Range(0, candidates.Count);
        int faultsThisScenario = candidates[idx];

        if (faultsThisScenario == 3) remainingA--;
        else if (faultsThisScenario == 2) remainingB--;
        else if (faultsThisScenario == 1) remainingC--;

        if (debugLogs)
        {
            Debug.Log($"[RandomFailureScenario] 다음 시나리오 타입 선택: faults={faultsThisScenario}, 남은 A/B/C={remainingA}/{remainingB}/{remainingC}");
        }

        return faultsThisScenario;
    }

    void StartNewScenario()
    {
        // ✅ 일정 횟수마다 Clean(D) 실행
        if (cleanEveryNScenarios > 0 && scenariosCompletedSinceLastClean >= cleanEveryNScenarios)
        {
            StartCleanScenario();
            return;
        }

        currentFaults.Clear();

        int faultsPerScenario = DecideFaultCountForNextScenario();

        if (allTunnels.Count < faultsPerScenario)
        {
            Debug.LogWarning("[RandomFailureScenario] faultsPerScenario가 터널 개수보다 많습니다. 가능한 만큼만 사용합니다.");
        }

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

            t.autoRepairFixedDelay = false;

            if (t.IsFault)
            {
                t.ForceRepair();
            }

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
            Debug.Log($"[RandomFailureScenario] 새 시나리오 시작: 고장 터널 {currentFaults.Count}개 (이번 타입 faultsPerScenario={faultsPerScenario})");
        }
    }

    // ==============================
    //  Clean(D)
    // ==============================
    private void StartCleanScenario()
    {
        scenarioActive = false;
        currentFaults.Clear();

        // ✅ (추가) Clean 시작 전 스포너 상태 백업
        BackupSpawnerStates();

        // 스폰 강제 정지
        SetSpawnersForceStop(true);

        cleanActive = true;
        cleanEndTime = Time.time + cleanDuration;

        if (debugLogs)
        {
            Debug.Log($"[RandomFailureScenario] === CLEAN(D) 시작 === duration={cleanDuration:F1}s | 고장 0개 | 스폰 OFF");
        }

        // 카운트 리셋
        scenariosCompletedSinceLastClean = 0;
    }

    private void EndCleanScenario()
    {
        cleanActive = false;

        // 스폰 재개
        SetSpawnersForceStop(false);

        // ✅ (추가) Clean 시작 전 상태로 복원 (HOLD로 남아버리는 문제 방지)
        RestoreSpawnerStates();

        // 다음 정상 시나리오 예약
        nextScenarioStartTime = Time.time + delayBetweenScenarios;

        if (debugLogs)
        {
            Debug.Log($"[RandomFailureScenario] === CLEAN(D) 종료 === → {delayBetweenScenarios:F1}s 뒤 정상 시나리오 재개");
        }
    }

    private void SetSpawnersForceStop(bool stop)
    {
        if (spawners == null) return;

        for (int i = 0; i < spawners.Count; i++)
        {
            var sp = spawners[i];
            if (sp == null) continue;
            sp.ForceStopSpawning(stop);
        }
    }

    // ✅ (추가) 스포너 상태 백업/복원
    private void BackupSpawnerStates()
    {
        spawnerStateBackup.Clear();
        if (spawners == null) return;

        for (int i = 0; i < spawners.Count; i++)
        {
            var sp = spawners[i];
            if (sp == null) continue;

            ProductSpawner.SpawnerState st = ProductSpawner.SpawnerState.RUN;
            if (sp.IsHold) st = ProductSpawner.SpawnerState.HOLD;
            else if (sp.IsHalfHold) st = ProductSpawner.SpawnerState.HALF_HOLD;
            else st = ProductSpawner.SpawnerState.RUN;

            spawnerStateBackup[sp] = st;

            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] (Backup) spawner={sp.name} state={st}");
            }
        }
    }

    private void RestoreSpawnerStates()
    {
        if (spawnerStateBackup == null || spawnerStateBackup.Count == 0) return;

        foreach (var kv in spawnerStateBackup)
        {
            var sp = kv.Key;
            if (sp == null) continue;

            // ForceStop이 false인 상태에서만 의미 있음
            sp.SetState(kv.Value);

            if (debugLogs)
            {
                Debug.Log($"[RandomFailureScenario] (Restore) spawner={sp.name} state={kv.Value}");
            }
        }

        spawnerStateBackup.Clear();
    }
}
