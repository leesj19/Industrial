using UnityEngine;

[DisallowMultipleComponent]
public class ProductSpawner : MonoBehaviour
{
    // ====== Graph Node ======
    [Header("Graph Node")]
    [Tooltip("그래프에서 이 Spawner의 노드 ID (0부터 시작 추천)")]
    public int nodeId = -1;
    public int NodeId => nodeId;

    [Tooltip("이 Spawner에서 바로 이어지는 첫 터널들 (갈림길 여러 개면 전부 넣기)")]
    public TunnelController[] firstTunnels;

    // ====== State ======
    // RUN       : 정상 스폰
    // HALF_HOLD : 절반 속도 스폰 (Upstream HALF_HOLD에 대응)
    // HOLD      : 스폰 완전 정지
    public enum SpawnerState { RUN, HALF_HOLD, HOLD }

    [Header("State")]
    [SerializeField] private SpawnerState state = SpawnerState.RUN;
    public bool IsRun      => state == SpawnerState.RUN;
    public bool IsHalfHold => state == SpawnerState.HALF_HOLD;
    public bool IsHold     => state == SpawnerState.HOLD;

    // ✅ (추가) Clean 등 외부 이유로 스폰을 "강제 정지"할 때 사용
    [Header("Forced Stop (External Override)")]
    [Tooltip("true이면 업스트림 신호와 무관하게 스폰을 완전히 멈춤 (Clean 시나리오용)")]
    [SerializeField] private bool forceStopSpawning = false;
    public bool ForceStopSpawningFlag => forceStopSpawning;

    // ====== Visual ======
    [Header("Status Visual")]
    [Tooltip("상태 색을 바꿀 Renderer (예: 상단 판 MeshRenderer)")]
    [SerializeField] private Renderer statusRenderer;
    [SerializeField] private string colorProperty = "_Color";

    [SerializeField] private Color runColor       = new Color(0.2f, 0.8f, 0.3f, 1f);   // 초록 (정상)
    [SerializeField] private Color halfHoldColor  = new Color(1f, 0.85f, 0.1f, 1f);    // 노랑 (부분 정지)
    [SerializeField] private Color holdColor      = new Color(1f, 0.65f, 0.1f, 1f);    // 주황 (완전 HOLD)

    private Material instMat;

    // ====== Links ======
    [Header("Links")]
    public ProductPool pool;
    public TargetPath path;
    public Transform spawnPoint;   // 없으면 path 첫 포인트 사용

    // ====== Product Settings ======
    [Header("Product Settings")]
    [Tooltip("스폰된 product가 살아 있을 최대 시간(초). 0 이하면 무제한.")]
    public float productLifetimeSeconds = 300f;  // 기본 5분

    // ====== Spawn Policy ======
    [Header("Spawn Policy")]
    [Tooltip("시작하자마자 자동 스폰할지 여부")]
    public bool autoStart = true;

    [Tooltip("RUN 상태일 때 기준 스폰 간격(초)")]
    public float spawnInterval = 1.5f;

    private float t;

    private void Awake()
    {
        if (statusRenderer != null)
            instMat = statusRenderer.material;
        ApplyStatusVisual();
    }

    private void Update()
    {
        // ✅ 강제 스폰 정지(Clean 등)
        if (forceStopSpawning)
            return;

        // 자동 스폰: HOLD가 아닌 상태에서만 시도
        if (!autoStart || pool == null || path == null || IsHold)
            return;

        // HALF_HOLD 상태에서는 스폰 간격을 2배로 늘려서 50%만 발송
        float effInterval = spawnInterval;
        if (IsHalfHold)
            effInterval *= 2f;

        t += Time.deltaTime;
        if (t >= effInterval)
        {
            t = 0f;
            SpawnOne();
        }
    }

    /// <summary>
    /// 수동 스폰 호출.
    /// </summary>
    public void SpawnOne()
    {
        // ✅ 강제 스폰 정지(Clean 등)
        if (forceStopSpawning)
            return;

        // 완전 HOLD면 스폰 금지
        if (IsHold || pool == null || path == null)
            return;

        var go = pool.Get();
        if (go == null)
            return;

        Vector3 pos = spawnPoint ? spawnPoint.position : path.GetPoint(0).position;
        go.transform.SetPositionAndRotation(pos, Quaternion.identity);

        var follower = go.GetComponent<PathFollower>();
        if (!follower) follower = go.AddComponent<PathFollower>();
        follower.SetPath(path);

        var ret = go.GetComponent<ReturnToPoolOnFinish>();
        if (!ret) ret = go.AddComponent<ReturnToPoolOnFinish>();
        ret.pool = pool;

        if (productLifetimeSeconds > 0f)
            ret.lifetimeSeconds = productLifetimeSeconds;
    }

    // ✅ (추가) 외부에서 스폰 강제 OFF/ON
    public void ForceStopSpawning(bool stop)
    {
        forceStopSpawning = stop;
        if (stop)
        {
            // Clean 들어갈 때는 안전하게 HOLD로 잠가두고 타이머도 리셋
            state = SpawnerState.HOLD;
            t = 0f;
        }
        ApplyStatusVisual();
    }

    // ====== External control (from upstream tunnel) ======
    public void SetState(SpawnerState newState)
    {
        // ✅ Clean 등으로 강제 정지 중이면 업스트림 상태 변경 무시
        if (forceStopSpawning) return;

        if (state == newState) return;
        state = newState;
        ApplyStatusVisual();
    }

    public void EnterHold()
    {
        // forceStopSpawning 중이어도 HOLD는 문제없으니 허용
        if (state == SpawnerState.HOLD) return;
        state = SpawnerState.HOLD;
        ApplyStatusVisual();
    }

    public void EnterRun()
    {
        // ✅ Clean 등으로 강제 정지 중이면 RUN 복귀 금지
        if (forceStopSpawning) return;

        if (state == SpawnerState.RUN) return;
        state = SpawnerState.RUN;
        ApplyStatusVisual();
    }

    public void EnterHalfHold()
    {
        // ✅ Clean 등으로 강제 정지 중이면 HALF_HOLD 복귀 금지
        if (forceStopSpawning) return;

        if (state == SpawnerState.HALF_HOLD) return;
        state = SpawnerState.HALF_HOLD;
        ApplyStatusVisual();
    }

    public void OnUpstreamHold()      => EnterHold();
    public void OnUpstreamResume()    => EnterRun();
    public void OnUpstreamHalfHold()  => EnterHalfHold();

    // ====== Visual ======
    private void ApplyStatusVisual()
    {
        if (instMat == null) return;

        var prop = string.IsNullOrEmpty(colorProperty) ? "_Color" : colorProperty;
        Color c = runColor;

        // ✅ 강제 정지 중이면 시각적으로 HOLD로 고정
        if (forceStopSpawning)
        {
            c = holdColor;
        }
        else
        {
            switch (state)
            {
                case SpawnerState.RUN:       c = runColor;      break;
                case SpawnerState.HALF_HOLD: c = halfHoldColor; break;
                case SpawnerState.HOLD:      c = holdColor;     break;
            }
        }

        if (instMat.HasProperty(prop))
            instMat.SetColor(prop, c);
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        Color c = runColor;

        if (forceStopSpawning)
        {
            c = holdColor;
        }
        else
        {
            switch (state)
            {
                case SpawnerState.RUN:       c = runColor;      break;
                case SpawnerState.HALF_HOLD: c = halfHoldColor; break;
                case SpawnerState.HOLD:      c = holdColor;     break;
            }
        }

        Gizmos.color = c;
        Gizmos.DrawWireCube(
            transform.position + Vector3.up * 0.1f,
            new Vector3(0.18f, 0.02f, 0.18f)
        );
    }
#endif
}
