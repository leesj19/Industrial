using UnityEngine;

/// <summary>
/// Product가 경로(PathFollower)를 따라가다가
/// - 경로 끝(PathFollower.OnFinished)에 도달했을 때
/// - 혹은 lifetimeSeconds 이상 시간이 지났을 때
/// 풀(ProductPool)로 되돌리는 스크립트.
/// </summary>
[DisallowMultipleComponent]
public class ReturnToPoolOnFinish : MonoBehaviour
{
    [Header("Pool 설정")]
    [Tooltip("이 Product를 되돌려 보낼 풀")]
    public ProductPool pool;      // ProductSpawner에서 할당

    [Header("수명 설정")]
    [Tooltip("생성 후 이 시간이 지나면 자동 회수 (초). 0 이하이면 수명 제한 없음")]
    public float lifetimeSeconds = 0f;   // 기본값: 무한 수명

    // 내부 상태
    private float elapsed = 0f;
    private bool  isReturned = false;

    // Path 끝 이벤트를 받기 위한 참조
    private PathFollower follower;

    private void Awake()
    {
        // 같은 오브젝트에 붙어 있는 PathFollower를 찾는다.
        follower = GetComponent<PathFollower>();
    }

    private void OnEnable()
    {
        elapsed    = 0f;
        isReturned = false;

        // 경로 끝(OnFinished) 이벤트 구독
        if (follower != null)
        {
            follower.OnFinished -= HandlePathFinished;  // 중복 구독 방지
            follower.OnFinished += HandlePathFinished;
        }
    }

    private void OnDisable()
    {
        // 이벤트 해제 (메모리 누수/예상치 못한 콜백 방지)
        if (follower != null)
        {
            follower.OnFinished -= HandlePathFinished;
        }
    }

    private void Update()
    {
        if (isReturned) return;

        // 수명 제한이 있을 때만 타이머 체크
        if (lifetimeSeconds > 0f)
        {
            elapsed += Time.deltaTime;
            if (elapsed >= lifetimeSeconds)
            {
                ReturnToPool("lifetime expired");
            }
        }
    }

    /// <summary>
    /// PathFollower가 경로 끝까지 도달했을 때 호출되는 콜백
    /// </summary>
    private void HandlePathFinished()
    {
        // 경로 끝 도착 → 회수
        ForceReturn("path finished");
    }

    /// <summary>
    /// 외부에서 강제로 회수하고 싶을 때 호출할 수 있는 함수
    /// </summary>
    public void ForceReturn(string reason = "forced")
    {
        ReturnToPool(reason);
    }

    private void ReturnToPool(string reason)
    {
        if (isReturned) return;
        isReturned = true;

        if (pool != null)
        {
            pool.Return(gameObject);   // 풀로 회수 (비활성 + 보관)
        }
        else
        {
            // 풀이 없으면 안전하게 삭제
            Destroy(gameObject);
        }
    }
}
