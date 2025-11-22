using UnityEngine;

[RequireComponent(typeof(PathFollower))]
public class ReturnToPoolOnFinish : MonoBehaviour
{
    public ProductPool pool;
    private PathFollower follower;

    private void Awake()
    {
        follower = GetComponent<PathFollower>();
        follower.OnFinished += HandleFinish;
    }

    private void HandleFinish()
    {
        // 필요 시 여기서 상태 초기화 로직 추가
        if (pool) pool.Return(gameObject);
        else gameObject.SetActive(false); // 안전장치
    }
}
