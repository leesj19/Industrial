using UnityEngine;

public class RobotDecisionController : MonoBehaviour
{
    public AStarAgent agent;
    public float decisionInterval = 1.0f;
    private float nextDecisionTime = 0f;

    private FactoryEnvManager env;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<AStarAgent>();

        env = FactoryEnvManager.Instance;
        if (env == null)
        {
            env = FindObjectOfType<FactoryEnvManager>();
            if (env == null)
            {
                Debug.LogError("[RobotDecisionController] FactoryEnvManager를 찾지 못했습니다.");
            }
        }
    }

    void Update()
    {
        if (agent == null || env == null) return;

        if (Time.time < nextDecisionTime)
            return;

        nextDecisionTime = Time.time + decisionInterval;

        // 통합된 EnvManager에서 가장 우선순위 높은 고장 터널 찾기
        var best = env.GetBestFaultyTunnel();
        if (best == null) return;

        // 터널 transform 쪽으로 이동 (원하면 나중에 별도 Target Transform 추가 가능)
        agent.SetTarget(best.transform, true);
    }
}
