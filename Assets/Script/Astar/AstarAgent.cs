using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class AStarAgent : MonoBehaviour
{
    public GridManager grid;
    public Transform target;
    public float moveSpeed = 3f;
    public float stoppingDistance = 0.1f;

    Rigidbody rb;

    List<Vector3> path;
    int pathIndex = 0;
    bool hasPath = false;

    // 이동 기반은 FixedUpdate로
    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.interpolation = RigidbodyInterpolation.Interpolate;
    }

    void Start()
    {
        RecalculatePath();
    }

    void Update()
    {
        // 타깃 바뀌면 path 갱신
        if (Input.GetKeyDown(KeyCode.Space))
            RecalculatePath();
    }

    void FixedUpdate()
    {
        if (hasPath)
            FollowPath();
    }

    public void RecalculatePath()
    {
        if (grid == null || target == null)
        {
            Debug.LogWarning("Grid or Target not assigned.");
            return;
        }

        path = grid.GetPath(transform.position, target.position);
        pathIndex = 0;
        hasPath = (path != null && path.Count > 0);
    }

    void FollowPath()
    {
        if (!hasPath || pathIndex >= path.Count)
            return;

        Vector3 targetPos = path[pathIndex];
        targetPos.y = transform.position.y;

        Vector3 dir = targetPos - transform.position;
        float dist = dir.magnitude;

        if (dist < stoppingDistance)
        {
            pathIndex++;
            return;
        }

        dir.Normalize();

        // Rigidbody 이동
        Vector3 newPos =
            transform.position + dir * moveSpeed * Time.fixedDeltaTime;
        rb.MovePosition(newPos);

        // 부드러운 회전
        if (dir != Vector3.zero)
        {
            Quaternion targetRot = Quaternion.LookRotation(dir);
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation, targetRot, 180f * Time.fixedDeltaTime);
        }
    }
}
