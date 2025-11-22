using System.Collections.Generic;
using UnityEngine;

public class ProductPool : MonoBehaviour
{
    [Header("Pool Setup")]
    public GameObject prefab;  // 상자 프리팹
    public int prewarm = 10;   // 시작 시 미리 만들어둘 개수
    public Transform storage;  // 비활성 보관용 부모(없으면 자동 생성)

    private readonly Queue<GameObject> q = new();

    private void Awake()
    {
        if (!storage)
        {
            var go = new GameObject("ProductPoolStorage");
            storage = go.transform;
            storage.gameObject.SetActive(false); // 씬에서 안 보이게
        }

        for (int i = 0; i < Mathf.Max(0, prewarm); i++)
            q.Enqueue(CreateOne());
    }

    private GameObject CreateOne()
    {
        var go = Instantiate(prefab, storage);
        go.SetActive(false);
        return go;
    }

    public GameObject Get()
    {
        var go = (q.Count > 0) ? q.Dequeue() : CreateOne();
        go.transform.SetParent(null, true);
        go.SetActive(true);
        return go;
    }

    public void Return(GameObject go)
    {
        if (!go) return;
        go.SetActive(false);
        go.transform.SetParent(storage, false);
        q.Enqueue(go);
    }
}
