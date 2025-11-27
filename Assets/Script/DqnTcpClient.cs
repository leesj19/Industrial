using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

/// <summary>
/// Python DQN.py와 TCP로 통신하는 클라이언트.
/// - JSON 한 줄(line) 단위 송수신
/// - transition 전송: {"type":"transition", ...}
/// - q_update 수신:   {"type":"q_update", "node_ids":[...], "q_values":[...]}
/// </summary>
public class DqnTcpClient : MonoBehaviour
{
    [Header("TCP 설정")]
    public string host = "127.0.0.1";
    public int port = 50007;
    public bool autoConnectOnStart = true;

    [Header("Debug 옵션")]
    public bool debugLogs = true;

    TcpClient client;
    NetworkStream stream;
    Thread recvThread;
    volatile bool running = false;
    readonly object sendLock = new object();

    struct QUpdateItem
    {
        public int nodeId;
        public float qValue;
    }

    readonly Queue<QUpdateItem> qUpdateQueue = new Queue<QUpdateItem>();
    readonly object qLock = new object();

    /// <summary>
    /// 메인 스레드에서 호출되는 콜백 (node_id, q_value)
    /// </summary>
    public Action<int, float> OnQUpdate;

    // === 수신 메시지용 DTO ===
    [Serializable]
    class QUpdateMsg
    {
        public string type;
        public int[] node_ids;
        public float[] q_values;
    }

    void Start()
    {
        if (autoConnectOnStart)
            Connect();
    }

    void Update()
    {
        // 수신 스레드에서 쌓아둔 q_update를 메인 스레드에서 처리
        if (OnQUpdate == null) return;

        while (true)
        {
            QUpdateItem item;
            lock (qLock)
            {
                if (qUpdateQueue.Count == 0)
                    break;
                item = qUpdateQueue.Dequeue();
            }

            try
            {
                OnQUpdate?.Invoke(item.nodeId, item.qValue);
            }
            catch (Exception e)
            {
                Debug.LogError($"[DqnTcpClient] OnQUpdate callback error: {e}");
            }
        }
    }

    void OnDestroy()
    {
        Close();
    }

    public void Connect()
    {
        if (client != null)
            return;

        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            stream = client.GetStream();

            running = true;
            recvThread = new Thread(RecvLoop);
            recvThread.IsBackground = true;
            recvThread.Start();

            if (debugLogs)
                Debug.Log($"[DqnTcpClient] Connected to {host}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[DqnTcpClient] Connect error: {e}");
            Close();
        }
    }

    public void Close()
    {
        running = false;

        try { stream?.Close(); } catch { }
        try { client?.Close(); } catch { }

        stream = null;
        client = null;
    }

    /// <summary>
    /// JSON 문자열을 한 줄(line)로 전송
    /// </summary>
    public void SendJsonLine(string json)
    {
        if (client == null || stream == null || !client.Connected)
            return;

        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(json + "\n");
            lock (sendLock)
            {
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush();
            }

            if (debugLogs)
                Debug.Log($"[DqnTcpClient] Sent: {json}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[DqnTcpClient] Send error: {e}");
            Close();
        }
    }

    void RecvLoop()
    {
        byte[] buffer = new byte[4096];
        StringBuilder sb = new StringBuilder();

        try
        {
            while (running && client != null && client.Connected)
            {
                int n = stream.Read(buffer, 0, buffer.Length);
                if (n <= 0)
                {
                    if (debugLogs)
                        Debug.Log("[DqnTcpClient] Connection closed by server.");
                    break;
                }

                string chunk = Encoding.UTF8.GetString(buffer, 0, n);
                sb.Append(chunk);

                while (true)
                {
                    string current = sb.ToString();
                    int idx = current.IndexOf('\n');
                    if (idx < 0)
                        break;

                    string line = current.Substring(0, idx).Trim();
                    sb.Remove(0, idx + 1);

                    if (!string.IsNullOrEmpty(line))
                        HandleLine(line);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[DqnTcpClient] RecvLoop error: {e}");
        }

        running = false;
    }

    void HandleLine(string line)
    {
        try
        {
            var msg = JsonUtility.FromJson<QUpdateMsg>(line);
            if (msg != null && msg.type == "q_update" &&
                msg.node_ids != null && msg.q_values != null)
            {
                int count = Mathf.Min(msg.node_ids.Length, msg.q_values.Length);
                for (int i = 0; i < count; i++)
                {
                    lock (qLock)
                    {
                        qUpdateQueue.Enqueue(new QUpdateItem
                        {
                            nodeId = msg.node_ids[i],
                            qValue = msg.q_values[i]
                        });
                    }
                }

                if (debugLogs)
                {
                    for (int i = 0; i < count; i++)
                    {
                        Debug.Log($"[DqnTcpClient] q_update: node_id={msg.node_ids[i]}, Q={msg.q_values[i]:+0.000}");
                    }
                }
            }
            else
            {
                if (debugLogs)
                    Debug.Log($"[DqnTcpClient] recv line (unknown): {line}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[DqnTcpClient] JSON parse error: {e}, line={line}");
        }
    }
}
