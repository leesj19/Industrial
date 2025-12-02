import socket
import json
import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb

HOST = "127.0.0.1"
PORT = 50007

# ==============================
#  W&B 설정
# ==============================
ENTITY  = "lsj77205619"          # 필요하면 바꿔도 됨
PROJECT = "IndustryDQN_Factory"  # 새 프로젝트 이름 (원하면 수정)

# 예전 설정 잔재 제거
os.environ.pop("WANDB_ENTITY", None)
os.environ.pop("WANDB_PROJECT", None)
os.environ.pop("WANDB_BASE_URL", None)
os.environ["WANDB_RESUME"] = "never"
os.environ["WANDB_MODE"]   = "online"

run = wandb.init(
    project=PROJECT,
    name=f"IndustryDQN_{uuid.uuid4().hex[:8]}",
    resume="never",
    id=str(uuid.uuid4()),
    mode="online",
)

# 축 정의
wandb.define_metric("env_step")                 # per-step
wandb.define_metric("episode")                  # per-episode
wandb.define_metric("train/*", step_metric="env_step")
wandb.define_metric("episodic/*", step_metric="episode")
wandb.define_metric("buffer/*", step_metric="episode")

# 체크포인트 설정
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_INTERVAL = 5000  # step마다 저장 간격


# ==============================
#  Replay Buffer
# ==============================
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0

    def push(self, s, a, r, s_next, done):
        idx = self.pos
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s_next
        self.dones[idx] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            states=self.states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
            dones=self.dones[idxs],
        )
        return batch

    def __len__(self):
        return self.size


# ==============================
#  Q Network
# ==============================
class QNetwork(nn.Module):
    """
    Q(s, a)를 근사하는 MLP.
    - 입력: state(s): (B, state_dim)
           action(node_id): (B, 1)  -> 정규화된 float 로 넣음
    - 출력: Q(s, a): (B, 1)
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim

        input_dim = state_dim + 1  # state + (정규화된 node_id)
        hidden = 256
        hidden2 = 128

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),  # Q-value 하나
        )

    def forward(self, state, action_id):
        """
        state: (B, state_dim)
        action_id: (B, 1)  (float)
        """
        x = torch.cat([state, action_id], dim=1)
        q = self.net(x)
        return q  # (B, 1)


# ==============================
#  DQN Learner
# ==============================
class DqnLearner:
    def __init__(
        self,
        state_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        capacity: int = 100_000,
        warmup: int = 1_000,
        target_update_interval: int = 1_000,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_update_interval = target_update_interval

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim).to(self.device)
        self.target_net = QNetwork(state_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity, state_dim)

        self.train_step_count = 0
        self.last_loss = None  # 최근 학습 loss 저장

        # 관측된 node_id set (max_a' Q 계산용)
        self.known_actions = set()

        # node_id 정규화를 위한 스케일 (대충 큰 값으로 나눔)
        self.node_id_scale = 100.0

        print(f"[PY] DqnLearner 초기화: state_dim={state_dim}, device={self.device}")

    def _to_tensor(self, arr, dtype=torch.float32):
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def _action_to_tensor(self, actions):
        """
        actions: np.ndarray or list of int, shape (B,)
        -> (B, 1) float tensor, node_id / node_id_scale
        """
        a = np.array(actions, dtype=np.float32) / self.node_id_scale
        t = self._to_tensor(a, dtype=torch.float32).unsqueeze(-1)  # (B, 1)
        return t

    def observe(self, s, a, r, s_next, done=False):
        """
        transition 하나 관찰.
        s, s_next: np.ndarray (state_dim,)
        a: node_id
        r: reward
        done: 에피소드 종료 여부
        return: train이 일어난 경우 loss(float), 아니면 None
        """
        self.replay.push(s, a, r, s_next, done)
        self.known_actions.add(int(a))

        loss_val = None
        if len(self.replay) >= self.warmup:
            loss_val = self._train_step()

        return loss_val

    def _train_step(self):
        batch = self.replay.sample(self.batch_size)

        states = self._to_tensor(batch["states"])           # (B, state_dim)
        actions = batch["actions"]                          # (B,)
        rewards = self._to_tensor(batch["rewards"])         # (B,)
        next_states = self._to_tensor(batch["next_states"]) # (B, state_dim)
        dones = self._to_tensor(batch["dones"])             # (B,)

        actions_t = self._action_to_tensor(actions)         # (B, 1)

        # Q(s, a)
        q_values = self.policy_net(states, actions_t).squeeze(-1)  # (B,)

        # max_a' Q_target(s', a')
        if len(self.known_actions) == 0:
            max_next_q = torch.zeros_like(rewards)
        else:
            # 모든 known_actions에 대해 s'마다 Q(s', a')를 계산 후 max
            action_list = sorted(self.known_actions)
            A = len(action_list)
            B = next_states.shape[0]

            actions_all = np.array(action_list, dtype=np.float32) / self.node_id_scale
            actions_all_t = self._to_tensor(actions_all).view(1, A).repeat(B, 1)  # (B, A)
            # (B, A, 1)
            actions_all_t = actions_all_t.unsqueeze(-1).view(B * A, 1)

            # states 반복
            states_rep = next_states.unsqueeze(1).expand(B, A, self.state_dim)
            states_rep = states_rep.reshape(B * A, self.state_dim)

            with torch.no_grad():
                q_all = self.target_net(states_rep, actions_all_t)  # (B*A, 1)
                q_all = q_all.view(B, A)  # (B, A)
                max_next_q, _ = q_all.max(dim=1)  # (B,)

        # target = r + gamma * (1-done) * max_next_q
        targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.last_loss = float(loss.item())

        if self.train_step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[PY] target_net 업데이트 (train_step={self.train_step_count})")

        # (원래 500 step마다 loss 출력하던 부분은 wandb로 대체 가능)
        return self.last_loss

    def predict_q(self, s, a) -> float:
        """
        단일 (s, a)에 대해 policy_net Q(s,a) 추론.
        """
        s_t = self._to_tensor(s).unsqueeze(0)            # (1, state_dim)
        a_t = self._action_to_tensor([a])                # (1, 1)

        with torch.no_grad():
            q = self.policy_net(s_t, a_t).item()
        return float(q)

    def save(self, step_count: int):
        path = os.path.join(CHECKPOINT_DIR, f"dqn_step{step_count:07d}.pt")
        torch.save(
            {
                "step": step_count,
                "model_state": self.policy_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "known_actions": list(self.known_actions),
            },
            path,
        )
        print(f"[PY] checkpoint saved: {path}")


# ==============================
#  TCP 서버 + 학습 루프
# ==============================
def main():
    step_count = 0
    state_dim = None
    learner = None  # DqnLearner 인스턴스

    # ---- 에피소드 관리용 변수 ----
    episode_idx = 1          # 몇 번째 에피소드인지 (1부터 시작)
    episode_step = 0         # 현재 에피소드 안에서 몇 스텝째인지
    episode_return = 0.0     # 에피소드 누적 reward
    MAX_STEPS_PER_EPISODE = 100

    # 최근 epsilon (Unity에서 msg로 넘어오는 값)
    latest_epsilon = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[PY] Listening on {HOST}:{PORT} ...")

        conn, addr = s.accept()
        print(f"[PY] Connected by {addr}")

        with conn:
            buf = b""
            while True:
                data = conn.recv(4096)
                if not data:
                    print("[PY] Connection closed.")
                    break

                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        msg = json.loads(line.decode("utf-8-sig"))
                    except Exception as e:
                        print(f"[PY] JSON parse error: {e}, line={line[:200]}")
                        continue

                    msg_type = msg.get("type")

                    # ==============================
                    # 1) 액션 요청 처리
                    # ==============================
                    if msg_type == "action_request":
                        state_list = msg.get("state", [])
                        cand_ids = msg.get("candidate_node_ids", [])
                        epsilon = float(msg.get("epsilon", 0.1))
                        latest_epsilon = epsilon  # transition 로그 때 사용

                        if len(state_list) == 0 or len(cand_ids) == 0:
                            print("[PY] action_request: 빈 state 또는 candidate_node_ids")
                            continue

                        s_t = np.array(state_list, dtype=np.float32)

                        if state_dim is None:
                            state_dim = s_t.shape[0]
                            print(f"[PY] 첫 state 수신 (action_request). state_dim={state_dim}")

                        if learner is None:
                            learner = DqnLearner(
                                state_dim=state_dim,
                                gamma=0.99,
                                lr=1e-3,
                                batch_size=64,
                                capacity=100_000,
                                warmup=1_000,
                                target_update_interval=1_000,
                            )

                        cand_ids = [int(x) for x in cand_ids]

                        # 후보들에 대한 Q(s, a) 추정
                        q_values = []
                        for nid in cand_ids:
                            try:
                                q_values.append(learner.predict_q(s_t, nid))
                            except Exception as e:
                                print(f"[PY] predict_q error for node_id={nid}: {e}")
                                q_values.append(0.0)

                        # ε-greedy 액션 선택
                        rand_val = np.random.rand()
                        if rand_val < epsilon:
                            idx = np.random.randint(len(cand_ids))
                            is_random = True
                        else:
                            idx = int(np.argmax(q_values))
                            is_random = False

                        chosen_node_id = int(cand_ids[idx])

                        reply = {
                            "type": "action_reply",
                            "chosen_node_id": chosen_node_id,
                            "candidate_node_ids": cand_ids,
                            "q_values": [float(q) for q in q_values],
                            "epsilon": float(epsilon),
                            "is_random": bool(is_random),
                        }

                        try:
                            conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                            print(
                                f"[PY] action_reply: chosen={chosen_node_id}, "
                                f"eps={epsilon:.3f}, random={is_random}"
                            )
                        except Exception as e:
                            print(f"[PY] action_reply send error: {e}")

                        # 다음 메시지 처리
                        continue

                    # ==============================
                    # 2) transition 처리
                    # ==============================
                    if msg_type == "transition":
                        action_id = msg.get("action_id", -1)
                        node_id = msg.get("node_id", -1)
                        reward = float(msg.get("reward", 0.0))

                        s_t = np.array(msg.get("state_t", []), dtype=np.float32)
                        s_tp1 = np.array(msg.get("state_tp1", []), dtype=np.float32)

                        if state_dim is None:
                            state_dim = s_t.shape[0]
                            print(f"[PY] 첫 state 수신. state_dim={state_dim}")

                        # DQN learner 초기화
                        if learner is None:
                            learner = DqnLearner(
                                state_dim=state_dim,
                                gamma=0.99,
                                lr=1e-3,
                                batch_size=64,
                                capacity=100_000,
                                warmup=1_000,  # 1000 스텝 모이면 학습 시작
                                target_update_interval=1_000,
                            )

                        # ---- 전역 step 카운터 증가 ----
                        step_count += 1

                        # ---- 에피소드 스텝/리턴 업데이트 ----
                        episode_step += 1
                        episode_return += reward

                        # 에피소드 처음 스텝이면 시작 로그
                        if episode_step == 1:
                            print(f"[PY] === Episode {episode_idx} 시작 ===")

                        # ---- DQN에 transition 전달 ----
                        done = False
                        # 100 스텝마다 '가상의 에피소드' 종료 플래그
                        if episode_step >= MAX_STEPS_PER_EPISODE:
                            done = True

                        loss_val = learner.observe(s_t, node_id, reward, s_tp1, done=done)

                        # ---- 현재 (s_t, node_id)에 대한 Q(s,a) 예측해서 Unity로 q_update 전송 ----
                        try:
                            q_est = learner.predict_q(s_t, node_id)
                        except Exception as e:
                            print(f"[PY] predict_q error: {e}")
                            q_est = reward  # fallback

                        q_msg = {
                            "type": "q_update",
                            "node_ids": [int(node_id)],
                            "q_values": [float(q_est)],
                        }

                        try:
                            conn.sendall((json.dumps(q_msg) + "\n").encode("utf-8"))
                            print(
                                f"[PY] step={step_count} | episode={episode_idx} "
                                f"step={episode_step}/{MAX_STEPS_PER_EPISODE} : "
                                f"action_id={action_id}, node_id={node_id}, "
                                f"reward={reward:+.3f}"
                            )
                        except Exception as e:
                            print(f"[PY] q_update send error: {e}")

                        # 앞 몇 개 상태값만 샘플로 출력 (초기 몇 step)
                        if step_count <= 3:
                            head = 12
                            print(f"      s_t[0:{head}]   = {s_t[:head]}")
                            print(f"      s_tp1[0:{head}] = {s_tp1[:head]}")

                        # ---- W&B per-step 로깅 ----
                        wandb.log(
                            {
                                "env_step": step_count,
                                "train/reward": reward,
                                "train/q_est": float(q_est),
                                "train/loss": float(loss_val) if loss_val is not None else 0.0,
                                "train/done_flag": float(done),
                                "train/epsilon": float(latest_epsilon) if latest_epsilon is not None else 0.0,
                                "buffer/size": len(learner.replay),
                            }
                        )

                        # ---- 에피소드 종료 처리 ----
                        if done:
                            avg_reward = episode_return / float(max(1, episode_step))
                            print(
                                f"[PY] === Episode {episode_idx} done === "
                                f"(episode_step={episode_step}, return={episode_return:+.3f}, "
                                f"avg_reward={avg_reward:+.3f})"
                            )

                            # W&B per-episode 로깅
                            wandb.log(
                                {
                                    "episode": episode_idx,
                                    "episodic/return": float(episode_return),
                                    "episodic/avg_reward": float(avg_reward),
                                    "episodic/length": episode_step,
                                    "buffer/size": len(learner.replay),
                                }
                            )

                            episode_idx += 1
                            episode_step = 0
                            episode_return = 0.0

                        # ---- 체크포인트 저장 ----
                        if step_count % CHECKPOINT_INTERVAL == 0:
                            learner.save(step_count)

                        continue

                    # ==============================
                    # 3) 그 외 타입
                    # ==============================
                    print(f"[PY] Unknown msg type: {msg_type}")


if __name__ == "__main__":
    main()
