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
CHECKPOINT_INTERVAL = 500  # step마다 저장 간격


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

        input_dim = state_dim + 1
        hidden = 256
        hidden2 = 128

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, state, action_id):
        x = torch.cat([state, action_id], dim=1)
        q = self.net(x)
        return q


# ==============================
#  DQN Learner (Double DQN + Soft Target Update)
# ==============================
class DqnLearner:
    def __init__(
        self,
        state_dim: int,
        gamma: float = 0.99,
        lr: float = 3e-4,
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
        self.last_loss = None

        self.known_actions = set()
        self.node_id_scale = 100.0

        self.tau = 0.005

        print(f"[PY] DqnLearner 초기화: state_dim={state_dim}, device={self.device}")

    def _to_tensor(self, arr, dtype=torch.float32):
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def _action_to_tensor(self, actions):
        a = np.array(actions, dtype=np.float32) / self.node_id_scale
        t = self._to_tensor(a, dtype=torch.float32).unsqueeze(-1)
        return t

    def observe(self, s, a, r, s_next, done=False):
        self.replay.push(s, a, r, s_next, done)
        self.known_actions.add(int(a))

        loss_val = None
        if len(self.replay) >= self.warmup:
            loss_val = self._train_step()
        return loss_val

    def _train_step(self):
        batch = self.replay.sample(self.batch_size)

        states = self._to_tensor(batch["states"])
        actions = batch["actions"]
        rewards = self._to_tensor(batch["rewards"])
        next_states = self._to_tensor(batch["next_states"])
        dones = self._to_tensor(batch["dones"])

        actions_t = self._action_to_tensor(actions)

        q_values = self.policy_net(states, actions_t).squeeze(-1)

        if len(self.known_actions) == 0:
            max_next_q = torch.zeros_like(rewards)
        else:
            action_list = sorted(self.known_actions)
            A = len(action_list)
            B = next_states.shape[0]

            actions_all = np.array(action_list, dtype=np.float32) / self.node_id_scale
            actions_all_t = self._to_tensor(actions_all)
            actions_all_t = actions_all_t.view(1, A).repeat(B, 1)
            actions_all_t = actions_all_t.unsqueeze(-1).view(B * A, 1)

            states_rep = next_states.unsqueeze(1).expand(B, A, self.state_dim)
            states_rep = states_rep.reshape(B * A, self.state_dim)

            with torch.no_grad():
                q_all_policy = self.policy_net(states_rep, actions_all_t).view(B, A)
                best_idx = q_all_policy.argmax(dim=1)

                q_all_target = self.target_net(states_rep, actions_all_t).view(B, A)
                max_next_q = q_all_target.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.last_loss = float(loss.item())

        with torch.no_grad():
            tau = self.tau
            for target_param, param in zip(self.target_net.parameters(),
                                           self.policy_net.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)

        return self.last_loss

    def predict_q(self, s, a) -> float:
        s_t = self._to_tensor(s).unsqueeze(0)
        a_t = self._action_to_tensor([a])
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
#  epsilon schedule (episode-based)
# ==============================
def epsilon_by_episode(
    episode_idx: int,
    eps_start: float = 1.0,
    eps_min: float = 0.1,
    decay_episodes: int = 1000,
) -> float:
    """
    episode 1 -> eps_start
    episode decay_episodes -> eps_min (선형 감소)
    episode >= decay_episodes -> eps_min 유지
    """
    if episode_idx <= 1:
        return eps_start
    if episode_idx >= decay_episodes:
        return eps_min

    # 1..1000 구간에서 선형감소
    t = (episode_idx - 1) / float(decay_episodes - 1)  # 0..1
    eps = eps_start + t * (eps_min - eps_start)
    return max(eps_min, float(eps))


# ==============================
#  TCP 서버 + 학습 루프
# ==============================
def main():
    step_count = 0
    state_dim = None
    learner = None

    # ---- 에피소드 관리용 변수 ----
    episode_idx = 1
    episode_step = 0
    episode_return = 0.0

    # ✅ 에피소드 기반 추가 통계
    episode_q_sum = 0.0
    episode_q_count = 0
    episode_loss_sum = 0.0
    episode_loss_count = 0
    episode_random_count = 0

    # 🔹 한 에피소드 = 30 step
    MAX_STEPS_PER_EPISODE = 30

    # 로깅용: Unity가 보내온 epsilon (참고)
    latest_base_epsilon_unity = None
    # 로깅용: 실제 사용 epsilon
    latest_epsilon_used = None

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

                        # Unity에서 넘어온 epsilon (참고용)
                        base_epsilon_unity = float(msg.get("epsilon", 0.1))
                        latest_base_epsilon_unity = base_epsilon_unity

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
                                lr=3e-4,
                                batch_size=64,
                                capacity=100_000,
                                warmup=1_000,
                                target_update_interval=1_000,
                            )

                        cand_ids = [int(x) for x in cand_ids]

                        # 후보들 Q(s,a)
                        q_values = []
                        for nid in cand_ids:
                            try:
                                q_values.append(learner.predict_q(s_t, nid))
                            except Exception as e:
                                print(f"[PY] predict_q error for node_id={nid}: {e}")
                                q_values.append(0.0)

                        # ✅ episode 기반 epsilon 사용
                        epsilon = epsilon_by_episode(
                            episode_idx=episode_idx,
                            eps_start=1.0,
                            eps_min=0.1,
                            decay_episodes=2000,
                        )
                        latest_epsilon_used = epsilon

                        # ε-greedy
                        rand_val = np.random.rand()
                        if rand_val < epsilon:
                            idx = np.random.randint(len(cand_ids))
                            is_random = True
                        else:
                            idx = int(np.argmax(q_values))
                            is_random = False

                        if is_random:
                            episode_random_count += 1

                        chosen_node_id = int(cand_ids[idx])

                        reply = {
                            "type": "action_reply",
                            "chosen_node_id": chosen_node_id,
                            "candidate_node_ids": cand_ids,
                            "q_values": [float(q) for q in q_values],
                            "epsilon": float(epsilon),         # Unity로도 전달 (표시/디버그용)
                            "is_random": bool(is_random),
                        }

                        try:
                            conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                            if episode_step == 0:
                                print(f"[PY] === Episode {episode_idx} 시작 === (epsilon={epsilon:.3f})")
                            print(f"[PY] action_reply: chosen={chosen_node_id}, eps={epsilon:.3f}, random={is_random}")
                        except Exception as e:
                            print(f"[PY] action_reply send error: {e}")

                        # per-step로도 epsilon 로깅 (선택된 epsilon 확인용)
                        wandb.log(
                            {
                                "env_step": step_count,
                                "train/epsilon_used": float(latest_epsilon_used),
                                "train/base_epsilon_unity": float(latest_base_epsilon_unity),
                            }
                        )

                        continue

                    # ==============================
                    # 2) transition 처리
                    # ==============================
                    if msg_type == "transition":
                        action_id = msg.get("action_id", -1)
                        node_id = int(msg.get("node_id", -1))
                        reward = float(msg.get("reward", 0.0))

                        s_t = np.array(msg.get("state_t", []), dtype=np.float32)
                        s_tp1 = np.array(msg.get("state_tp1", []), dtype=np.float32)

                        if state_dim is None:
                            state_dim = s_t.shape[0]
                            print(f"[PY] 첫 state 수신. state_dim={state_dim}")

                        if learner is None:
                            learner = DqnLearner(
                                state_dim=state_dim,
                                gamma=0.99,
                                lr=3e-4,
                                batch_size=64,
                                capacity=100_000,
                                warmup=500,
                                target_update_interval=1_000,
                            )

                        # ---- 전역 step ----
                        step_count += 1

                        # ---- episode step / return ----
                        episode_step += 1
                        episode_return += reward

                        done = (episode_step >= MAX_STEPS_PER_EPISODE)

                        loss_val = learner.observe(s_t, node_id, reward, s_tp1, done=done)

                        # q_est
                        try:
                            q_est = learner.predict_q(s_t, node_id)
                        except Exception as e:
                            print(f"[PY] predict_q error: {e}")
                            q_est = reward

                        # episode 통계 집계
                        episode_q_sum += float(q_est)
                        episode_q_count += 1

                        if loss_val is not None:
                            episode_loss_sum += float(loss_val)
                            episode_loss_count += 1

                        # Unity로 q_update
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
                                f"action_id={action_id}, node_id={node_id}, reward={reward:+.3f}"
                            )
                        except Exception as e:
                            print(f"[PY] q_update send error: {e}")

                        if step_count <= 3:
                            head = 12
                            print(f"      s_t[0:{head}]   = {s_t[:head]}")
                            print(f"      s_tp1[0:{head}] = {s_tp1[:head]}")

                        # per-step W&B
                        wandb.log(
                            {
                                "env_step": step_count,
                                "train/reward": reward,
                                "train/q_est": float(q_est),
                                "train/loss": float(loss_val) if loss_val is not None else 0.0,
                                "train/done_flag": float(done),
                                "train/epsilon_used": float(latest_epsilon_used) if latest_epsilon_used is not None else 0.0,
                                "train/base_epsilon_unity": float(latest_base_epsilon_unity) if latest_base_epsilon_unity is not None else 0.0,
                                "buffer/size": len(learner.replay),
                            }
                        )

                        # ---- episode 종료 처리 ----
                        if done:
                            avg_reward = episode_return / float(max(1, episode_step))
                            avg_q = episode_q_sum / float(max(1, episode_q_count))
                            avg_loss = (episode_loss_sum / float(episode_loss_count)) if episode_loss_count > 0 else 0.0
                            random_rate = episode_random_count / float(MAX_STEPS_PER_EPISODE)

                            print(
                                f"[PY] === Episode {episode_idx} done === "
                                f"(len={episode_step}, return={episode_return:+.3f}, avg_reward={avg_reward:+.3f}, "
                                f"avg_q={avg_q:+.3f}, avg_loss={avg_loss:.6f}, random_rate={random_rate:.2f})"
                            )

                            # per-episode W&B
                            wandb.log(
                                {
                                    "episode": episode_idx,
                                    "episodic/return": float(episode_return),
                                    "episodic/avg_reward": float(avg_reward),
                                    "episodic/length": int(episode_step),
                                    "episodic/avg_q_est": float(avg_q),
                                    "episodic/avg_loss": float(avg_loss),
                                    "episodic/random_rate": float(random_rate),
                                    "episodic/epsilon_used": float(latest_epsilon_used) if latest_epsilon_used is not None else 0.0,
                                    "buffer/size": len(learner.replay),
                                }
                            )

                            episode_idx += 1
                            episode_step = 0
                            episode_return = 0.0

                            # episode 통계 리셋
                            episode_q_sum = 0.0
                            episode_q_count = 0
                            episode_loss_sum = 0.0
                            episode_loss_count = 0
                            episode_random_count = 0

                        # ---- 체크포인트 저장 ----
                        if step_count % CHECKPOINT_INTERVAL == 0:
                            learner.save(step_count)

                        continue

                    print(f"[PY] Unknown msg type: {msg_type}")


if __name__ == "__main__":
    main()
