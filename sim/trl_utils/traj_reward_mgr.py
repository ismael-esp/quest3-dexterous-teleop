# utils/traj_reward_mgr.py

import torch
import numpy as np
import os
import pickle

class TrajectoryRewardManager:
    def __init__(
        self,
        device,
        num_envs,
        max_len,
        task_name,
        base_dir,
        num_cubes,
        expected_action_dim=None,
        traj_step_scale=0.0,
    ):
        self.device = device
        self.num_envs = num_envs
        self.max_len = max_len
        self.task_name = task_name
        self.base_dir = base_dir
        self.num_cubes = num_cubes
        self.expected_action_dim = int(expected_action_dim) if expected_action_dim is not None else None
        self.traj_step_scale = float(traj_step_scale)

        # =========================
        # Traj Parameters
        # =========================
        self.traj_w = 1.0
        self.traj_allow_recovery = True
        self.traj_sr_ema = 0.0
        self.traj_sr_beta = 0.05
        
        self.traj_stage = 0
        self.traj_max_stage_reached = 0
        self.traj_stage_to_w = [1.0, 1.0, 1.0, 1.0] # [1.0, 0.7, 0.3, 0.0]
        
        self.traj_thr_down = [0.60, 0.75, 0.85]
        self.traj_thr_up = [0.45, 0.65, 0.80]
        self.traj_recovery_cap = 1.0
        self.traj_alpha_down = 0.05
        self.traj_alpha_up = 0.05
        self.traj_sigma = 0.25
        
        self.traj_sched_n = 0

        # =========================
        # Demo Data
        # =========================
        self.demo_loaded = False
        self.demo_raw_actions = None
        self.demo_init_pos = None
        self.demo_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._warned_action_mismatch = False
        
        # A zero trajectory step-scale means the shaping term is disabled.
        # In that case skip demo I/O and all matching work on the rollout path.
        if self.traj_step_scale > 1e-12:
            self._load_demo()

    def _traj_enabled(self):
        return self.demo_loaded and (self.traj_w >= 1e-6) and (self.traj_step_scale > 1e-12)

    def _load_demo(self):
        demo_path = os.path.join(self.base_dir, "demos", f"Demo_Traj_{self.task_name}.pickle")
        
        if not os.path.isfile(demo_path):
            print(f"[TrajMgr] Demo not found: {demo_path}")
            return

        with open(demo_path, "rb") as f:
            data = pickle.load(f)
        
        demo_data = data[0]
        if "raw_actions" not in demo_data:
            print("[TrajMgr] Key 'raw_actions' not found.")
            return

        # (T, M, 2, 7)
        raw_np = np.stack(demo_data["raw_actions"], axis=0)
        self.demo_raw_actions = torch.tensor(raw_np, device=self.device, dtype=torch.float32)

        if "init_cube_pos" in demo_data:
            pos_np = demo_data["init_cube_pos"]
            self.demo_init_pos = torch.tensor(pos_np, device=self.device, dtype=torch.float32)
        else:
            print("[TrajMgr] 'init_cube_pos' missing. Using zeros.")
            self.demo_init_pos = torch.zeros((self.demo_raw_actions.shape[1], self.num_cubes, 3), device=self.device)
        
        self.demo_K = self.demo_raw_actions.shape[0]
        self.demo_D = self.demo_raw_actions.shape[1]
        demo_action_dim = int(self.demo_raw_actions.shape[-1]) if self.demo_raw_actions.ndim >= 4 else None

        if (self.expected_action_dim is not None) and (demo_action_dim is not None):
            if int(demo_action_dim) != int(self.expected_action_dim):
                print(
                    f"[TrajMgr] Action dim mismatch: demo={demo_action_dim}, expected={self.expected_action_dim}. "
                    f"Disable trajectory reward until matching demo is recollected."
                )
                self.demo_loaded = False
                self.demo_raw_actions = None
                self.demo_init_pos = None
                return

        self.demo_loaded = True
        print(f"[TrajMgr] Loaded: {demo_path} | Shape: {tuple(self.demo_raw_actions.shape)}")

    def update_schedule(self, ep_success: torch.Tensor):
        """성공률 기반 가중치(w) 업데이트"""
        if self.traj_step_scale <= 1e-12:
            return
        if ep_success.numel() == 0:
            return

        batch_sr = float(ep_success.float().mean().item())
        
        # EMA
        self.traj_sr_ema = (1.0 - self.traj_sr_beta) * self.traj_sr_ema + self.traj_sr_beta * batch_sr
        sr = self.traj_sr_ema
        
        stage = self.traj_stage
        
        # Down
        if stage == 0 and sr >= self.traj_thr_down[0]: stage = 1
        elif stage == 1 and sr >= self.traj_thr_down[1]: stage = 2
        elif stage == 2 and sr >= self.traj_thr_down[2]: stage = 3
        
        # Up
        if self.traj_allow_recovery:
            if stage == 1 and sr <= self.traj_thr_up[0]: stage = 0
            elif stage == 2 and sr <= self.traj_thr_up[1]: stage = 1
            elif stage == 3 and sr <= self.traj_thr_up[2]: stage = 2
            
        self.traj_stage = int(stage)
        self.traj_max_stage_reached = max(self.traj_max_stage_reached, stage)
        
        target_w = float(self.traj_stage_to_w[self.traj_stage])
        
        # Smooth update
        cur_w = self.traj_w
        alpha = self.traj_alpha_down if target_w < cur_w else self.traj_alpha_up
        self.traj_w = (1.0 - alpha) * cur_w + alpha * target_w
        
        # Logging
        self.traj_sched_n += 1
        if (stage != self.traj_stage) or (self.traj_sched_n % 20 == 0):
            print(f"[TrajMgr] Upd#{self.traj_sched_n} SR={batch_sr:.2f}(EMA:{sr:.2f}) "
                  f"Stage={stage} W={self.traj_w:.3f}")

    def retrieve_demos(self, env_ids, current_local_pos):
        """
        리셋된 환경(env_ids)의 큐브 위치(current_local_pos)와 가장 가까운 데모를 찾아서 매칭
        current_local_pos: (N, num_cubes, 3)
        """
        if not self._traj_enabled():
            return

        # Distance calculation
        # Env: (N, 1, C, 3) - Demo: (1, M, C, 3)
        diff = current_local_pos.unsqueeze(1) - self.demo_init_pos.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        total_dist = torch.sum(torch.sqrt(dist_sq), dim=-1) # (N, M)

        # Top-K
        K = min(5, self.demo_D)
        top_dists, top_indices = torch.topk(total_dist, k=K, dim=1, largest=False)
        
        # Random Select from Top-K
        rand_choice = torch.randint(0, K, (len(env_ids),), device=self.device)
        selected_indices = torch.gather(top_indices, 1, rand_choice.unsqueeze(1)).squeeze(1)
        
        self.demo_ids[env_ids] = selected_indices

        # Log for env 0
        if 0 in env_ids:
            idx = (env_ids == 0).nonzero(as_tuple=True)[0].item()
            err_val = torch.gather(top_dists, 1, rand_choice.unsqueeze(1)).squeeze(1)[idx].item()
            print(f"[TrajMgr] Env 0 -> Demo #{selected_indices[idx].item()} (Err: {err_val:.4f}m)")

    def compute_reward(self, step_buf, last_cmd_raw, is_clean_mask):
        """
        현재 스텝의 보상 계산
        """
        if not self._traj_enabled():
            return torch.zeros(step_buf.shape[0], device=self.device)

        # Step matching
        step_i = (step_buf.to(torch.long) - 1).clamp(0, self.max_len - 1)
        base_idx = (step_i * self.demo_K // max(self.max_len, 1)).clamp(0, self.demo_K - 1)
        
        target_action = self.demo_raw_actions[base_idx, self.demo_ids]
        if target_action.shape != last_cmd_raw.shape:
            if not self._warned_action_mismatch:
                print(
                    f"[TrajMgr] Runtime shape mismatch: target={tuple(target_action.shape)} "
                    f"last_cmd_raw={tuple(last_cmd_raw.shape)}. Return zero traj reward."
                )
                self._warned_action_mismatch = True
            return torch.zeros(step_buf.shape[0], device=self.device)
        
        # MSE
        mse = ((last_cmd_raw - target_action)**2).mean(dim=(1, 2))
        
        # Reward
        raw_traj = torch.exp(-mse / (self.traj_sigma**2 + 1e-6))
        
        # Clean하면 보상 1.0배 (원하면 조절 가능)
        phase_factor = torch.where(is_clean_mask, 1.0, 1.0)
        
        return self.traj_w * self.traj_step_scale * raw_traj * phase_factor
    
    @torch.no_grad()
    def debug_traj_metrics(self, step_buf, last_cmd_raw, mse_cap=0.15, T=2.0):
        """
        로그용: 현재 traj 매칭의 mse와 여러 mapping(exp/linear/sigmoid) 값을 계산해서 반환.
        학습에는 절대 사용하지 말고 print용으로만.
        """
        if not self._traj_enabled():
            return None

        # Step matching (compute_reward와 동일)
        step_i = (step_buf.to(torch.long) - 1).clamp(0, self.max_len - 1)
        base_idx = (step_i * self.demo_K // max(self.max_len, 1)).clamp(0, self.demo_K - 1)

        target_action = self.demo_raw_actions[base_idx, self.demo_ids]
        if target_action.shape != last_cmd_raw.shape:
            return None

        mse = ((last_cmd_raw - target_action) ** 2).mean(dim=(1, 2))  # (N,)

        # 1) exp (현재 방식)
        raw_exp = torch.exp(-mse / (self.traj_sigma ** 2 + 1e-6))

        # 2) linear clamp (포화 덜함)
        raw_lin = torch.clamp(1.0 - mse / mse_cap, 0.0, 1.0)

        # 3) batch-normalized sigmoid (상대적 랭킹 유지, 포화 덜함)
        m = mse.mean()
        s = mse.std().clamp_min(1e-6)
        raw_sig = torch.sigmoid(-(mse - m) / (s * T))

        return {
            "mse0": float(mse[0].item()),
            "exp0": float(raw_exp[0].item()),
            "lin0": float(raw_lin[0].item()),
            "sig0": float(raw_sig[0].item()),
            "mse_mean": float(m.item()),
            "mse_std": float(s.item()),
            "exp_mean": float(raw_exp.mean().item()),
            "exp_std": float(raw_exp.std().item()),
            "lin_mean": float(raw_lin.mean().item()),
            "lin_std": float(raw_lin.std().item()),
            "sig_mean": float(raw_sig.mean().item()),
            "sig_std": float(raw_sig.std().item()),
        }
