
import torch
import copy
import numpy as np

from .base import AsyncParallelCollector
import torch.multiprocessing as mp

import torchrl.policies as policies

from torchrl.env.get_env import *
from torchrl.env.continuous_wrapper import *
from collections import OrderedDict


class AsyncSingleTaskParallelCollector(AsyncParallelCollector):
    def __init__(
            self,
            reset_idx=False,
            **kwargs):
        self.reset_idx = reset_idx
        super().__init__(**kwargs)

    @staticmethod
    def eval_worker_process(
            shared_pf, env_info, shared_que, start_barrier, epochs, reset_idx):

        pf = copy.deepcopy(shared_pf).to(env_info.device)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        # env_info.env.eval()
        # env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())

            eval_rews = []

            done = False
    
            for idx in range(env_info.eval_episodes):
                if reset_idx:
                    eval_ob = env_info.env.reset_with_index(idx)
                else:
                    eval_ob = env_info.env.reset()
                rew = 0
        
                while not done:
                    act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
                    act = act.detach().cpu().numpy()
                    eval_ob, r, done, info = env_info.env.step( act )
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()

                eval_rews.append(rew)
                done = False
        

            shared_que.put({
                'eval_rewards': eval_rews,
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
    
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

        self.env_info.env_cls  = self.env_cls
        self.env_info.env_args = self.env_args

        for i in range(self.worker_nums):
            self.env_info.task_rank = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.train_epochs))
            p.start()
            self.workers.append(p)

        for i in range(self.eval_worker_nums):
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, self.reset_idx))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def eval_one_epoch(self):
        # self.eval_start_barrier.wait()
        eval_rews = []
        
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            eval_rews += worker_rst["eval_rewards"]
           

        return {
            'eval_rewards':eval_rews,
        }


class AsyncMultiTaskParallelCollectorUniform(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.tasks=self.task_list
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        self.progress_alpha = progress_alpha
  

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf = funcs["pf"]
        ob = ob_info["ob"]
        task_idx = env_info.task_rank
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies.ActionRepresentationGuassianContPolicy_v1)
        

        pf.eval()

        with torch.no_grad():
            if idx_flag:
                idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                if embedding_flag:
                    task_input = torch.zeros(env_info.num_tasks)
                    task_input[env_info.task_rank] = 1
                    task_input = task_input.unsqueeze(0).to(env_info.device)
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0), task_input,
                        [task_idx])
                else:
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0),
                        idx_input)
                act = out["action"]
                # act = act[0]
            else:
                if embedding_flag:
                    task_input = torch.zeros(env_info.num_tasks)
                    task_input[env_info.task_rank] = 1
                    task_input = task_input.unsqueeze(0).to(env_info.device)
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0), task_input)
                else:    
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0))
                act = out["action"]


        act = act.detach().cpu().numpy()
        if not env_info.continuous:
            act = act[0]
        
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, rewards, done, info = env_info.env.step(act)
        reward=rewards[task_idx]
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idxs": [env_info.task_rank],
            "rewards": [reward],
            "terminals": [done]
        }
        if embedding_flag:
            sample_dict["task_inputs"] = task_input.cpu().numpy()

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample( sample_dict, env_info.task_rank)

        return next_ob, done, reward, info

    @staticmethod
    def train_worker_process(cls, shared_funcs, env_info,
        replay_buffer, shared_que,
        start_barrier, epochs, start_epoch, task_name, shared_dict):

        replay_buffer.rebuild_from_tag()
        local_funcs = copy.deepcopy(shared_funcs)
        for key in local_funcs:
            local_funcs[key].to(env_info.device)

        # # Rebuild Env
        # env_info.env = env_info.env_cls(**env_info.env_args)

        # norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        # if norm_obs_flag:
        #     shared_dict[task_name] = {
        #         "obs_mean": env_info.env._obs_mean,
        #         "obs_var": env_info.env._obs_var
        #     }
            # print("Put", task_name)
        
        c_ob = {
            "ob": env_info.env.reset()
        }
        train_rew = 0
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'train_rewards': None,
                    'train_epoch_reward': None
                })
                continue
            if current_epoch > epochs:
                break

            for key in shared_funcs:
                local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

            train_rews = []
            train_epoch_reward = 0    

            for t in range(env_info.epoch_frames):

                next_ob, done, reward, info = cls.take_actions(local_funcs, env_info, c_ob, replay_buffer )
                c_ob["ob"] = next_ob
                train_rew += reward
                train_epoch_reward += reward
                train_rews.append(train_rew)
                train_rew = 0
                # if done:
                #     train_rews.append(train_rew)
                #     train_rew = 0

            # if norm_obs_flag:
            #     shared_dict[task_name] = {
            #         "obs_mean": env_info.env._obs_mean,
            #         "obs_var": env_info.env._obs_var
            #     }
                # print("Put", task_name)
            
            shared_que.put({
                'train_rewards':train_rews,
                'train_epoch_reward':train_epoch_reward
            })

    @staticmethod
    def eval_worker_process(shared_pf, 
        env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

        pf = copy.deepcopy(shared_pf).to(env_info.device)
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies.ActionRepresentationGuassianContPolicy_v1)

        # # Rebuild Env
        # env_info.env = env_info.env_cls(**env_info.env_args)

        # norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        # env_info.env.eval()
        # env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'eval_rewards': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())
            pf.eval()

            # print("Get", task_name)
            # if norm_obs_flag:
            #     env_info.env._obs_mean = shared_dict[task_name]["obs_mean"]
            #     env_info.env._obs_var = shared_dict[task_name]["obs_var"]
                # print(env_info.env._obs_mean)
                #  = {
                #     "obs_mean": env_info.env._obs_mean,
                #     "obs_var": env_info.env._obs_var
                # }

            eval_rews = []  

            done = False
          
            for idx in range(env_info.eval_episodes):

                eval_ob = env_info.env.reset()
                rew = 0

                task_idx = env_info.task_rank
                
                while not done:
               
                    if idx_flag:
                        idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                        if embedding_flag:
                            # embedding_input = torch.zeros(env_info.num_tasks)
                            task_input = torch.zeros(env_info.num_tasks)
                            task_input[env_info.task_rank] = 1
                            # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                            task_input = task_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), task_input, [task_idx] )
                        else:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), idx_input )
                    else:
                        if embedding_flag:
                            # embedding_input = torch.zeros(env_info.num_tasks)
                            task_input = torch.zeros(env_info.num_tasks)
                            task_input[env_info.task_rank] = 1
                            # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                            task_input = task_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), task_input)
                        else:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))

                    eval_ob, rs, done, info = env_info.env.step( act )
                    r=rs[task_idx]
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()
                   

                eval_rews.append(rew)
                done = False
      
            shared_que.put({
                'eval_rewards': eval_rews,
             
                'task_name': task_name
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


        self.shared_dict = self.manager.dict()

        

        assert self.worker_nums == len(self.task_list)

        self.env_info.env = self.env
       
        
        self.env_info.num_tasks = len(self.task_list)
       
        single_mt_env_args = {
            "task_name": None,
            "task_rank": 0,
            "num_tasks": len(self.task_list),
            "max_obs_dim": np.prod(self.env.observation_space.shape),
        }
        
      
        tasks=self.task_list
        for i, task in enumerate(tasks):
          
            
            self.env_info.task_rank = i
            
            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_name"] = task

            
            start_epoch = 0
            self.env_info.env_args["task_rank"] = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.train_epochs, start_epoch, task, self.shared_dict))
            p.start()
            self.workers.append(p)
            # i += 1



        assert self.eval_worker_nums == len(self.task_list)
      
        self.env_info.env = self.env
        self.env_info.num_tasks = len(self.task_list)
       
        single_mt_env_args = {
            "task_name": None,
            "task_rank": 0,
            "num_tasks": len(self.task_list),
            "max_obs_dim": np.prod(self.env.observation_space.shape),
        }

        for i, task in enumerate(tasks):
           

            self.env_info.task_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_name"] = task

            start_epoch = 0
  

            self.env_info.env_args["task_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, start_epoch, task, self.shared_dict))
            eval_p.start()
            self.eval_workers.append(eval_p)


    def eval_one_epoch(self):
        
        eval_rews = []
      
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())

        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                eval_rews += worker_rst["eval_rewards"]
                tasks_result.append((worker_rst["task_name"], 
                np.mean(worker_rst["eval_rewards"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, eval_rewards in tasks_result:
            
            dic[task_name+"_eval_rewards"] = eval_rewards
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)

        dic['eval_rewards']      = eval_rews
        

        return dic


    def train_one_epoch(self):
        train_rews = []
        train_epoch_reward = 0

        for key in self.shared_funcs:
            self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
        
        active_worker_nums = 0
        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            
            if worker_rst["train_rewards"] is not None:
                
                train_rews += worker_rst["train_rewards"]
                train_epoch_reward += worker_rst["train_epoch_reward"]
                active_worker_nums += 1
        self.active_worker_nums = active_worker_nums
     
        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }

class AsyncMultiTaskParallelCollectorForActionRepresentation(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.tasks=self.task_list
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        self.progress_alpha = progress_alpha
        self.pf_state=self.pf[0]
        self.pf_task=self.pf[1]
        self.pf_action=self.pf[2]
  

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf_state = funcs["pf_state"]
        pf_task = funcs["pf_task"]
        pf_action = funcs["pf_action"]
        ob = ob_info["ob"]
        task_idx = env_info.task_rank
        

        pf_state.eval()
        pf_task.eval()
        pf_action.eval()

        with torch.no_grad():
        
            idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
            task_input = torch.zeros(env_info.num_tasks)
            task_input[env_info.task_rank] = 1
            task_input = task_input.to(env_info.device).unsqueeze(0)
            ob = torch.Tensor( ob ).to(env_info.device).unsqueeze(0)
            embedding = pf_task.forward(task_input)
            representation = pf_state.forward(ob)

            out = pf_action.explore(representation, embedding)
            
            act = out["action"]
            # act = act[0]
            


        act = act.detach().cpu().numpy()
        if not env_info.continuous:
            act = act[0]
        
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, rewards, done, info = env_info.env.step(act)
        reward=rewards[task_idx]
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idxs": [env_info.task_rank],
            "rewards": [reward],
            "terminals": [done],
            "task_inputs": task_input.cpu().numpy()
        }
       

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample( sample_dict, env_info.task_rank)

        return next_ob, done, reward, info

    @staticmethod
    def train_worker_process(cls, shared_funcs, env_info,
        replay_buffer, shared_que,
        start_barrier, epochs, start_epoch, task_name, shared_dict):

        replay_buffer.rebuild_from_tag()
        local_funcs = copy.deepcopy(shared_funcs)
        for key in local_funcs:
            local_funcs[key].to(env_info.device)
        
        c_ob = {
            "ob": env_info.env.reset()
        }
        train_rew = 0
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'train_rewards': None,
                    'train_epoch_reward': None
                })
                continue
            if current_epoch > epochs:
                break

            for key in shared_funcs:
                local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

            train_rews = []
            train_epoch_reward = 0    

            for t in range(env_info.epoch_frames):

                next_ob, done, reward, info = cls.take_actions(local_funcs, env_info, c_ob, replay_buffer )
                c_ob["ob"] = next_ob
                train_rew += reward
                train_epoch_reward += reward
                train_rews.append(train_rew)
                train_rew = 0
                # if done:
                #     train_rews.append(train_rew)
                #     train_rew = 0

            # if norm_obs_flag:
            #     shared_dict[task_name] = {
            #         "obs_mean": env_info.env._obs_mean,
            #         "obs_var": env_info.env._obs_var
            #     }
                # print("Put", task_name)
            
            shared_que.put({
                'train_rewards':train_rews,
                'train_epoch_reward':train_epoch_reward
            })

    @staticmethod
    def eval_worker_process(shared_pf_state,shared_pf_task,shared_pf_action, 
        env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

        pf_state = copy.deepcopy(shared_pf_state).to(env_info.device)
        pf_task = copy.deepcopy(shared_pf_task).to(env_info.device)
        pf_action = copy.deepcopy(shared_pf_action).to(env_info.device)

        # # Rebuild Env
        # env_info.env = env_info.env_cls(**env_info.env_args)

        # norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        # env_info.env.eval()
        # env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'eval_rewards': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break
            pf_state.load_state_dict(shared_pf_state.state_dict())
            pf_task.load_state_dict(shared_pf_task.state_dict())
            pf_action.load_state_dict(shared_pf_action.state_dict())
            pf_state.eval()
            pf_task.eval()
            pf_action.eval()
            # print("Get", task_name)
            # if norm_obs_flag:
            #     env_info.env._obs_mean = shared_dict[task_name]["obs_mean"]
            #     env_info.env._obs_var = shared_dict[task_name]["obs_var"]
                # print(env_info.env._obs_mean)
                #  = {
                #     "obs_mean": env_info.env._obs_mean,
                #     "obs_var": env_info.env._obs_var
                # }

            eval_rews = []  

            done = False
          
            for idx in range(env_info.eval_episodes):

                eval_ob = env_info.env.reset()
                rew = 0

                task_idx = env_info.task_rank
                
                while not done:
               
                  
                  
                    # embedding_input = torch.zeros(env_info.num_tasks)
                    task_input = torch.zeros(env_info.num_tasks)
                    task_input[env_info.task_rank] = 1
                    task_input = task_input.unsqueeze(0).to(env_info.device)
                    eval_ob =  torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0)
                    representation = pf_state.forward(eval_ob)
                    embedding = pf_task.forward(task_input)
                    act = pf_action.eval_act(representation, embedding)
                    eval_ob, rs, done, info = env_info.env.step( act )
                    r=rs[task_idx]
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()
                   

                eval_rews.append(rew)
                done = False
      
            shared_que.put({
                'eval_rewards': eval_rews,
             
                'task_name': task_name
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


        self.shared_dict = self.manager.dict()

        

        assert self.worker_nums == len(self.task_list)

        self.env_info.env = self.env
       
        
        self.env_info.num_tasks = len(self.task_list)
       
        single_mt_env_args = {
            "task_name": None,
            "task_rank": 0,
            "num_tasks": len(self.task_list),
            "max_obs_dim": np.prod(self.env.observation_space.shape),
        }
        
      
        tasks=self.task_list
        for i, task in enumerate(tasks):
          
            
            self.env_info.task_rank = i
            
            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_name"] = task

            
            start_epoch = 0
            self.env_info.env_args["task_rank"] = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.train_epochs, start_epoch, task, self.shared_dict))
            p.start()
            self.workers.append(p)
            # i += 1



        assert self.eval_worker_nums == len(self.task_list)
      
        self.env_info.env = self.env
        self.env_info.num_tasks = len(self.task_list)
       
        single_mt_env_args = {
            "task_name": None,
            "task_rank": 0,
            "num_tasks": len(self.task_list),
            "max_obs_dim": np.prod(self.env.observation_space.shape),
        }

        for i, task in enumerate(tasks):
           

            self.env_info.task_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_name"] = task

            start_epoch = 0
  

            self.env_info.env_args["task_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf_state"],self.shared_funcs["pf_task"],self.shared_funcs["pf_action"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, start_epoch, task, self.shared_dict))
            eval_p.start()
            self.eval_workers.append(eval_p)


    def eval_one_epoch(self):
        
        eval_rews = []
      
        self.shared_funcs["pf_state"].load_state_dict(self.funcs["pf_state"].state_dict())
        self.shared_funcs["pf_task"].load_state_dict(self.funcs["pf_task"].state_dict())
        self.shared_funcs["pf_action"].load_state_dict(self.funcs["pf_action"].state_dict())
        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                eval_rews += worker_rst["eval_rewards"]
                tasks_result.append((worker_rst["task_name"], 
                np.mean(worker_rst["eval_rewards"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, eval_rewards in tasks_result:
            
            dic[task_name+"_eval_rewards"] = eval_rewards
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)

        dic['eval_rewards']      = eval_rews
        

        return dic


    def train_one_epoch(self):
        train_rews = []
        train_epoch_reward = 0

        for key in self.shared_funcs:
            self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
        
        active_worker_nums = 0
        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            
            if worker_rst["train_rewards"] is not None:
                
                train_rews += worker_rst["train_rewards"]
                train_epoch_reward += worker_rst["train_epoch_reward"]
                active_worker_nums += 1
        self.active_worker_nums = active_worker_nums
     
        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }
    def to(self, device):
        for func in self.funcs:
            self.funcs[func].to(device)
    @property
    def funcs(self):
        return {
            "pf_state": self.pf[0],
            "pf_task": self.pf[1],
            "pf_action": self.pf[2]
        }
    
