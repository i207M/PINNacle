#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:53:56 2021

@author: bmoseley

本模块定义了FBPINN的训练流程
可以忽略 full_model_PINN 函数和 PINNTrainer 类，后续的代码没有用到
"""

# This module defines trainer classes for FBPINNs and PINNs. It is the main entry point for training FBPINNs and PINNs
# To train a FBPINN / PINN, use a Constants object to setup the problem and define its hyperparameters, and pass that 
# to one of the trainer classes defined here


import time

import numpy as np
import torch
import torch.optim
from torch.profiler import profile, record_function, ProfilerActivity
import itertools
from scipy.interpolate import griddata
import scipy

import plot_main
import losses
from trainersBase import _Trainer
from constants import Constants
from domains import ActiveRectangularDomainND


## HELPER FUNCTIONS


def _x_random(subdomain_xs, batch_size, device):
    "Get flattened random samples of x"
    s = torch.tensor([[x.min(), x.max()] for x in subdomain_xs], dtype=torch.float32, device=device).T.unsqueeze(1)# (2, 1, nd)
    x_random = s[0]+(s[1]-s[0])*torch.rand((np.prod(batch_size),len(subdomain_xs)), device=device)# random samples in domain
    return x_random

def _x_mesh(subdomain_xs, batch_size, device):
    "Get flattened samples of x on a mesh"
    x_mesh = [torch.linspace(x.min(), x.max(), b, device=device) for x,b in zip(subdomain_xs, batch_size)]
    x_mesh = torch.stack(torch.meshgrid(*x_mesh), -1).view(-1, len(subdomain_xs))# nb torch.meshgrid uses np indexing="ij"
    return x_mesh

def full_model_FBPINN(x, models, c, D):
    """
    Get the full FBPINN prediction over all active and fixed models (forward inference only)
    本函数对FBPINN的进行不记录计算图的前向求值, 其结果不可用来反向传播误差
    """
    
    def _single_model(im):# use separate function to ensure computational graph/memory is released
        
        x_ = x.detach().clone().requires_grad_(True)
        
        # normalise, run model, add window function
        mu, sd = D.n_torch[im]# (nd)
        y = models[im]((x_-mu)/sd)
        y_raw = y.detach().clone()
        y = y*c.Y_N[1] + c.Y_N[0]
        y = D.w[im](x_)*y
        
        # get gradients
        yj = c.P.get_gradients(x_, y)# problem-specific
        
        # detach from graph
        yj = [t.detach() for t in yj]
        
        # apply boundary conditions (for QC only)
        yj_bc = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
        
        return yj, yj_bc, y_raw
    
    # run all models
    yjs, yjs_bc, ys_raw = [], [], []
    for im in D.active_fixed_ims:
        yj, yj_bc, y_raw = _single_model(im)
        
        # add to model lists
        yjs.append(yj); yjs_bc.append(yj_bc); ys_raw.append(y_raw)
    
    # sum across models
    yj = [torch.sum(torch.stack(ts, -1), -1) for ts in zip(*yjs)]# note zip(*) transposes
    
    # apply boundary condition to summed solution
    yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
    
    return yj, yjs_bc, ys_raw

def full_model_PINN(x, model, c):
    """Get the full PINN prediction (forward inference only)"""
    
    # get normalisation values
    xmin, xmax = torch.tensor([[x.min(), x.max()] for x in c.SUBDOMAIN_XS], dtype=torch.float32, device=x.device).T
    mu = (xmin + xmax)/2; sd = (xmax - xmin)/2
    
    # get full model solution using test data
    x_ = x.detach().clone().requires_grad_(True)
    y = model((x_-mu)/sd)
    y_raw = y.detach().clone()
    y = y*c.Y_N[1] + c.Y_N[0]
    
    # get gradients
    yj = c.P.get_gradients(x_, y)# problem-specific
    
    # detach from graph
    yj = [t.detach() for t in yj]
    
    # apply boundary conditions
    yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
    
    return yj, y_raw


## MAIN TRAINER CLASSES


class FBPINNTrainer(_Trainer):
    "FBPINN model trainer class"
    
    def _train_step(self, models, optimizers, c, D, i, active_updated):# use separate function to ensure computational graph/memory is released
        """
        本函数让FBPINN训练迭代一步
        """
        
        def forward_step(in_segments, in_domain=False):
            """
            本函数对FBPINN进行记录计算图的前向求值
            """
            ## RUN MODELS (ACTIVE AND FIXED NEIGHBOURS)
            xs, yjs = [], []
            isegim2j2j3 = dict() # mapping from (iseg, im) to (j2, j3), because iseg and im are global and unique identifiers
            for im,_ in D.active_fixed_neighbours_ims:
                # for every active or fixed model
                x = [in_segments[iseg] for iseg in D.m[im]]
                x = torch.cat(x, dim=0).detach().clone().requires_grad_(True) # (N, nd), concat to obtain the input 'x' for this model
                
                # update mapping from (iseg, im) to (j2, j3)
                if not in_domain:
                    j_prev = 0
                    for iseg in D.m[im]:
                        isegim2j2j3[(iseg, im)] = (j_prev, j_prev + in_segments[iseg].shape[0])
                        j_prev += in_segments[iseg].shape[0]
                
                # normalise, run model, add window function
                mu, sd = D.n_torch[im] # (nd)
                y = models[im]((x-mu)/sd)
                y = y*c.Y_N[1] + c.Y_N[0] # Constants.Y_N: unnormalization parameters
                y = D.w[im](x)*y # multiply window functions but do not sum
                
                # get gradients
                yj = c.P.get_gradients(x, y) # problem-specific . get_gradients returns both the value and the gradient
                
                # add to model lists
                yjs.append(yj)
                xs.append(x)
            
            ## SUM OVERLAPPING MODELS, APPLY BCS (ACTIVE)
            yjs_sum = [] # the final output for each active model on its input
            for im,i1 in D.active_ims:
                # for every active model
                
                # for each segment in model
                yjs_segs = []
                for iseg in D.m[im]:
                    
                    # for each model which contributes to that segment
                    yjs_seg = []
                    for im2,j1,j2,j3 in D.s[iseg]: # im2: guest model identifier; j1: guest model's index in D.active_fixed_neighbours_ims
                        # get start_idx, end_idx
                        start_idx, end_idx = (j2, j3) if in_domain else isegim2j2j3[(iseg, im2)]
                        
                        # get model yj segment iseg
                        yj = yjs[j1] # j1 is the index of yj for model im2 in yjs above
                        if im2 == im: yj = [t[start_idx:end_idx]           for t in yj] # get appropriate segment
                        else:         yj = [t[start_idx:end_idx].detach()  for t in yj]
                        # detach(): does not require grad computed w.r.t windowed output of neighboring models
                        
                        # add to model list
                        yjs_seg.append(yj)
                    
                    # print([[t.shape for t in ts] for ts in zip(*yjs_seg)][0])
                    # sum across models
                    yj_seg = [torch.sum(torch.stack(ts, -1), -1) for ts in zip(*yjs_seg)]# note zip(*) transposes
                    
                    # add to segment list
                    yjs_segs.append(yj_seg)
                
                # concat (across segments) to obtain summed windowed output for this active model
                yj = [torch.cat(ts) for ts in zip(*yjs_segs)]# note zip(*) transposes
                
                # apply boundary conditions
                x = xs[i1] # the input 'x' for this active model
                yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
                
                # add to model list
                yjs_sum.append(yj)
                
                #if (i % c.SUMMARY_FREQ) == 0: 
                #    print(*[t.shape for t in yj])# should be the same as above!
                #    print("rep 2",flush=True)
            
            return xs, yjs_sum, yjs

        # 1) 清零梯度
        ## ZERO PARAMETER GRADIENTS, SET TO TRAIN
        for optimizer in optimizers: optimizer.zero_grad() # zero_grad()
        for model in models: model.train()
        
        # 2) 采样区域内采样点（及其掩码）、边界采样点、数据采样点
        ## RANDOMLY SAMPLE ALL SEGMENTS
        if active_updated or c.RANDOM: # cached value is outdated
            self.x_seg_cache, self.mask_seg_cache, self.bd_seg_cache, self.data_cache = None, None, None, None
        if self.x_seg_cache != None:
            x_segments, mask_segments, bd_segments, data_segments = self.x_seg_cache, self.mask_seg_cache, self.bd_seg_cache, self.data_cache
        else:
            x_segments, mask_segments, bd_segments, data_segments = D.sample_segments() #sample the input 'x' for each segment
            self.x_seg_cache, self.mask_seg_cache, self.bd_seg_cache, self.data_cache = x_segments, mask_segments, bd_segments, data_segments

        # 3) 拼接得到每个子区域的区域采样点的掩码
        ## GENERATE MASKS
        if self.need_mask: # mask_segments != None
            masks = []
            for im,_ in D.active_fixed_neighbours_ims:
                mask = [mask_segments[iseg] for iseg in D.m[im]]
                mask = torch.cat(mask, dim=0)
                masks.append(mask)
        
        # 4) 前向求值
        xs, yjs_sum, yjs = forward_step(x_segments, in_domain=True)
        if self.need_bd: # bd_segments != None
            bds, valjs_sum, valjs = forward_step(bd_segments)
        if self.need_od:
            ods, odjs_sum, odjs = forward_step(data_segments)
            
        # 5) 对每个子区域上的模型分别反向传播误差、更新参数
        ## BACKPROPAGATE LOSS (ACTIVE)
        # Update parameters for im-th PINN/model (w_im) corresponding to the im-th subdomain
        for im,i1 in D.active_ims:
            # calculate physics loss
            x, yj = xs[i1], yjs_sum[i1]
            if self.need_mask:
                if masks[i1].any():
                    x_masked = x[masks[i1]]
                    yj_masked = (yji[masks[i1]] for yji in yj)
                    loss_physics = c.P.physics_loss(x_masked, *yj_masked)# problem-specific
                else: # every point in the subdomain is masked out
                    loss_physics = torch.tensor(0.).requires_grad_(True)
            else:
                loss_physics = c.P.physics_loss(x, *yj)
            # calculate boundary loss
            loss_boundary = torch.tensor(0.).requires_grad_(True)
            if self.need_bd:
                bd, valj = bds[i1], valjs_sum[i1]
                if np.prod(bd.shape) > 0:
                    loss_boundary = c.P.bd_loss(bd, *valj)
            # calculate data loss
            loss_data = torch.tensor(0.).requires_grad_(True)
            if self.need_od:
                od, odj = ods[i1], odjs_sum[i1]
                if np.prod(od.shape) > 0:
                    loss_data = c.P.data_loss(od, *odj)
            loss = loss_physics + c.BOUNDARY_WEIGHT * loss_boundary + c.DATALOSS_WEIGHT * loss_data
            loss.backward()
            optimizers[im].step()
        
        # return result
        return ([t.detach() for t in xs], 
                [[t.detach() for t in ts] for ts in yjs], 
                [[t.detach() for t in ts] for ts in yjs_sum], loss.item())
    
    def _test_step(self, x_test, mesh_test, yj_true,   xs, yjs, yjs_sum,   models, c, D, i, mstep, fstep, writer, yj_test_losses, train_tot_loss):# use separate function to ensure computational graph/memory is released
        
        # get full model solution using test data
        yj_full, yjs_full, ys_full_raw = full_model_FBPINN(x_test, models, c, D)
        print(x_test.shape, yj_true[0].shape, yj_full[0].shape)
        
        # first dim select (retain dimensions where there is exact solution)
        if hasattr(c.P, "exact_dim_select"):
            fds = c.P.exact_dim_select
        else:
            fds = slice(None)
        yj_full_fds = [el for el in yj_full]
        yj_full_fds[0] = yj_full_fds[0][:,fds]
        for ii in range(len(yjs_full)):
            tmp = list(yjs_full[ii])
            tmp[0] = tmp[0][:,fds]
            yjs_full[ii] = tuple(tmp)
        for e in ys_full_raw:
            e = e[:,fds]
        # for ttt in yjs_sum[0]:
        #     print(ttt.shape)
        # print("+==+")
        for ii in range(len(yjs_sum)):
            yjs_sum[ii][0] = yjs_sum[ii][0][:,fds]
        
        # save model train loss
        writer.add_scalar("loss_istep/tot_loss/train", train_tot_loss, i + 1)
        writer.add_scalar("loss_mstep/tot_loss/train", train_tot_loss, mstep)
        writer.add_scalar("loss_zfstep/tot_loss/train", train_tot_loss, fstep)

        # get full model solution on test boundary points
        if self.need_bd:
            bd_test = torch.tensor(c.P.sample_bd(c.BOUNDARY_BATCH_SIZE_TEST), dtype=torch.float32, device=self.device)
            bd_yj_full, bd_yjs_full, bd_ys_full_raw = full_model_FBPINN(bd_test, models, c, D)
            bd_loss = c.P.bd_loss(bd_test, *bd_yj_full).item()
            writer.add_scalar("loss_istep/zboundary/test", bd_loss, i + 1)

        # get losses over test data
        yj_test_loss = [losses.l2_rel_err(a[mesh_test], b[mesh_test]).item() for a,b in zip(yj_true, yj_full_fds)]
        physics_loss = c.P.physics_loss(x_test[mesh_test], *(yj[mesh_test] for yj in yj_full)).item()# problem-specific
        yj_test_losses.append([i + 1, mstep, fstep]+yj_test_loss+[physics_loss]+([bd_loss]if self.need_bd else []))
        for j,l in enumerate(yj_test_loss): 
            for step,tag in zip([i + 1, mstep, fstep], ["istep", "mstep", "zfstep"]):
                #istep: step number, mstep: number of weights updated, zfstep:  number of FLOPS
                writer.add_scalar("loss_%s/yj%i/test"%(tag,j), l, step)
        writer.add_scalar("loss_istep/zphysics/test", physics_loss, i + 1)
        writer.add_scalar("loss_mstep/zphysics/test", physics_loss, mstep)
        writer.add_scalar("loss_zfstep/zphysics/test", physics_loss, fstep)
        if self.need_bd:
            writer.add_scalar("loss_istep/phy_plus_w_mult_bd/test",physics_loss+c.BOUNDARY_WEIGHT*bd_loss, i + 1)
            writer.add_scalar("loss_mstep/phy_plus_w_mult_bd/test",physics_loss+c.BOUNDARY_WEIGHT*bd_loss, mstep)
            writer.add_scalar("loss_zfstep/phy_plus_w_mult_bd/test",physics_loss+c.BOUNDARY_WEIGHT*bd_loss, fstep)
        
        # get L2RE/MSE over ref data or copy L2RE/MSE from test data
        if hasattr(c.P, "ref_data"):
            ref_x_torch = torch.tensor(c.P.ref_x, device=x_test.device)
            yj_full2, _, _ = full_model_FBPINN(ref_x_torch, models, c, D)
            a1, a2 = torch.from_numpy(c.P.ref_y).to(self.device), yj_full2[0]
        else:
            a1, a2 = yj_true[0], yj_full_fds[0]
        l2re, l1re = losses.l2_rel_err(a1, a2), losses.l1_rel_err(a1, a2)
        mse, mae = losses.l2_loss(a1, a2), losses.l1_loss(a1, a2)
        maxe, csve = losses.max_err(a1, a2), losses.err_csv(a1, a2)
        for e,ee in zip([l2re, mse, maxe, csve],["l2re","mse", "maxe", "csve"]):
            writer.add_scalar("loss_istep/"+ee+"/test", e, i + 1)
            writer.add_scalar("loss_mstep/"+ee+"/test", e, mstep)
            writer.add_scalar("loss_zfstep/"+ee+"/test", e, fstep)
        #writer.add_scalar("")

        # calculate fRMSE
        fRMSE_l, fRMSE_h = 5, 13
        err_low, err_mid, err_high = np.float64(0.), np.float64(0.), np.float64(0.)
        if self.calc_frmse:
            # use [:,0] to select the first output dimension, only calculate fRMSE for the first output dimension
            if hasattr(c.P, "ref_data"):
                res = scipy.interpolate.LinearNDInterpolator(self.test_x_delaunay, c.P.ref_y[:,0] - yj_full2[0][:,0].cpu().numpy())\
                    (self.frmse_sample_x.reshape((-1, c.P.d[0])))
            else:
                frmse_x_torch = torch.tensor(self.frmse_sample_x.reshape((-1, c.P.d[0])).astype(np.float32), device=x_test.device)
                frmse_y, _, _ = full_model_FBPINN(frmse_x_torch, models, c, D)
                res = self.frmse_y_true[:,0] - frmse_y[0][:,0].cpu().numpy()
            res = res.reshape(self.frmse_sample_x.shape[:-1])
            err = np.abs(np.fft.rfftn(res)) ** 2 / res.size
            if c.P.d[0] == 1:
                err_low = err[:fRMSE_l].mean()
                err_mid = err[fRMSE_l:fRMSE_h].mean()
                err_high = err[fRMSE_h:].mean()
            else:
                err_low, err_mid, err_high = 0.0, 0.0, 0.0
                err_low_cnt, err_mid_cnt, err_high_cnt = 0, 0, 0
                for ids in itertools.product(*[range((k+1)//2) for k in err.shape[:-1]]):
                    freq2 = sum(i ** 2 for i in ids)
                    ilow = min(int(np.sqrt(max(0, fRMSE_l**2 - freq2))), err.shape[-1])
                    ihigh = min(int(np.sqrt(max(0, fRMSE_h**2 - freq2))), err.shape[-1])

                    err_low += err[(*ids, slice(None, ilow, None))].sum()
                    err_mid += err[(*ids, slice(ilow, ihigh, None))].sum()
                    err_high += err[(*ids, slice(ihigh, None, None))].sum()

                    err_low_cnt += ilow 
                    err_mid_cnt += ihigh - ilow
                    err_high_cnt += err.shape[-1] - ihigh
                
                err_low /= err_low_cnt # calculate mean square error
                err_mid /= err_mid_cnt
                err_high /= err_high_cnt


        # PLOTS
        
        if (i + 1) % c.TEST_FREQ == 0:
            
            # if need to mask
            if self.need_mask:
                def msk(r):
                    return [ [not el]*r for el in mesh_test]
                yj_true = [np.ma.masked_where(msk(e.cpu().shape[1]),e.cpu()) for e in yj_true]
                yj_full = [np.ma.masked_where(msk(e.cpu().shape[1]),e.cpu()) for e in yj_full_fds]
                yjs_full = [(np.ma.masked_where(msk(d.cpu().shape[1]),d.cpu()) for d in e) for e in yjs_full]
                ys_full_raw = [np.ma.masked_where(msk(e.cpu().shape[1]),e.cpu()) for e in ys_full_raw]
            
            # save figures
            fs = plot_main.plot_FBPINN(x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw,   yj_test_losses,   c, D, i + 1)
            if fs is not None: self._save_figs(i, fs)
        
        del x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw# fixes weird over-allocation of GPU memory bug caused by plotting (?)
        
        return yj_test_losses, (l2re.item(), l1re.item(), mse.item(), mae.item(), maxe.item(), csve.item(), err_low.item(), err_mid.item(), err_high.item())
    
    def train(self):
        "Train model"
        
        c, device, writer = self.c, self.device, self.writer #self.c is the Constants Object passed to _Trainer upon initialization
        
        # define domain
        D = ActiveRectangularDomainND(c.SUBDOMAIN_XS, c.SUBDOMAIN_WS, problem=c.P, device=device)
        D.update_sampler(c.BATCH_SIZE, c.RANDOM, c.BOUNDARY_BATCH_SIZE)
        A = c.ACTIVE_SCHEDULER(c.N_STEPS, D, *c.ACTIVE_SCHEDULER_ARGS) #Scheduler copys the N_STEPS from Constants class
        
        # create models
        models = [c.MODEL(c.P.d[0], c.P.d[1], c.N_HIDDEN, c.N_LAYERS) for _ in range(D.N_MODELS)]# problem-specific
        
        # create optimisers
        optimizers = [torch.optim.Adam(model.parameters(), lr=c.LRATE) for model in models]
        
        # put models on device
        for model in models: model.to(device)

        # get exact solution if it exists
        x_test = _x_mesh(c.SUBDOMAIN_XS, c.BATCH_SIZE_TEST, device)
        mesh_test = c.P.mask_x(x_test) if self.need_mask else slice(None)
        yj_true = c.P.exact_solution(x_test, c.BATCH_SIZE_TEST)# problem-specific
        
        # prepare for frmse calculation
        self.frmse_init()

        ## TRAIN
        
        mstep, fstep, yj_test_losses = 0, 0, []
        start, gpu_time = time.time(), 0.
        self.x_seg_cache, self.mask_seg_cache, self.bd_seg_cache, self.data_cache = None, None, None, None
        for i,active in enumerate(A):
            
            # update active if required
            if active is not None: 
                D.update_active(active)
                print(i, "Active updated:\n", active)
                active_updated = True
            else:
                active_updated = False
                
            gpu_start = time.time()
            xs, yjs, yjs_sum, loss = self._train_step(models, optimizers, c, D, i, active_updated)
            for im,i1 in D.active_ims: mstep += models[im].size# record number of weights updated
            for im,i1 in D.active_fixed_neighbours_ims: fstep += models[im].flops(xs[i1].shape[0])# record number of FLOPS
            gpu_time += time.time()-gpu_start
            
            # METRICS
            
            if (i + 1) % c.SUMMARY_FREQ == 0: #Tests and generates a figure every SUMMARY_FREQ(default 5000) steps
                
                # set counters
                rate, gpu_time = c.SUMMARY_FREQ / gpu_time, 0.
                
                # print summary
                self._print_summary(i, loss, rate, start)
                
                # test step
                yj_test_losses, l2l1s = self._test_step(x_test, mesh_test, yj_true,   xs, yjs, yjs_sum,   models, c, D, i, mstep, fstep, writer, yj_test_losses, loss)
            
            # SAVE
            
            if (i + 1) % c.MODEL_SAVE_FREQ == 0:
                
                # save models, losses and active array
                for im,model in enumerate(models):
                    self._save_model(i, model, im)
                np.save(c.MODEL_OUT_DIR+"active_%.8i.npy"%(i + 1), D.active)
                np.save(c.MODEL_OUT_DIR+"loss_%.8i.npy"%(i + 1), np.array(yj_test_losses))
        end_loop_i = i

        # cleanup
        writer.close()
        print("Finished training")

        # convenient logging
        usetime = time.time() - start
        train_loss = loss
        # l2l1s is a tuple: (l2re_, l1re_, mse_, mae_, maxe_, csve_)
        if hasattr(c, "hyperparam_name"):
            if c.hyperparam_name == "width":
                f_log = open("benchmark_results/hyperparam/width/"+c.P.name+"_"+str(c.hyperparam_value)+"_"+str(c.SEED), 'w')
            elif c.hyperparam_name == "div":
                f_log = open("benchmark_results/hyperparam/div/"+c.P.name+"_"+str(c.hyperparam_value)+"_"+str(c.SEED), 'w')
        elif hasattr(c, "parameterized_value"):
            f_log = open("benchmark_results/parampde/"+c.P.name+"_"+str(c.parameterized_value)+"_"+str(c.SEED), 'w')
        else:
            f_log = open("benchmark_results/"+ ("fb" if D.N_MODELS > 1 else "ctrl") +"/"+c.P.name+"_"+str(c.SEED), 'w')
        f_log.write(str(usetime)+" "+str(train_loss)+" "+" ".join([str(l) for l in l2l1s])+" "+"_".join([str(len(arr)-1) for arr in c.SUBDOMAIN_XS]))
        f_log.close()
        # draw figures
        figpath = self.c.SUMMARY_OUT_DIR+"%s_%.8i.png"%("train-test", end_loop_i + 1)
        from shutil import copyfile
        copyfile(figpath, "benchmark_results/figs/"+ ("fb/" if D.N_MODELS > 1 else "ctrl/")+c.P.name+"_"+str(c.SEED)+".png")
    
    def frmse_init(self):
        c = self.c
        self.calc_frmse = True
        if hasattr(c.P, "mask_x") or c.P.d[0] > 3:
            self.calc_frmse = False
            return
        # generate mesh
        ptn = 3e4 # generate about 3e4 uniform sampling points in the domain
        for i in range(c.P.d[0]):
            ptn /= c.P.bbox[i * 2 + 1] - c.P.bbox[i * 2]
        ptn = ptn ** (1 / c.P.d[0])
        xlist = [np.linspace(c.P.bbox[i * 2], c.P.bbox[i * 2 + 1], int(np.ceil((c.P.bbox[i*2+1] - c.P.bbox[i*2]) * ptn)) + 1, endpoint=False)[1:] \
                 for i in range(c.P.d[0])]
        self.frmse_sample_x = np.stack(np.meshgrid(*xlist), axis=-1)
        # prepare calculation
        if hasattr(c.P, "ref_data"):
            self.test_x_delaunay = scipy.spatial.Delaunay(c.P.ref_x)
        else:
            frmse_x_torch = torch.tensor(self.frmse_sample_x.reshape((-1, c.P.d[0])))
            self.frmse_y_true = c.P.exact_solution(frmse_x_torch, self.frmse_sample_x.shape[:-1])[0].cpu().numpy()


"""class PINNTrainer(_Trainer):
    "Standard PINN model trainer class"
    
    def _train_step(self, model, optimizer, c, i, mu, sd, device):# use separate function to ensure computational graph/memory is released
        
        optimizer.zero_grad()
        model.train()
        
        sampler = _x_random if c.RANDOM else _x_mesh #_x_random and _x_mesh are samplers for regular pinns
        x = sampler(c.SUBDOMAIN_XS, c.BATCH_SIZE, device).requires_grad_(True) #x is of shape (np.prod(batch_size), n_dims)
        #for example, if batch_size is (2,3) and it is a 2D problem, then x is [[x1,y1], [x2,y2], [...], [...], [...], [...]]
        y = model((x-mu)/sd)
        y = y*c.Y_N[1] + c.Y_N[0]
        
        # get gradients
        yj = c.P.get_gradients(x, y)# problem-specific
    
        # apply boundary conditions
        yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific

        # backprop loss
        loss = c.P.physics_loss(x, *yj)# problem-specific
        loss.backward()
        optimizer.step()
        
        if (i % c.SUMMARY_FREQ) == 0: 
            print(*[t.shape for t in yj], x.shape)
                
        # return result
        return x.detach(), [t.detach() for t in yj], loss.item()
    
    def _test_step(self, x_test, yj_true,   x, yj,   model, c, i, mstep, fstep, writer, yj_test_losses):# use separate function to ensure computational graph/memory is released
        
        # get full model solution using test data
        yj_full, y_full_raw = full_model_PINN(x_test, model, c)
        print(x_test.shape, yj_true[0].shape, yj_full[0].shape)
        
        # get losses over test data
        yj_test_loss = [losses.l2_rel_err(a,b).item() for a,b in zip(yj_true, yj_full)]
        physics_loss = c.P.physics_loss(x_test, *yj_full).item()# problem-specific
        yj_test_losses.append([i + 1, mstep, fstep]+yj_test_loss+[physics_loss])
        for j,l in enumerate(yj_test_loss): 
            for step,tag in zip([i + 1, mstep, fstep], ["istep", "mstep", "zfstep"]):
                writer.add_scalar("loss_%s/yj%i/test"%(tag,j), l, step)
        writer.add_scalar("loss_istep/zphysics/test", physics_loss, i + 1)
        writer.add_scalar("loss_mstep/zphysics/test", physics_loss, mstep)
        writer.add_scalar("loss_zfstep/zphysics/test", physics_loss, fstep)
        
        # get L2RE over ref data or copy L2RE from test data
        if hasattr(c.P, "ref_data"):
            ref_x_torch = torch.tensor(c.P.ref_x, device=x_test.device)
            yj_full2, _ = full_model_PINN(ref_x_torch, model, c)
            l2re = losses.l2_rel_err(torch.from_numpy(c.P.ref_y).to(self.device), yj_full2[0])
        else:
            l2re = losses.l2_rel_err(yj_true[0], yj_full[0])
        writer.add_scalar("loss_istep/l2re/test", l2re, i + 1)
        writer.add_scalar("loss_mstep/l2re/test", l2re, mstep)
        writer.add_scalar("loss_zfstep/l2re/test", l2re, fstep)

        # PLOTS
        
        if (i + 1) % c.TEST_FREQ == 0:
            
            # save figures
            fs = plot_main.plot_PINN(x_test, yj_true,   x, yj,   yj_full, y_full_raw,   yj_test_losses,   c, i + 1)
            if fs is not None: self._save_figs(i, fs)
            
        del x_test, yj_true,   x, yj,   yj_full, y_full_raw# fixes weird over-allocation of GPU memory bug caused by plotting (?)
        
        return yj_test_losses
                
    def train(self):
        "Train model"
        
        c, device, writer = self.c, self.device, self.writer
        
        # define model
        model = c.MODEL(c.P.d[0], c.P.d[1], c.N_HIDDEN, c.N_LAYERS)# problem-specific
        
        # create optimiser
        optimizer = torch.optim.Adam(model.parameters(), lr=c.LRATE)
        
        # put model on device
        model.to(device)
        
        # get normalisation values
        xmin, xmax = torch.tensor([[x.min(), x.max()] for x in c.SUBDOMAIN_XS], dtype=torch.float32, device=device).T
        mu = (xmin + xmax)/2; sd = (xmax - xmin)/2
        print(mu, sd, mu.shape, sd.shape)# (nd)# broadcast below
        
        # get exact solution if it exists
        x_test = _x_mesh(c.SUBDOMAIN_XS, c.BATCH_SIZE_TEST, device)
        yj_true = c.P.exact_solution(x_test, c.BATCH_SIZE_TEST)# problem-specific
        
        ## TRAIN
        
        mstep, fstep, yj_test_losses = 0, 0, []
        start, gpu_time = time.time(), 0.
        for i in range(c.N_STEPS):
            #print("STEP NUMBER "+str(i),flush=True)
            gpu_start = time.time()
            x, yj, loss = self._train_step(model, optimizer, c, i, mu, sd, device)
            mstep += model.size# record number of weights updated
            fstep += model.flops(x.shape[0])# record number of FLOPS
            gpu_time += time.time()-gpu_start
            
            
            # METRICS
            
            if (i + 1) % c.SUMMARY_FREQ == 0:
                
                # set counters
                rate, gpu_time = c.SUMMARY_FREQ / gpu_time, 0.
                
                # print summary
                self._print_summary(i, loss, rate, start)
                
                # test step
                yj_test_losses = self._test_step(x_test, yj_true,   x, yj,   model, c, i, mstep, fstep, writer, yj_test_losses)
            
            # SAVE
            
            if (i + 1) % c.MODEL_SAVE_FREQ == 0:
                
                # save model and losses
                self._save_model(i, model)
                np.save(c.MODEL_OUT_DIR+"loss_%.8i.npy"%(i + 1), np.array(yj_test_losses))
        
        # cleanup
        writer.close()
        print("Finished training")"""
    
    
if __name__ == "__main__":
    #'''
    c = Constants(
                  N_LAYERS=2,
                  N_HIDDEN=16,
                  TEST_FREQ=1000,
                  RANDOM=True,
                  )
    run = FBPINNTrainer(c)
    '''
    
    c = Constants(
                  N_LAYERS=4,
                  N_HIDDEN=64,
                  TEST_FREQ=1000,
                  RANDOM=True,
                  )
    run = PINNTrainer(c)
    '''
    
    run.train()