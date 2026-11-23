"""Balanced schedule variant of the preserved K_Arm_Scanner.scanning.

The inherited initialization, CE + cost*L1, Adam and cost schedule are unchanged.
Only allocation/global stopping and measurement callbacks differ. Batch190 only.
Original scanner remains untouched for the baseline and historical runs.
"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json
import time
from K_Arm_Scanner import K_Arm_Scanner


class BalancedScanner(K_Arm_Scanner):
    def scanning(
        self,
        target_classes_all,
        data_loader_arr,
        y_target_index,
        pattern_init,
        mask_init,
        trigger_type,
        direction,
    ):
        self.reset_state(pattern_init, mask_init)
        # for TrojAI round1, the data format is BGR, then need permute
        # permute = [2,1,0]

        self.update_tensor(
            self.mask_tanh_tensor, self.pattern_tanh_tensor, y_target_index, True
        )

        # K-arms bandits version
        best_mask = [None] * self.num_classes
        best_pattern = [None] * self.num_classes
        best_reg = [1e10] * self.num_classes

        best_acc = [0] * self.num_classes

        log = []
        cost_set_counter = [0] * self.num_classes
        cost_down_counter = [0] * self.num_classes
        cost_up_counter = [0] * self.num_classes
        cost_up_flag = [False] * self.num_classes
        cost_down_flag = [False] * self.num_classes
        early_stop_counter = [0] * self.num_classes
        early_stop_reg_best = [1e10] * self.num_classes
        early_stop_tag = [False] * self.num_classes
        update = [False] * self.num_classes

        avg_loss_ce = [1e10] * self.num_classes
        avg_loss_reg = [1e10] * self.num_classes
        avg_loss = [1e10] * self.num_classes
        avg_loss_acc = [1e10] * self.num_classes
        reg_down_vel = [-1e10] * self.num_classes
        times = [0] * self.num_classes
        total_times = [0] * self.num_classes
        first_best_reg = [1e10] * self.num_classes
        y_target_tensor = (
            torch.Tensor([target_classes_all[y_target_index]]).long().to(self.device)
        )

        optimizer_list = []

        for i in range(self.num_classes):
            optimizer = optim.Adam(
                [self.pattern_tanh_tensor[i], self.mask_tanh_tensor[i]],
                lr=self.lr,
                betas=(0.5, 0.9),
            )
            optimizer_list.append(optimizer)

        criterion = nn.CrossEntropyLoss()

        self.early_stop = False
        assert all(len(loader) == 1 for loader in data_loader_arr)
        pbar = tqdm(range(self.steps))
        timing_started = time.monotonic()
        observed_minibatches = 0

        # for step in range(self.steps):

        for step in pbar:

            y_target_tensor = (
                torch.Tensor([target_classes_all[y_target_index]])
                .long()
                .to(self.device)
            )
            total_times[y_target_index] += 1

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for idx, (img, name, label) in enumerate(data_loader_arr[y_target_index]):
                # Optional budgeted adapter correction: metrics below describe
                # this pre-update state, not the state after optimizer.step().
                if getattr(self, "capture_preupdate", False):
                    measured_mask = self.mask_tensor[y_target_index].detach().clone()
                    measured_pattern = (
                        self.pattern_raw_tensor[y_target_index].detach().clone()
                    )
                img = img.to(self.device)
                Y_target = y_target_tensor.repeat(img.size()[0])

                X_adv_tensor = (
                    1 - self.mask_tensor[y_target_index]
                ) * img + self.mask_tensor[y_target_index] * self.pattern_raw_tensor[
                    y_target_index
                ]

                optimizer_list[y_target_index].zero_grad()

                output_tensor = self.model(X_adv_tensor)

                pred = output_tensor.argmax(dim=1, keepdim=True)

                self.loss_acc = pred.eq(Y_target.long().view_as(pred)).sum().item() / (
                    img.size()[0]
                )
                self.loss_ce = criterion(output_tensor, Y_target)

                self.loss_reg = torch.sum(torch.abs(self.mask_tensor[y_target_index]))

                self.loss = (
                    self.loss_ce + self.loss_reg * self.cost_tensor[y_target_index]
                )

                self.loss.backward()

                optimizer_list[y_target_index].step()
                observed_minibatches += 1
                self.update_tensor(
                    self.mask_tanh_tensor, self.pattern_tanh_tensor, y_target_index
                )

                pbar.set_description(
                    "Target: {}, victim: {}, Loss: {:.4f}, Acc: {:.2f}%, CE_Loss: {:.2f}, Reg_Loss:{:.2f}, Cost:{:.2f} best_reg:{:.2f} avg_loss_reg:{:.2f}".format(
                        target_classes_all[y_target_index],
                        label[0],
                        self.loss,
                        self.loss_acc * 100,
                        self.loss_ce,
                        self.loss_reg,
                        self.cost_tensor[y_target_index],
                        best_reg[y_target_index],
                        avg_loss_reg[y_target_index],
                    )
                )
                loss_ce_list.append(self.loss_ce.item())
                loss_reg_list.append(self.loss_reg.item())
                loss_list.append(self.loss.item())
                loss_acc_list.append(self.loss_acc)

            if step == 0 or (step + 1) % 25 == 0:
                elapsed = time.monotonic() - timing_started
                print(
                    json.dumps(
                        dict(
                            event="scan_progress",
                            direction=direction,
                            kind=trigger_type,
                            completed_steps=step + 1,
                            step_budget=self.steps,
                            observed_minibatches=observed_minibatches,
                            elapsed_seconds=elapsed,
                            remaining_scan_seconds_linear=elapsed
                            / (step + 1)
                            * (self.steps - step - 1),
                            peak_cuda_allocated_bytes=(
                                torch.cuda.max_memory_allocated()
                                if str(self.device).startswith("cuda")
                                else None
                            ),
                            observed_unix=time.time(),
                        )
                    ),
                    flush=True,
                )
            # K-arms Bandits
            avg_loss_ce[y_target_index] = np.mean(loss_ce_list)
            avg_loss_reg[y_target_index] = np.mean(loss_reg_list)
            avg_loss[y_target_index] = np.mean(loss_list)
            avg_loss_acc[y_target_index] = np.mean(loss_acc_list)
            assert np.isfinite(avg_loss[y_target_index]), "Nonfinite optimization loss"

            if avg_loss_acc[y_target_index] > best_acc[y_target_index]:
                best_acc[y_target_index] = avg_loss_acc[y_target_index]

            if direction == "forward":

                if (
                    (
                        total_times[y_target_index] > 20
                        and best_acc[y_target_index] < 0.3
                        and trigger_type == "polygon_specific"
                    )
                    or (
                        total_times[y_target_index] > 200
                        and best_acc[y_target_index] < 0.8
                        and trigger_type == "polygon_specific"
                    )
                    or (
                        total_times[y_target_index] > 10
                        and best_acc[y_target_index] == 0
                        and trigger_type == "polygon_specific"
                    )
                ):

                    early_stop_tag[y_target_index] = True

            elif direction == "backward":
                if (
                    total_times[y_target_index] > 200
                    and best_acc[y_target_index] < 1
                    and trigger_type == "polygon_specific"
                ):
                    # for the backward check
                    early_stop_tag[y_target_index] = True

            update[y_target_index] = False
            if (
                avg_loss_acc[y_target_index] >= self.attack_succ_threshold
                and avg_loss_reg[y_target_index] < best_reg[y_target_index]
            ):
                best_mask[y_target_index] = self.mask_tensor[y_target_index]
                if getattr(self, "capture_preupdate", False):
                    assert len(data_loader_arr[y_target_index]) == 1
                    best_mask[y_target_index] = measured_mask

                # print('best_mask update')
                update[y_target_index] = True
                times[y_target_index] += 1
                best_pattern[y_target_index] = self.pattern_raw_tensor[y_target_index]
                if getattr(self, "capture_preupdate", False):
                    best_pattern[y_target_index] = measured_pattern

                if times[y_target_index] == 1:
                    first_best_reg[y_target_index] = 2500
                    # self.cost_tensor[y_target_index] = 1e-3
                # reg_down_vel[y_target_index] = 1e+4 * (np.log10(first_best_reg[y_target_index]) - np.log10(avg_loss_reg[y_target_index])) / (times[y_target_index] + total_times[y_target_index] / 10)
                reg_down_vel[y_target_index] = (
                    (first_best_reg[y_target_index]) - (avg_loss_reg[y_target_index])
                ) / (times[y_target_index] + (total_times[y_target_index] / 2))
                # print('best_reg:',best_reg[y_target_index])
                # print('avg_loss_reg:',avg_loss_reg[y_target_index])

                best_reg[y_target_index] = avg_loss_reg[y_target_index]

            if self.early_stop:

                if best_reg[y_target_index] < 1e10:
                    if (
                        best_reg[y_target_index]
                        >= self.early_stop_threshold
                        * early_stop_reg_best[y_target_index]
                    ):
                        # print('best_reg:',best_reg[y_target_index])
                        # print('early_stop_best_reg:',early_stop_reg_best[y_target_index])
                        early_stop_counter[y_target_index] += 1
                    else:
                        early_stop_counter[y_target_index] = 0
                early_stop_reg_best[y_target_index] = min(
                    best_reg[y_target_index], early_stop_reg_best[y_target_index]
                )

                if (times[y_target_index] > self.round) or (
                    cost_down_flag[y_target_index]
                    and cost_up_flag[y_target_index]
                    and early_stop_counter[y_target_index] > self.early_stop_patience
                    and trigger_type == "polygon_global"
                ):

                    if y_target_index == torch.argmin(torch.Tensor(best_reg)):
                        print("early stop for all!")
                        break
                    else:
                        early_stop_tag[y_target_index] = True

                        if all(ele == True for ele in early_stop_tag):
                            break

            if early_stop_tag[y_target_index] == False:

                if (
                    self.cost[y_target_index] == 0
                    and avg_loss_acc[y_target_index] >= self.attack_succ_threshold
                ):
                    cost_set_counter[y_target_index] += 1
                    if cost_set_counter[y_target_index] >= 2:
                        self.cost[y_target_index] = self.init_cost[y_target_index]
                        self.cost_tensor[y_target_index] = self.cost[y_target_index]
                        cost_up_counter[y_target_index] = 0
                        cost_down_counter[y_target_index] = 0
                        cost_up_flag[y_target_index] = False
                        cost_down_flag[y_target_index] = False
                else:
                    cost_set_counter[y_target_index] = 0

                if avg_loss_acc[y_target_index] >= self.attack_succ_threshold:
                    cost_up_counter[y_target_index] += 1
                    cost_down_counter[y_target_index] = 0
                else:
                    cost_up_counter[y_target_index] = 0
                    cost_down_counter[y_target_index] += 1

                if cost_up_counter[y_target_index] >= self.patience:
                    cost_up_counter[y_target_index] = 0
                    self.cost[y_target_index] *= self.cost_multiplier_up
                    self.cost_tensor[y_target_index] = self.cost[y_target_index]
                    cost_up_flag[y_target_index] = True
                elif cost_down_counter[y_target_index] >= self.patience:
                    cost_down_counter[y_target_index] = 0
                    self.cost[y_target_index] /= self.cost_multiplier_down
                    self.cost_tensor[y_target_index] = self.cost[y_target_index]
                    cost_down_flag[y_target_index] = True

            # Fixed round-robin allocation; original objective/cost update above.
            if total_times[y_target_index] in self.measurements:
                self.measurement_callback(
                    y_target_index,
                    total_times[y_target_index],
                    best_mask[y_target_index],
                    best_pattern[y_target_index],
                    dict(
                        best_reg=best_reg[y_target_index],
                        best_acc=best_acc[y_target_index],
                        successful_updates=times[y_target_index],
                        cost=self.cost[y_target_index],
                    ),
                )
            y_target_index = (y_target_index + 1) % self.num_classes
            continue

        self.diagnostics = {
            "best_batch_asr": best_acc,
            "last_batch_asr": avg_loss_acc,
            "early_stop": early_stop_tag,
            "successful_updates": times,
        }
        return best_pattern, best_mask, best_reg, total_times
