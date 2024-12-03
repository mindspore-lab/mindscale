# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""sapp pipeline"""

import os
import sys
import yaml

from toolkit.pipeline_balance.utils.logger import logger
import toolkit.pipeline_balance.simulator.pp_simulator as sim
from toolkit.pipeline_balance.sapp.sapp_solver import SappSolver
from toolkit.pipeline_balance.utils.layer import Layer, filter_layer_type
import toolkit.pipeline_balance.utils.recompute as Recompute


class SappPipeline:
    """pipeline balancer"""

    def __init__(
            self,
            model_name: str,
            num_of_stage: int,
            num_of_micro_batch: int,
            max_memory: int,
            layers: list[Layer],
            vpp_less_memory: bool = False,
            num_of_interleave: int = 1,
            constant_memory: int = 0,
            optimization_level: int = 1,
    ):
        self.model_name_ = model_name
        self.num_of_stage_ = num_of_stage
        self.num_of_micro_batch_ = num_of_micro_batch
        self.num_of_interleave_ = num_of_interleave
        self.max_memory_ = max_memory
        self.vpp_less_memory_ = vpp_less_memory
        self.constant_memory_ = constant_memory
        self.optimization_level = optimization_level

        self.problem_ = None
        self.layers_ = layers
        self.layers_sorted_ = {
            Layer.type_enum.HEAD: filter_layer_type(layers,
                                                    Layer.type_enum.HEAD),
            Layer.type_enum.BODY: filter_layer_type(layers,
                                                    Layer.type_enum.BODY),
            Layer.type_enum.TAIL: filter_layer_type(layers,
                                                    Layer.type_enum.TAIL),
        }

    def has_some_memory_info(self) -> bool:
        """Check if there is all information for memory constraint."""
        return self.problem_.has_some_memory_info()

    def construct_problem(self, solver: str = "pulp"):
        """Construct the problem to solve, chose the solver."""
        if solver == "pulp":
            self.problem_ = self._construct_problem_pulp_()
        elif solver == "other":
            logger.warning(
                "No other solver available..., automatically switch to pulp!!!"
            )
            self.problem_ = self._construct_problem_pulp_()
        else:
            logger.warning(
                "No other solver available..., automatically switch to pulp!!!"
            )
            self.problem_ = self._construct_problem_pulp_()

    def solve_problem(self, time_limit=90, dump_folder=None):
        """Solve the problem to get the schedule pipeline."""
        self.problem_.solve(time_limit, dump_folder)

    def get_result(self) -> dict[str, list[list[str]]]:
        """Get result distribution of the solution (compact form)."""
        return self.problem_.result()

    def get_memory_activation(self) -> list[float]:
        """Get the activation memory per stage for simulator."""
        return self.problem_.get_simulator_memory_activation()

    def get_memory_parameter(self) -> list[float]:
        """Get the parameter memory per stage for simulator."""
        return self.problem_.get_simulator_memory_parameter()

    def get_fw_time(self) -> list[float]:
        """Get the forward time per stage for simulator."""
        time = self.problem_.get_simulator_forward_time()
        return time

    def get_recompute_time(self) -> list[float]:
        """Get the recompute time per stage for simulator."""
        time = self.problem_.get_simulator_recompute_time()
        return time

    def get_time(self) -> list[float]:
        """Get the time per stage for simulator."""
        return self.problem_.get_simulator_time()

    def naive_layer_per_stage(self,
                              layer_num: int,
                              num_of_interleave=1) -> list[list[int]]:
        """Get the even layer to stage assignment of LLM"""
        return [[layer_num // (self.num_of_stage_ * num_of_interleave)] *
                self.num_of_stage_] * num_of_interleave

    def print_yaml_results(self):
        """Print results"""

        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            nass = self.naive_layer_per_stage(layer.nb_layer_,
                                              self.num_of_interleave_)
            yaml_format = Recompute.yaml_from_internal(
                self.num_of_interleave_,
                self.num_of_stage_,
                self.problem_.variables_[layer.name_],
                nass,
            )
            logger.output(f"layer-to-stage assignment baseline is \n\t{nass}")
            yaml_results = "\nTo put in yaml configuration:"
            for y, v in yaml_format.items():
                yaml_results += f"\n\t{y}: {v}"
            logger.output(yaml_results)

    def get_manual_memory_activation(self,
                                     layer_per_recompute,
                                     interleave_num=1) -> list[float]:
        """
        Give the activation memory per stage for manual layer assignment without
        interleave for simulator.
        """
        memory_active = []
        unused_rec = Recompute.get_unused_list(layer_per_recompute)
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            if self.has_some_memory_info():
                for inter in range(interleave_num):
                    memory_active.append([])
                    for stage in range(self.num_of_stage_):
                        memory_active[inter].append(0)
                        memory_active[inter][stage] = sum(
                            layer_per_recompute[rec][inter][stage] *
                            layer.memory_activation_rec_[rec]
                            for rec in Recompute.TYPE if rec not in unused_rec
                            and layer_per_recompute[rec][inter][stage] > 0)
        return memory_active

    def get_manual_memory_parameter(self,
                                    layer_per_recompute,
                                    interleave_num=1) -> list[float]:
        """
        Give the parameter memory per stage for manual layer assignment
        without interleave for simulator.
        """
        unused_rec = Recompute.get_unused_list(layer_per_recompute)
        memory_param_stage = [0] * self.num_of_stage_
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            if layer.memory_parameter_ is not None:
                for inter in range(interleave_num):
                    for stage in range(self.num_of_stage_):
                        memory_param_stage[stage] += sum(
                            layer_per_recompute[rec][inter][stage] *
                            layer.memory_parameter_ for rec in Recompute.TYPE
                            if rec not in unused_rec
                            and layer_per_recompute[rec][inter][stage] > 0)
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            if head.memory_parameter_ is not None:
                memory_param_stage[0] += head.memory_parameter_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            if tail.memory_parameter_ is not None:
                memory_param_stage[self.num_of_stage_ -
                                   1] += tail.memory_parameter_
        memory_param = [memory_param_stage] * interleave_num
        return memory_param

    def get_manual_time(self,
                        layer_per_recompute,
                        interleave_num=1) -> list[float]:
        """
        Get the time per stage for a naive layer assignment
        without interleave for simulator.
        """
        time = []
        rec = Recompute.TYPE.NONE
        for i in range(interleave_num):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if layer_per_recompute[rec][i][s] > 0:
                            time[i][s] += layer_per_recompute[rec][i][s] * (
                                layer.forward_time_ +
                                layer.backward_time_rec_[rec])

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[interleave_num - 1][self.num_of_stage_ - 1] += tail.time_
        return time

    def get_manual_fw_time(self,
                           layer_per_recompute,
                           interleave_num=1) -> list[float]:
        """
        Give the time per stage for a naive layer assignment
        without interleave for simulator.
        """
        time = []
        rec = Recompute.TYPE.NONE
        unused_rec = Recompute.get_unused_list(layer_per_recompute)
        for i in range(interleave_num):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if rec not in unused_rec and layer_per_recompute[rec][
                                i][s] > 0:
                            time[i][s] += layer_per_recompute[rec][i][s] * (
                                layer.forward_time_)
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_ / 3
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[interleave_num - 1][self.num_of_stage_ - 1] += tail.time_ / 3
        return time

    def get_manual_recompute_time(self,
                                  layer_per_recompute,
                                  interleave_num=1) -> list[float]:
        """
        Give the time per stage for a manual layer assignment
        without interleave for simulator.
        """
        time_all_rec = []
        time_no_rec = []
        unused_rec = Recompute.get_unused_list(layer_per_recompute)
        for i in range(interleave_num):
            time_all_rec.append([])
            time_no_rec.append([])
            for s in range(self.num_of_stage_):
                time_all_rec[i].append(0)
                time_no_rec[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if rec not in unused_rec and layer_per_recompute[rec][
                                i][s] > 0:
                            time_all_rec[i][s] += layer_per_recompute[rec][i][
                                s] * (layer.backward_time_rec_[rec])
                            time_no_rec[i][s] += layer_per_recompute[rec][i][
                                s] * (layer.backward_time_rec_[
                                    Recompute.TYPE.NONE])

        return [[r - n for r, n in zip(ar, nr)]
                for ar, nr in zip(time_all_rec, time_no_rec)]

    def simulate(self, show=True, file_name=None):
        """Use simulator to visualize output."""
        forward_time = self.get_fw_time()
        recompute_overhead = self.get_recompute_time()
        stage_mem_par = 0
        stage_mem_act = 0
        if self.has_some_memory_info():
            stage_mem_par = self.get_memory_parameter()
            stage_mem_act = self.get_memory_activation()

        return self.simulation(
            forward_time,
            recompute_overhead,
            stage_mem_par,
            stage_mem_act,
            self.constant_memory_,
            show,
            file_name,
        )

    def simulate_naive(self, layers, output_folder):
        """simulate naive configs"""
        num_layers = 0
        for layer in layers:
            if layer.type_ == Layer.type_enum.BODY:
                num_layers = layer.nb_layer_

        self.simulate_yaml(
            yaml_format={
                "offset": 0,
                "recompute": True
            },
            show=True,
            interleave_num=self.num_of_interleave_,
            file_name=os.path.join(output_folder,
                                   "result_naive_all_recomp.svg"),
        )

        if num_layers % self.num_of_stage_ == 0:
            self.simulate_yaml(
                yaml_format={
                    "offset": 0,
                    "recompute": False
                },
                show=True,
                interleave_num=self.num_of_interleave_,
                file_name=os.path.join(output_folder,
                                       "result_naive_no_recomp.svg"),
            )
        else:
            logger.warning("num layer cannot be divided by num stage")

    def simulate_file(self, manual_config_file, output_folder):
        """simulate manual input config"""
        with open(manual_config_file, encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        yaml_data = {}
        for manual in data.values():
            yaml_data[Recompute.OFFSET] = manual.get(Recompute.OFFSET)
            for rec in Recompute.YAML_NAME.values():
                yaml_data[rec] = manual.get(rec)
                if isinstance(yaml_data[rec], list) and all(
                        isinstance(item, int) for item in yaml_data[rec]):
                    yaml_data[rec] = [yaml_data[rec]]
            interleave_num = manual.get("interleave_num",
                                        self.num_of_interleave_)
            show = manual.get("show", False)
            file_name = manual.get("file_name")
            file_name = os.path.join(output_folder,
                                     file_name) if (file_name) else None
            self.simulate_yaml(yaml_data, show, interleave_num, file_name)

    def simulate_yaml(self, yaml_format, show=True, interleave_num=1, file_name=None):
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            layer_num = layer.nb_layer_
        nass = self.naive_layer_per_stage(layer_num,
                                          num_of_interleave=interleave_num)
        layer_per_recompute = Recompute.internal_from_yaml(
            interleave_num, self.num_of_stage_, yaml_format, nass)
        return self.simulate_manual(
            layer_per_recompute,
            show,
            interleave_num=interleave_num,
            file_name=file_name,
        )

    def simulate_manual(self,
                        layer_per_recompute=None,
                        show=True,
                        interleave_num=1,
                        file_name=None):
        """Use simulator to visualize output."""
        for layer in layer_per_recompute.values():
            if len(layer) != interleave_num:
                logger.error(
                    f"number of list {len(layer)} does not match interleave number {interleave_num}"
                )
                return sys.maxsize

        for rec in Recompute.TYPE:
            if any(x < 0 for sublist in layer_per_recompute[rec]
                   for x in sublist):
                raise ValueError(
                    f"in {Recompute.YAML_NAME[rec]}, there is strategy less than 0"
                )

        forward_time = self.get_manual_fw_time(layer_per_recompute,
                                               interleave_num)
        recompute_overhead = self.get_manual_recompute_time(
            layer_per_recompute, interleave_num)
        stage_mem_par = 0
        stage_mem_act = 0
        if self.has_some_memory_info():
            stage_mem_par = self.get_manual_memory_parameter(
                layer_per_recompute, interleave_num=interleave_num)
            stage_mem_act = self.get_manual_memory_activation(
                layer_per_recompute, interleave_num=interleave_num)
        return self.simulation(
            forward_time,
            recompute_overhead,
            stage_mem_par,
            stage_mem_act,
            self.constant_memory_,
            show,
            file_name,
        )

    def simulation(
            self,
            forward_time,
            recompute_overhead=0,
            stage_mem_par=0,
            stage_mem_act=0,
            constant_mem=0,
            show=True,
            file_name=None,
    ):
        """Use simulator to visualize output."""
        if self.has_some_memory_info():
            logger.output(
                f"PipelineSimulator(\n\t{forward_time}, {self.num_of_micro_batch_},"
                f"\n\tblock_mem_act={stage_mem_act},"
                f"\n\tblock_mem_par={stage_mem_par},"
                f"\n\tlayer_recompute={recompute_overhead},"
                f"\n\tless_memory={self.vpp_less_memory_} )")

            sim_method = "vpp2" if self.vpp_less_memory_ else "vpp"
            simulator = sim.PipelineSimulator(
                forward_time,
                self.num_of_micro_batch_,
                block_mem=stage_mem_act,
                block_mem_par=stage_mem_par,
                constant_mem=constant_mem,
                layer_recompute=recompute_overhead,
                method=sim_method,
            )
        else:
            logger.output(
                f"PipelineSimulator(\n\t{forward_time}, {self.num_of_micro_batch_},"
                f"\n\tlayer_recompute={recompute_overhead})"
                f"\n\tless_memory={self.vpp_less_memory_} )")
            simulator = sim.PipelineSimulator(
                forward_time,
                self.num_of_micro_batch_,
                layer_recompute=recompute_overhead,
                less_memory=self.vpp_less_memory_,
            )

        simulator.run(comm=False)
        if show:
            simulator.show(file_name=file_name)
        return simulator.end_time

    def _construct_problem_pulp_(self) -> SappSolver:
        """construct the problem using pulp"""
        prob = SappSolver(
            num_of_stage=self.num_of_stage_,
            num_of_micro_batch=self.num_of_micro_batch_,
            num_of_interleave=self.num_of_interleave_,
            max_memory=self.max_memory_,
            vpp_less_memory=self.vpp_less_memory_,
            constant_memory=self.constant_memory_,
            layers=self.layers_,
            layers_sorted=self.layers_sorted_,
            optimization_level=self.optimization_level,
        )
        return prob

    def _recompute_considered(self):
        return self.problem_.recompute_considered_


def choose_interleave(
        model_name: str,
        number_of_stage: int,
        number_of_micro_batch: int,
        max_memory: int,
        layers: list[Layer],
) -> tuple[int, int, dict[str, list[list[str]]]]:
    """Simulates different interleaves and returns the best."""
    max_inter = 4
    best_time = int(sys.maxsize)
    best_inter = 1
    best_distribution = {}

    for inter in range(1, max_inter + 1):
        pipe = SappPipeline(
            model_name=model_name,
            num_of_stage=number_of_stage,
            num_of_micro_batch=number_of_micro_batch,
            max_memory=max_memory,
            layers=layers,
            num_of_interleave=inter,
        )

        pipe.construct_problem(solver="pulp")
        pipe.solve_problem()
        time = pipe.simulate(show=False)
        logger.output(f"for interleave {inter}, time = {time}")
        if time < best_time:
            best_time = time
            best_inter = inter
            best_distribution = pipe.get_result()

    return (best_inter, best_time, best_distribution)


def flatten(inter_stage_list):
    """Flatten an interleave x stage list to a stage list"""
    stage_list = [0] * len(inter_stage_list[0])
    for inter, _ in enumerate(inter_stage_list):
        for stage, _ in enumerate(inter_stage_list[inter]):
            stage_list[stage] += inter_stage_list[inter][stage]
    return stage_list
