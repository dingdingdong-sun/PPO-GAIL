import math
from enum import Enum

class Machine_state(Enum):
    free = 1
    mix = 2
    busy = 3

class Machine(object):
    def __init__(self, id, processor, gpu, mem):
        self.id = id
        self.all_processor = int(processor)
        self.processor = int(processor)
        self.gpu = gpu
        self.mem = mem
        self.running_job_id_list = []
        self.state = Machine_state.free
        self.job_history = []
        self.now_proc = int(processor)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self,other):
        if isinstance(other, self.__class__):
            return hash(self.id) == hash(other.id)
        else:
            return False

    def taken_by_job(self, job, request_procs):
        if self.state.value != 3 and self.processor >= request_procs:
            self.running_job_id_list.append(job.job_id)
            self.processor -= request_procs
            self.now_proc = self.processor
            assert self.processor >= 0
            if self.processor:
                self.state = Machine_state.mix
            else:
                self.state = Machine_state.busy
            self.job_history.append(job.job_id)
            return True
        else:
            return False

    def release(self, release_num_procs, job_id):
        if self.state.value == 1:
            return -1
        elif release_num_procs + self.processor == self.all_processor:
            self.state = Machine_state.free
            self.running_job_id_list.remove(job_id)
            self.processor = self.all_processor
        else:
            self.state = Machine_state.mix
            self.running_job_id_list.remove(job_id)
            self.processor += release_num_procs
        self.now_proc = self.processor
        return 1

    def reset(self):
        self.state = Machine_state.free
        self.processor = self.all_processor
        self.now_proc = self.processor
        self.running_job_id_list = []
        self.job_history = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return "M["+str(self.id)+"] "


class Cluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node, num_mem_per_node):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.num_mem_per_node = num_mem_per_node
        self.busy_node = 0
        self.mix_node = 0
        self.num_procs_per_node = int(num_procs_per_node)
        self.all_free_procs = num_procs_per_node * node_num
        self.node_list = []
        self.all_nodes = {}

        for i in range(self.num_procs_per_node):
            self.all_nodes[i] = []

        machine_list = []
        for i in range(self.total_node):
            m = Machine(i, self.num_procs_per_node, 0, num_mem_per_node)
            self.node_list.append(m)
            machine_list.append(m)
        self.all_nodes[self.num_procs_per_node] = machine_list[:]

    def feature(self):
        return [self.free_node]

    def can_allocated(self, job):
        request_procs_per_node = int(
            math.ceil(float(job.request_number_of_processors) / float(job.request_number_of_nodes)))
        count = 0
        if job.request_number_of_nodes > self.free_node + self.mix_node:
            # print(job.request_number_of_processors)
            # print(job.request_number_of_nodes)
            return False
        else:
            for key in self.all_nodes.keys():
                if key >= request_procs_per_node:
                    count += len(self.all_nodes[key])
        if count >= job.request_number_of_nodes:
            return True
        else:
            # print(job.request_number_of_processors)
            # print(job.request_number_of_nodes)
            # print()
            # # print(self.all_nodes)
            # print("\n")
            return False


        # job.request_number_of_nodes = request_node
        # if request_node > self.free_node:
        #     return False
        # else:
        #     return True

    def allocate(self, job):
        allocated_machines = {}
        # request_node = int(math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))
        request_procs = int(math.ceil(float(job.request_number_of_processors) / float(job.request_number_of_nodes)))

        if not self.can_allocated(job):
            return []

        allocated_num_procs = 0

        for key in range(request_procs, self.num_procs_per_node+1):
            need_remove = []
            for m in self.all_nodes[key]:
                if allocated_num_procs == job.request_number_of_processors:
                    break
                elif job.request_number_of_processors-allocated_num_procs < request_procs:
                    request_procs = job.request_number_of_processors-allocated_num_procs
                if m.taken_by_job(job, request_procs):
                    if m.state.value == 3:
                        self.busy_node += 1
                        if request_procs < m.all_processor:
                            self.mix_node -= 1
                        else:
                            self.free_node -= 1
                        need_remove.append(m)
                        self.all_nodes[0].append(m)
                    else:
                        self.mix_node += 1
                        if request_procs < m.all_processor - m.processor:
                            self.mix_node -= 1
                        else:
                            self.free_node -= 1
                        need_remove.append(m)
                        self.all_nodes[m.processor].append(m)
                    # self.free_node -= 1
                    allocated_num_procs += request_procs
                    allocated_machines[m] = request_procs
                    # allocated_nodes[m] = request_procs
            for m in need_remove:
                self.all_nodes[key].remove(m)
        if allocated_num_procs == job.request_number_of_processors:
            return allocated_machines
        # if allocated == request_node:
        #     return allocated_nodes
        print(job.request_number_of_processors-allocated_num_procs)
        print("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases, job_id):
        # self.used_node -= len(releases)
        # self.free_node += len(releases)

        for m in releases.keys():
            if releases[m] + m.processor == m.all_processor:
                if m.state.value == 2:
                    self.mix_node -= 1
                    self.free_node += 1
                elif m.state.value == 3:
                    self.busy_node -= 1
                    self.free_node += 1
            elif m.state.value == 3:
                self.busy_node -= 1
                self.mix_node += 1
            self.all_nodes[m.processor].remove(m)
            self.all_nodes[releases[m] + m.processor].append(m)
            m.release(releases[m], job_id)

    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.busy_node = 0
        self.mix_node = 0
        self.free_node = self.total_node
        machine_list = []
        for key in self.all_nodes.keys():
            for m in self.all_nodes[key]:
                m.reset()
                machine_list.append(m)

        for i in range(self.num_procs_per_node+1):
            self.all_nodes[i] = []

        for m in machine_list:
            self.all_nodes[self.num_procs_per_node].append(m)

class FakeList:
    def __init__(self, l):
        self.len = l
    def __len__(self):
        return self.len

class SimpleCluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []

    def feature(self):
        return [self.free_node]

    def can_allocated(self, job):
        if job.request_number_of_nodes != -1:
            if job.request_number_of_nodes > self.free_node:
                return False
            else:
                return True

        request_node = int(math.ceil(float(job.request_number_of_processors)/float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            return True

    def allocate(self, job_id, request_num_procs):
        allocated_nodes = FakeList(0)
        request_node = int(math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = request_node

        self.used_node += allocated
        self.free_node -= allocated
        allocated_nodes.len = allocated
        if allocated == request_node:
            return allocated_nodes

        print ("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)


    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node