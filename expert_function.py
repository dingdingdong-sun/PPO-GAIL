import math

def judge_fn(func_type, job_old,job_new):
    old_submit_time = job_old.submit_time
    old_runtime = job_old.run_time
    old_procs = job_old.request_number_of_processors
    new_submit_time = job_new.submit_time
    new_runtime = job_new.run_time
    new_procs = job_new.request_number_of_processors
    if func_type == 0: #HPC2N
        old_score = math.log10(old_runtime) * old_procs + 40 * math.log10(old_submit_time)
        new_score = math.log10(new_runtime) * new_procs + 40 * math.log10(new_submit_time)
        if old_score < new_score:
            return False
    if func_type == 5:  # SDSC-BLUE-2000
        old_score = 53 * math.sqrt(old_runtime) * 3.4 * 1e-5 * old_procs + 79 * math.log10(old_submit_time)
        new_score = 53 * math.sqrt(new_runtime) * 3.4 * 1e-5 * new_procs + 79 * math.log10(new_submit_time)
        if old_score < new_score:
            return False
    if func_type == 6:  # CTC-SP2-1996
        old_score = math.sqrt(old_runtime) * old_procs + 2560 * math.log10(old_submit_time)
        new_score = math.sqrt(new_runtime) * new_procs + 2560 * math.log10(new_submit_time)
        if old_score < new_score:
            return False
    return True


def test_fn(func_type, job_old,job_new, current_time):
    old_submit_time = job_old.submit_time
    old_runtime = job_old.run_time
    old_waittime = current_time - old_submit_time
    old_procs = job_old.request_number_of_processors
    new_submit_time = job_new.submit_time
    new_runtime = job_new.run_time
    new_waittime = current_time - new_submit_time
    new_procs = job_new.request_number_of_processors
    if func_type == 0: #FCFS
        if old_submit_time <= new_submit_time:
            return False
    elif func_type == 1: #SJF
        if old_runtime <= new_runtime:
            return False
    elif func_type == 2: #WFP3
        old_score = -((old_waittime/ old_runtime) ** 3) * old_procs
        new_score = -((new_waittime/ new_runtime) ** 3) * new_procs
        if old_score < new_score:
            return False
    elif func_type == 3: #UNICEP
        old_score = - old_waittime / (math.log2(old_procs) * old_runtime + 1e-5)
        new_score = - new_waittime / (math.log2(new_procs) * new_runtime + 1e-5)
        if old_score < new_score:
            return False
    if func_type == 4: #HPC2N
        old_score = 53 * math.sqrt(old_runtime) * 3.4 * 1e-5 * old_procs + 79 * math.log10(old_submit_time)
        new_score = 53 * math.sqrt(new_runtime) * 3.4 * 1e-5 * new_procs + 79 * math.log10(new_submit_time)
        if old_score < new_score:
            return False
    if func_type == 5: #SDSC-BLUE-2000
        old_score = math.sqrt(old_runtime) * old_procs + 2560 * math.log10(old_submit_time)
        new_score = math.sqrt(new_runtime) * new_procs + 2560 * math.log10(new_submit_time)
        if old_score < new_score:
            return False
    if func_type == 6: # CTC-SP2-1996
        old_score = math.log10(old_runtime) * old_procs + 40 * math.log10(old_submit_time)
        new_score = math.log10(new_runtime) * new_procs + 40 * math.log10(new_submit_time)
        if old_score < new_score:
            return False
    return True