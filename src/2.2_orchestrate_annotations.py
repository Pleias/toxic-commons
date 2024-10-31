import os, time
import pandas as pd

slurm_template = open("vllm_template.slurm", 'r').read()

start_job = 1
batch_size = 10000

#datasets = ['US-PD-Books', 'US-PD-Newspapers', 'French-PD-Books', 'French-PD-Newspapers', 'German-PD', 'German-PD-Newspapers', 'Italian-PD', 'Latin-PD', 'Polish-PD', 'Portuguese-PD', 'Spanish-PD-Books', 'Spanish-PD-Newspapers']

datasets = ['Dutch-PD', 'Portuguese-PD', 'Spanish-PD-Books', 'Spanish-PD-Newspapers']
for dataset in datasets:
    data = pd.read_csv(f"samples/{dataset}_samples.csv")
    end_job = len(data)  // batch_size
    for current_job in range(start_job, end_job + 1):
        start_batch = (current_job - 1) * batch_size

        suffix = f"""
        python3 create_annotations.py \
        --iteration {current_job} \
        --start {start_batch} \
        --batch_size {batch_size} \
        --dataset {dataset} 
        """ 

        slurm_command = slurm_template + suffix
        with open("temporary.slurm", 'w') as temp_slurm:
            temp_slurm.write(slurm_command)
            temp_slurm.write("\n nvidia-smi")
        time.sleep(1)
        os.system("sbatch temporary.slurm")
        print(f'Batch {current_job} has been submitted for dataset {dataset}!')
        time.sleep(10)
    time.sleep(1500)
