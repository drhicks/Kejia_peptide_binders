#python ./pipeline.py -p gpu -t 10:00 --gres "gpu:rtx2080:1" --benchmarks rsv5-1 rsv0-1 --num_per_condition 20 --num_per_job 1 --out 20220914_pipeline_test/ --args "diffuser.T=100|200 inference.cautious=True" --start_step score
#python ./pipeline.py --benchmarks rsv5-1 --num_per_condition 2 --num_per_job 1 --out test3/run1 --args "diffuser.T=20|50"
python ./pipeline.py --benchmarks rsv5-1     --num_per_condition 3 --num_per_job 1 --out test1/     --args "diffuser.T=20|50 diffuser.aa_decode_steps=5|10"
