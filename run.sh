
code=src/run_BP.py
dataset=tsp30-50


# baseline
python $code $dataset

# damping
python $code $dataset -d 0.5

# alternative configuration decoding
python $code $dataset -alt 1

# considering 4 lowest messages
python $code $dataset -alt 1 -b 4

