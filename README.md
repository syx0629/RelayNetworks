# RelayNetwork
Example code for training relay networks

Steps:
1. Install Dart-Env
2. Install Baselines


Training:

1) Hopper Example

mpirun -np 1 python -m train_relay --env=DartHopper-v1
 
Test : 
python test_policy.py 2

The argument here specifies the number of nodes in the relay. For example if there are 3 nodes in the relay chain, you would do
python test_policy.py 3
