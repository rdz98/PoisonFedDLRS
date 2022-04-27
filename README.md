# PoisonFedDLRS

This is the pytorch implementation for our IJCAI 2022 paper:

> Rong, Dazhong, et al. "Poisoning Deep Learning based Recommender Model in Federated Learning Scenarios." Proceedings of the 31st International Joint Conference on Artificial Intelligence. 2022.

## Environment
+ Python 3.7.9
+ numpy==1.18.5
+ torch==1.7.0+cu101


## Usage

To run attack with hard user mining (A-hum) on MovieLens (ML) dataset with ![](http://latex.codecogs.com/svg.latex?\rho=0.1\%):

`python main.py --dataset=ML --attack=A-hum --clients_limit=0.001`

There are two choices on dataset:

`--dataset=ML` and `--dataset=AZ`.

There are four choices on attack:

`--attack=A-ra`, `--attack=A-hum`, `--attack=EB` and `--attack=RA`.

## Output
```
Arguments: attack=A-hum,dim=8,layers=[8,8],num_neg=4,path=Data/,dataset=ML,device=cuda,lr=0.001,epochs=30,batch_size=256,items_limit=30,clients_limit=0.001 
Load data done [30.0 s]. #user=6046, #item=3706, #train=802552, #test=197656
Target items: [1805].
output format: ({HR@20, Prec@20, NDCG@20}), ({ER@5, ER@10, ER@20, ER@30})
Iteration 0(init), (0.0053093, 0.0096192, 0.0097856) on test, (0.0000000, 0.0000000, 0.0000000, 0.0000000) on target. [6.3s]
Iteration 1, loss = 0.52631 [49.6s], (0.0036713, 0.0076904, 0.0066621) on test, (0.0000000, 0.0000000, 0.0000000, 0.0000000) on target. [5.8s]
Iteration 2, loss = 0.50028 [53.1s], (0.0059283, 0.0110265, 0.0092476) on test, (0.0000000, 0.0000000, 0.0000000, 0.0000000) on target. [5.8s]
Iteration 3, loss = 0.50015 [51.3s], (0.0112714, 0.0192136, 0.0164828) on test, (0.0000000, 0.0000000, 0.0000000, 0.0000000) on target. [5.8s]
Iteration 4, loss = 0.49986 [52.2s], (0.0169052, 0.0246937, 0.0230123) on test, (0.0000000, 0.0000000, 0.0000000, 0.0000000) on target. [7.1s]
Iteration 5, loss = 0.49907 [56.5s], (0.0115334, 0.0184520, 0.0183481) on test, (0.0000000, 0.0000000, 0.0000000, 0.0000000) on target. [9.5s]
Iteration 6, loss = 0.49804 [58.2s], (0.0132934, 0.0234603, 0.0242366) on test, (0.0000000, 0.0000000, 0.0000000, 0.0000000) on target. [16.6s]
Iteration 7, loss = 0.49698 [59.3s], (0.0156124, 0.0280381, 0.0281607) on test, (0.0000000, 0.0024839, 0.0337804, 0.1097864) on target. [18.2s]
Iteration 8, loss = 0.49579 [59.7s], (0.0171802, 0.0305215, 0.0313257) on test, (0.0236794, 0.0832919, 0.3061765, 0.7481371) on target. [16.1s]
Iteration 9, loss = 0.49438 [59.6s], (0.0182686, 0.0317964, 0.0326551) on test, (0.0748468, 0.2204007, 0.7771154, 1.0000000) on target. [17.6s]
Iteration 10, loss = 0.49233 [59.2s], (0.0214253, 0.0364735, 0.0363936) on test, (0.1872827, 0.4744163, 1.0000000, 1.0000000) on target. [17.3s]
... ...
```

## License
The codes are for learning and research purposes only.

