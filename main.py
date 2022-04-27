import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset
from client import FedRecClient
from server import FedRecServer
from attack import AttackClient, BaselineAttackClient


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    t0 = time()
    m_item, all_train_ind, all_test_ind, items_popularity = load_dataset(args.path + args.dataset)
    _, target_items = torch.Tensor(-items_popularity).topk(1)
    target_items = target_items.tolist()  # Select the least popular item as the target item

    server = FedRecServer(m_item, args.dim, eval(args.layers)).to(args.device)
    clients = []
    for train_ind, test_ind in zip(all_train_ind, all_test_ind):
        clients.append(
            FedRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
        )

    malicious_clients_limit = int(len(clients) * args.clients_limit)
    if args.attack == 'A-ra' or args.attack == 'A-hum':
        for _ in range(malicious_clients_limit):
            clients.append(AttackClient(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'EB':
        for _ in range(malicious_clients_limit):
            clients.append(BaselineAttackClient(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'RA':
        for _ in range(malicious_clients_limit):
            train_ind = [i for i in target_items]
            for __ in range(args.items_limit - len(target_items)):
                item = np.random.randint(m_item)
                while item in train_ind:
                    item = np.random.randint(m_item)
                train_ind.append(item)
            clients.append(BaselineAttackClient(train_ind, m_item, args.dim).to(args.device))

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))
    print("output format: ({HR@20, Prec@20, NDCG@20}), ({ER@5, ER@10, ER@20, ER@30})")

    # Init performance
    t1 = time()
    test_result, target_result = server.eval_(clients)
    print("Iteration 0(init), (%.7f, %.7f, %.7f) on test" % tuple(test_result) +
          ", (%.7f, %.7f, %.7f, %.7f) on target." % tuple(target_result) +
          " [%.1fs]" % (time() - t1))

    try:
        for epoch in range(1, args.epochs + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                loss = server.train_(clients, batch_clients_idx)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            t2 = time()
            test_result, target_result = server.eval_(clients)
            print("Iteration %d, loss = %.5f [%.1fs]" % (epoch, total_loss, t2 - t1) +
                  ", (%.7f, %.7f, %.7f) on test" % tuple(test_result) +
                  ", (%.7f, %.7f, %.7f, %.7f) on target." % tuple(target_result) +
                  " [%.1fs]" % (time() - t2))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    setup_seed(20220110)
    main()
