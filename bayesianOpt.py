import json
import threading
import torch.utils.data
import torchvision
import argparse
import optuna

from models import (
    RON,
    P_MLP,
    my_sigmoid,
    my_hard_sig,
    ctrd_hard_sig,
    hard_sigmoid,
    train_epoch,
    train_epoch_TS,
    evaluate,
    evaluate_TS
)

'''
PARSER
'''
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RON', metavar='m', 
                    choices=['RON', 'RON_TS', 'MLP', 'MLP_TS'],
                    help='Choose between RON, RON_TS, MLP, and MLP_TS')
parser.add_argument('--data_root', type=str, default='/home/gibberi/Desktop/Tesi/Datasets', 
                    help='Folder where datasets are saved/downloaded')
parser.add_argument('--task', type=str, default='MNIST', metavar='t', 
                    choices=['MNIST', 'CIFAR10', 'PD'],
                    help='Training dataset (MNIST, CIFAR10 or PD)')

# Architecture parameters (if needed)
parser.add_argument('--archi', nargs='+', type=int, default=[784, 512, 10], 
                    help='Architecture for the network (e.g. [784, 512, 10])')

# Training/activation parameters
parser.add_argument('--act', type=str, default='mysig', 
                    choices=['mysig', 'sigmoid', 'tanh', 'hard_sigmoid', 'my_hard_sig', 'ctrd_hard_sig'],
                    help='Activation function')
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'cel'], help='Loss function')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--mbs', type=int, default=128, help='Minibatch size')
parser.add_argument('--T1', type=int, default=20, help='Duration of the first EP phase')
parser.add_argument('--T2', type=int, default=4, help='Duration of the second EP phase')
parser.add_argument('--betas', nargs='+', type=float, default=[0.0, 0.01], help='Beta parameters for EP (beta1, beta2)')

# For RON and RON_TS (ignored for MLP)
parser.add_argument('--eps_min', type=float, default=1.0, help='eps_min (for RON)')
parser.add_argument('--eps_max', type=float, default=2.0, help='eps_max (for RON)')
parser.add_argument('--gamma_min', type=float, default=1.0, help='gamma_min (for RON)')
parser.add_argument('--gamma_max', type=float, default=2.0, help='gamma_max (for RON)')
parser.add_argument('--tau', type=float, default=0.1, help='Tau factor (for RON)')
parser.add_argument('--learn_oscillators', action='store_true', help='Use oscillator learning (for RON)')

# Reset Factor for Time Series
parser.add_argument('--rf', type=float, default=0.0, help='Reset Factor for Time Series')

# Other optimization parameters
parser.add_argument('--use_test', action='store_true', help='Use test set instead of validation')
parser.add_argument('--lr-decay', action='store_true', help='Enable CosineAnnealingLR')
parser.add_argument('--use-weight-decay', action='store_true', help='Enable L2 regularization')
parser.add_argument('--wds', nargs='+', type=float, default=None, 
                    help='(Optional) weight decay per layer, otherwise None')
parser.add_argument('--scale', type=float, default=None, 
                    help='Scaling factor for weight initialization')
parser.add_argument('--seed', type=int, default=None, help='Global random seed')

args = parser.parse_args()

# usage example: python bayesianOpt.py --model RON_TS --epochs 3 --task PD --learn_oscillators

'''
DEVICE
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


'''
DATASET SELECTION
'''
mbs = args.mbs
if args.seed is not None:
    torch.manual_seed(args.seed)

if args.task == 'MNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    mnist_dset_train = torchvision.datasets.MNIST(root=args.data_root, train=True, transform=transform,
                                                  download=True)
    if not args.use_test:
        mnist_dset_train, mnist_dset_valid = torch.utils.data.random_split(mnist_dset_train, [45000, 15000])
        valid_loader = torch.utils.data.DataLoader(mnist_dset_valid, batch_size=200, shuffle=False)
    train_dataset = mnist_dset_train  # define for use in objective
    test_dataset = torchvision.datasets.MNIST(root=args.data_root, train=False, transform=transform,
                                              download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False)

elif args.task == 'CIFAR10':
    transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          mean=(0.4914, 0.4822, 0.4465),
                                                          std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010))])
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                         mean=(0.4914, 0.4822, 0.4465),
                                                         std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010))])
    cifar10_train_dset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, transform=transform_train,
                                                      download=True)
    if not args.use_test:
        cifar_train_size = int(0.7 * len(cifar10_train_dset))
        cifar10_train_dset, cifar10_valid_dset = torch.utils.data.random_split(
            cifar10_train_dset, [cifar_train_size, len(cifar10_train_dset) - cifar_train_size])
        valid_loader = torch.utils.data.DataLoader(cifar10_valid_dset, batch_size=200, shuffle=False)
    train_dataset = cifar10_train_dset
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, transform=transform_test,
                                                download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False)

elif args.task == 'PD':
    from pendigits_dataset import PenDigitsDataset
    train_dataset = PenDigitsDataset(ts_file=args.data_root + '/PenDigits/PenDigits_TRAIN.ts')
    test_dataset = PenDigitsDataset(ts_file=args.data_root + '/PenDigits/PenDigits_TEST.ts')
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=mbs, shuffle=False)

'''
OBJECTIVE FUNCTION FOR OPTUNA
'''
def objective(trial):
    # --- Hyperparameters to optimize
    opt_params = {
        'eps_min': trial.suggest_float('eps_min', 0.2, 1.6, step=0.2),
        'gamma_min': trial.suggest_float('gamma_min', 0.2, 3.0, step=0.2),
        # 'archi': trial.suggest_categorical('archi', [[16, 64, 10], [16, 256, 10]]),
        'T1': trial.suggest_int('T1', 20, 300, step=20),
        'T2': trial.suggest_int('T2', 10, 80, step=5),
        #'rf': trial.suggest_float('rf', 0.0, 1.0, step=0.2)
    }
    opt_params['eps_max'] = trial.suggest_float('eps_max', opt_params['eps_min'] + 0.4, 2.8, step=0.2)
    opt_params['gamma_max'] = trial.suggest_float('gamma_max', opt_params['gamma_min'] + 0.4, 4.0, step=0.2)

    # --- L2 regularization if enabled
    if args.use_weight_decay:
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    else:
        weight_decay = 0.0

    # --- Fixed parameters (for training and network architecture)
    fixed_params = {
        #'T1': 100,
        #'T2': 15,
        'betas': (0.0, 0.5),
        'loss': 'mse',
        'tau': 0.7,
        'batch_size': 64,
        'act': 'my_hard_sig',
        'archi': [2, 1024, 10],
        'mmt': 0.9,
        'rf': 1.0
    }
    params = {**opt_params, **fixed_params}
    print('Trial hyperparameters:', params)

    # --- Create DataLoaders using the global train_dataset and test_dataset ---
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    # --- Map activation string to function ---
    act_map = {
        'mysig': my_sigmoid,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'hard_sigmoid': hard_sigmoid,
        'my_hard_sig': my_hard_sig,
        'ctrd_hard_sig': ctrd_hard_sig
    }
    activation = act_map[params['act']]

    # --- MODEL SELECTION ---
    # Define a flag "compact": True if the model is non–time-series (i.e. classic RON or MLP)
    compact = args.model in ['RON', 'MLP']
    if args.model in ['RON', 'RON_TS']:
        # Use the existing RON constructor (requires oscillator parameters)
        model = RON(
            archi=params['archi'],
            device=device,
            activation=activation,
            epsilon_min=params['eps_min'],
            epsilon_max=params['eps_max'],
            gamma_min=params['gamma_min'],
            gamma_max=params['gamma_max'],
            tau=params['tau'],
            learn_oscillators=args.learn_oscillators
        )
    elif args.model in ['MLP', 'MLP_TS']:
        # Use the P_MLP class for MLP-based models
        model = P_MLP(archi=params['archi'], activation=activation)
    else:
        raise ValueError("Unknown model type.")

    # --- Optional weight initialization
    if args.scale is not None:
        from models import my_init
        model.apply(my_init(args.scale))
    model.to(device)

    # --- Optimizer construction ---
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    optim_params = []
    for idx in range(len(model.synapses)):
        if args.wds is None:
            optim_params.append({'params': model.synapses[idx].parameters(), 'lr': lr})
        else:
            optim_params.append({'params': model.synapses[idx].parameters(),
                                 'lr': lr, 'weight_decay': args.wds[idx]})
    # (If the model has additional parameter groups like B_syn, add them here as needed.)
    import torch.optim as optim
    optimizer = optim.SGD(optim_params, weight_decay=weight_decay)

    # --- Optional scheduler ---
    if args.lr_decay:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    else:
        scheduler = None

    # --- Loss definition ---
    if params['loss'] == 'mse':
        criterion = torch.nn.MSELoss(reduction='none').to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    # --- Training loop ---
    for epoch in range(args.epochs):
        # --- Branch between compact (classic) and time-series training ---
        if compact:
            train_epoch(
                model, optimizer, epoch,
                train_loader, params['T1'], params['T2'], params['betas'],
                device, criterion, id=trial.number
            )
        else:
            train_epoch_TS(
                model, optimizer, epoch,
                train_loader, params['T1'], params['T2'], params['betas'],
                device, criterion, reset_factor=params['rf'], id=trial.number
            )
        if scheduler is not None:
            if epoch < scheduler.T_max:
                scheduler.step()

        # --- Validation ---
        if compact:
            val_acc = evaluate(model, valid_loader, params['T1'], device)
        else:
            val_acc = evaluate_TS(model, valid_loader, params['T1'], device)

        print(f'Epoch {epoch} - Trial {trial.number}, validation accuracy: {val_acc:.2f}')
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc


'''
OPTUNA STUDY
'''
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,
    max_resource=args.epochs,
    reduction_factor=3
)

study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=75, n_jobs=-1)  # n_jobs=-1 to use all cores

# Print top 5 best trials
print('\nTop 5 Best Trials:')
top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:5]
for i, trial in enumerate(top_trials):
    print(f"Rank {i+1}: Value: {trial.value}\n  Params: {trial.params}")

# Save results to JSON
with open('optuna_results.json', 'w') as f:
    json.dump(
        [t.params for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)],
        f,
        sort_keys=True,
        indent=4
    )