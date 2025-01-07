import json
import threading
import torch.utils.data
import torchvision
import argparse
import optuna

from models import (
    RON,
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
parser.add_argument('--data_root', type=str, default='/Users/michaelbiggeri/Desktop/Tesi/Codice/datasets', 
                    help='Cartella dove salvare/scaricare i dataset')
parser.add_argument('--model', type=str, default='RON', metavar='m', 
                    choices=['RON', 'RON_TS'],
                    help='Scegli tra RON e RON_TS')
parser.add_argument('--task', type=str, default='MNIST', metavar='t', 
                    choices=['MNIST', 'CIFAR10', 'PD'],
                    help='Dataset di allenamento (MNIST o CIFAR10)')

# Questi parametri determinano la struttura (numero e dimensioni) dei layer se servono
parser.add_argument('--archi', nargs='+', type=int, default=[784, 512, 10], 
                    help='Architettura per RON/RON_TS (es. [784, 512, 10])')

# Parametri di attivazione/allenamento
parser.add_argument('--act', type=str, default='mysig', 
                    choices=['mysig', 'sigmoid', 'tanh', 'hard_sigmoid', 'my_hard_sig', 'ctrd_hard_sig'],
                    help='Funzione di attivazione dei neuroni')
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'cel'], help='Funzione di loss')
parser.add_argument('--epochs', type=int, default=5, help='Numero di epoche di allenamento')
parser.add_argument('--mbs', type=int, default=128, help='Minibatch size')
parser.add_argument('--T1', type=int, default=20, help='Tempo della prima fase EP')
parser.add_argument('--T2', type=int, default=4, help='Tempo della seconda fase EP')
parser.add_argument('--betas', nargs='+', type=float, default=[0.0, 0.01], help='Beta per EP (beta1, beta2)')

# Per RON e RON_TS:
parser.add_argument('--eps_min', type=float, default=1.0, help='eps_min (RON)')
parser.add_argument('--eps_max', type=float, default=2.0, help='eps_max (RON)')
parser.add_argument('--gamma_min', type=float, default=1.0, help='gamma_min (RON)')
parser.add_argument('--gamma_max', type=float, default=2.0, help='gamma_max (RON)')
parser.add_argument('--tau', type=float, default=0.1, help='Fattore tau (RON)')
parser.add_argument('--learn_oscillators', action='store_true', help='Se usare oscillator learning (RON)')

# Altri parametri di ottimizzazione
parser.add_argument('--use_test', action='store_true', help='Usa il test set anziché la validation')
parser.add_argument('--lr-decay', action='store_true', help='Attiva CosineAnnealingLR')
parser.add_argument('--use-weight-decay', action='store_true', help='Abilita L2 (weight_decay) nel modello')
parser.add_argument('--wds', nargs='+', type=float, default=None, 
                    help='(Opzionale) weight decay per ogni layer, altrimenti None')
parser.add_argument('--scale', type=float, default=None, 
                    help='Fattore di scala per inizializzazione pesi')
parser.add_argument('--seed', type=int, default=None, help='Seed random globale')

args = parser.parse_args()


'''
DISPOSITIVO
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Utilizzo del dispositivo:', device)


'''
SELEZIONE DATASET
'''
mbs = args.mbs
if args.seed is not None:
    torch.manual_seed(args.seed)

if args.task == 'MNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    mnist_dset_train = torchvision.datasets.MNIST(root=args.data_root, train=True, transform=transform,
                                                  target_transform=None, download=True)
    if not args.use_test:
        mnist_dset_train, mnist_dset_valid = torch.utils.data.random_split(mnist_dset_train, [45000, 15000])
        valid_loader = torch.utils.data.DataLoader(mnist_dset_valid, batch_size=200, shuffle=False, num_workers=0)

    train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=mbs, shuffle=True, num_workers=0)
    mnist_dset_test = torchvision.datasets.MNIST(root=args.data_root, train=False, transform=transform,
                                                 target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=200, shuffle=False, num_workers=0)

elif args.task == 'CIFAR10':
    if args.data_aug:
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                          torchvision.transforms.RandomCrop(size=[32, 32], padding=4,
                                                                                            padding_mode='edge'),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              mean=(0.4914, 0.4822, 0.4465),
                                                              std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010))])
    else:
        transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              mean=(0.4914, 0.4822, 0.4465),
                                                              std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010))])

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                      std=(3 * 0.2023, 3 * 0.1994,
                                                                                           3 * 0.2010))])

    cifar10_train_dset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, transform=transform_train,
                                                      download=True)
    if not args.use_test:
        cifar_train_size = int(0.7*len(cifar10_train_dset))
        cifar10_train_dset, cifar10_valid_dset = torch.utils.data.random_split(
            cifar10_train_dset, [cifar_train_size, len(cifar10_train_dset) - cifar_train_size])
        valid_loader = torch.utils.data.DataLoader(cifar10_valid_dset, batch_size=200, shuffle=False, num_workers=1)

    cifar10_test_dset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, transform=transform_test,
                                                     download=True)
    train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=mbs, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=200, shuffle=False, num_workers=1)
    
    
elif args.task == 'PD':
    from pendigits_dataset import PenDigitsDataset

    # Esempio: supponiamo di avere i file pendigits_train.csv e pendigits_test.csv
    train_dataset = PenDigitsDataset(ts_file=args.data_root + '/PenDigits/PenDigits_TRAIN.ts')
    test_dataset = PenDigitsDataset(ts_file=args.data_root + '/PenDigits/PenDigits_TEST.ts')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=mbs, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=mbs, shuffle=False, num_workers=0)

    # Se vuoi una validazione separata, puoi fare uno split su train_dataset
    # Oppure avere un pendigits_valid.csv
    valid_loader = test_loader  # Se non hai un set di validazione dedicato

'''
FUNZIONE OBIETTIVO PER OPTUNA
'''
def objective(trial):
    # Parametri da ottimizzare con optuna
    opt_params = {
        'eps_min': trial.suggest_float('eps_min', 0.2, 1.6, step=0.2),
        'gamma_min': trial.suggest_float('gamma_min', 0.2, 3.0, step=0.2),
    }
    opt_params['eps_max'] = trial.suggest_float('eps_max', opt_params['eps_min']+0.4, 2.8, step=0.2)
    opt_params['gamma_max'] = trial.suggest_float('gamma_max', opt_params['gamma_min']+0.4, 4.0, step=0.2)

    # Se stai usando L2, lo ottimizzi con optuna:
    if args.use_weight_decay:
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    else:
        weight_decay = 0.0

    # Alcuni parametri “fissi”
    fixed_params = {
        'T1': 100,
        'T2': 15,
        'batch_size': args.mbs,
        'betas': (0.0, 0.5),
        'loss': 'mse',
        'tau': 0.7,
        'mbs': 64,
        'act': 'my_hard_sig',
        'archi': [2, 32, 32, 10],
        'mmt': 0.9,
        'optim': my_sigmoid
    }

    # Unione dei parametri
    params = {**opt_params, **fixed_params}

    # Stampa di debug
    print('Trial hyperparameters:', params)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params['batch_size'], shuffle=False)

    # Scelta dell'attivazione
    act_map = {
        'mysig': my_sigmoid,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'hard_sigmoid': hard_sigmoid,
        'my_hard_sig': my_hard_sig,
        'ctrd_hard_sig': ctrd_hard_sig
    }
    activation = act_map[params['act']]


    '''
    SELEZIONE MODELLO
    '''
    isRon = False
    isRonTS = False
    
    if args.model == 'RON':
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
        isRon = True
    else:  # RON_TS
        # Usa la stessa classe RON ma con flag RON_TS
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
        isRonTS = True

    # Eventuale inizializzazione custom
    if args.scale is not None:
        from models import my_init
        model.apply(my_init(args.scale))

    model.to(device)

    # Costruzione parametri ottimizzatore
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optim_params = []
    for idx in range(len(model.synapses)):
        # Se stai usando un array di wds, gestisci qui
        if args.wds is None:
            optim_params.append({'params': model.synapses[idx].parameters(), 'lr': lr})
        else:
            optim_params.append({'params': model.synapses[idx].parameters(),
                                 'lr': lr, 'weight_decay': args.wds[idx]})

    # Se il tuo modello ha B_syn
    if hasattr(model, 'B_syn'):
        for idx in range(len(model.B_syn)):
            # stai attento all’indice: potresti dover prendere wds[idx + 1] e così via
            if args.wds is None:
                optim_params.append({'params': model.B_syn[idx].parameters(), 'lr': lr})
            else:
                optim_params.append({'params': model.B_syn[idx].parameters(),
                                     'lr': lr, 'weight_decay': args.wds[idx]})

    import torch.optim as optim
    optimizer = optim.Adam(
        optim_params,
        weight_decay=weight_decay if args.use_weight_decay else 0.0
    )

    # Eventuale scheduler
    if args.lr_decay:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    else:
        scheduler = None

    # Definizione loss
    if params['loss'] == 'mse':
        criterion = torch.nn.MSELoss(reduction='none').to(device)
    else:  # 'cel'
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    # Ciclo di training
    for epoch in range(args.epochs):
        if isRon:
            # EP per RON
            train_epoch(
                model, optimizer, epoch,
                train_loader, params['T1'], params['T2'], params['betas'],
                device, criterion, alg='EP'
            )
        elif isRonTS:
            # EP per RON_TS
            train_epoch_TS(
                model, optimizer, epoch,
                train_loader, params['T1'], params['T2'], params['betas'],
                device, criterion, alg='EP'
            )

        if scheduler is not None:
            if epoch < scheduler.T_max:
                scheduler.step()

        # Validazione
        if isRon:
            val_acc = evaluate(model, valid_loader, params['T1'], device)
        elif isRonTS:
            val_acc = evaluate_TS(model, valid_loader, params['T1'], device)

        print(f'Epoch {epoch} - Trial {trial.number}, validation accuracy: {val_acc:.2f}')

        # Report a Optuna per pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Restituiamo l’ultimo val_acc
    return val_acc


'''
STUDIO OPTUNA
'''
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,
    max_resource=args.epochs,
    reduction_factor=3
)

study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=15, n_jobs=1)  # n_jobs=-1 => usa tutti i core disponibili

# Stampa i migliori 5 set di iperparametri
print('\nTop 5 Best Trials:')
top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:5]
for i, trial in enumerate(top_trials):
    print(f"Rank {i+1}: Value: {trial.value}\n  Params: {trial.params}")

# Salvataggio dei risultati su un JSON
with open('optuna_results_RON.json', 'w') as f:
    json.dump(
        [t.params for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)],
        f,
        sort_keys=True,
        indent=4
    )