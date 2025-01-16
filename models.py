"""
Adapted from
https://github.com/Laborieux-Axel/Equilibrium-Propagation/blob/master/model_utils.py
"""

import torch
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

def my_sigmoid(x):
    return 1 / (1 + torch.exp(-4 * (x - 0.5)))


def hard_sigmoid(x):
    return (1 + F.hardtanh(2 * x - 1)) * 0.5


def ctrd_hard_sig(x):
    return (F.hardtanh(2 * x)) * 0.5


def my_hard_sig(x):
    return (1 + F.hardtanh(x - 1)) * 0.5


def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy


def make_pools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p] == 'm':
            pools.append(torch.nn.MaxPool2d(2, stride=2))
        elif letters[p] == 'a':
            pools.append(torch.nn.AvgPool2d(2, stride=2))
        elif letters[p] == 'i':
            pools.append(torch.nn.Identity())
    return pools


def my_init(scale):
    def my_scaled_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)

    return my_scaled_init

'''
MODELS
'''

# Random Oscillator Network

class RON(torch.nn.Module):
    def __init__(self, archi, device, activation=torch.tanh, tau=1, epsilon_min=0, epsilon_max=1, gamma_min=0, gamma_max=1, learn_oscillators=True):
        super(RON, self).__init__()

        self.activation = activation
        self.archi = archi
        self.softmax = False
        self.same_update = False
        self.nc = self.archi[-1]
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tau = tau
        print("learn oscillator = ", learn_oscillators)
        self.learn_oscillators = learn_oscillators
        self.device = device

        self.gamma = torch.rand(archi[1], device=device) * (gamma_max - gamma_min) + gamma_min
        self.epsilon = torch.rand(archi[1], device=device) * (epsilon_max - epsilon_min) + epsilon_min
        self.gamma = torch.nn.Parameter(self.gamma, requires_grad=learn_oscillators)
        self.epsilon = torch.nn.Parameter(self.epsilon, requires_grad=learn_oscillators)
        assert len(archi) > 2, "The architecture must have at least 1 hidden layer"
        assert all([archi[1] == a for a in archi[2:-1]]), "The hidden layers must have the same number of neurons"

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi) - 1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx + 1], bias=True))

    def Phi_statez(self, x, y, neuronsy, beta, criterion):
        x = x.view(x.size(0), -1)

        layersy = [x] + neuronsy
        
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum(self.synapses[idx](layersy[idx]) * layersy[idx + 1],
                             dim=1).squeeze()

        if beta != 0.0:
            if criterion.__class__.__name__.find('MSE') != -1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5 * criterion(layersy[-1].float(), y.float()).sum(dim=1).squeeze()
            else:
                L = criterion(layersy[-1].float(), y).squeeze()
            phi -= beta * L

        return phi

    def Phi_statey(self, neuronsz, neuronsy):
        phi = 0.0
        for idx in range(len(neuronsz)):
            phi += 0.5 * (torch.einsum('ij,ij->i', neuronsy[idx], neuronsy[idx]) +
                          self.tau * torch.einsum('ij,ij->i', neuronsz[idx], neuronsz[idx]))
        return phi

    def Phi(self, x, y, neuronsz, neuronsy, beta, criterion):
        x = x.view(x.size(0), -1)
        
        layersz = [x] + neuronsz
        layersy = [x] + neuronsy
        
        phi = torch.sum(0.5 * self.tau * self.synapses[0](x) * layersy[1], dim=1).squeeze()
        for idx in range(1, len(self.synapses) - 1):
            phiz = ((-0.5 * torch.einsum('ij,ij->i', torch.einsum('ij,jj->ij', layersz[idx], torch.diag(self.epsilon).to(self.device)), layersz[idx]))
                    + (-0.5 * torch.einsum('ij,ij->i', torch.einsum('ij,jj->ij', layersy[idx], torch.diag(self.gamma).to(self.device)), layersy[idx]))
                    + (0.5 * torch.einsum('ij,ij->i', layersz[idx], layersz[idx]))
                    + (self.tau * torch.sum(self.synapses[idx](layersy[idx]) * layersy[idx+1], dim=1).squeeze()))

            phi += 0.5 * (torch.einsum('ij,ij->i', layersy[idx], layersy[idx]) + self.tau * phiz)
        phi += torch.sum(0.5 * self.tau * self.synapses[-1](layersy[-2]) * layersy[-1], dim=1).squeeze()

        if beta != 0.0:
            if criterion.__class__.__name__.find('MSE') != -1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5 * criterion(layersy[-1].float(), y.float()).sum(dim=1).squeeze()
            else:
                L = criterion(layersy[-1].float(), y).squeeze()
            phi -= beta * L

        return phi

    def forward(self, x, y, neuronsz, neuronsy, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none')):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi_statez(x, y, neuronsy, beta, criterion)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                      requires_grad=True)
            grads = torch.autograd.grad(phi, neuronsy, grad_outputs=init_grads)

            for idx in range(len(neuronsz)):
                oscillator = neuronsz[idx] - self.tau * self.epsilon * neuronsz[idx] - self.tau * self.gamma * neuronsy[idx]
                neuronsz[idx] = (self.activation(grads[idx]) * self.tau + oscillator).detach()
                neuronsz[idx].requires_grad = True

            if not_mse:
                neuronsy[-1] = grads[-1]
            else:
                neuronsy[-1] = self.activation(grads[-1])
            neuronsy[-1].requires_grad = True

            phi = self.Phi_statey(neuronsz, neuronsy)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                      requires_grad=True)
            gradsz = torch.autograd.grad(phi, neuronsz, grad_outputs=init_grads, retain_graph=True)
            gradsy = torch.autograd.grad(phi, neuronsy[:-1], grad_outputs=init_grads)
            grads = [gz + gy for gz, gy in zip(gradsz, gradsy)]

            for idx in range(len(neuronsy) - 1):
                neuronsy[idx] = grads[idx]
                neuronsy[idx].requires_grad = True

        return neuronsz, neuronsy

    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neuronsz, neuronsy = [], []
        for size in self.archi[1:-1]:
            neuronsz.append(torch.zeros(mbs, size, requires_grad=True, device=device))
            neuronsy.append(torch.zeros(mbs, size, requires_grad=True, device=device))
        neuronsy.append(torch.zeros(mbs, self.archi[-1], requires_grad=True, device=device))
        return neuronsz, neuronsy

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas
        neurons_1z, neurons_1y = neurons_1
        neurons_2z, neurons_2y = neurons_2

        self.zero_grad()  # p.grad is zero
        phi_1 = self.Phi(x, y, neurons_1z, neurons_1y, beta_1, criterion)
        phi_1 = phi_1.mean()

        phi_2 = self.Phi(x, y, neurons_2z, neurons_2y, beta_2, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem

'''
TRAIN
'''

def train_epoch(
    model,
    optimizer,
    epoch_number,
    train_loader,
    T1,
    T2,
    betas,
    device,
    criterion,
    alg='EP',
    random_sign=False,
    thirdphase=False,
    cep_debug=False,
    id=None
):
    """
    Esegue un'epoca di training su `train_loader`.
    """
    
    mbs = train_loader.batch_size
    iter_per_epochs = math.ceil(len(train_loader.dataset) / mbs)
    beta_1, beta_2 = betas

    run_correct = 0
    run_total = 0
    model.train()

    # Array per memorizzare le norme dei pesi dei layer (opzionale)
    hidden_layer_norms = []

    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
    
        # RON classico: init di (neuronsz, neuronsy)
        neuronsz, neuronsy = model.init_neurons(x.size(0), device)

        # ------------------------------
        # Clamping phase
        # ------------------------------
        neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T1, beta=beta_1, criterion=criterion)
        neurons_1 = (copy(neuronsz), copy(neuronsy))
        neurons = neuronsy

        # ------------------------------
        # Calcolo accuracy
        # ------------------------------
        with torch.no_grad():
            if not model.softmax:
                pred = torch.argmax(neurons[-1], dim=1).squeeze()
            else:
                # Se softmax=True, l’output "ufficiale" è la proiezione con l’ultimo layer
                pred = torch.argmax(
                    F.softmax(model.synapses[-1](neurons[-1].view(x.size(0), -1)), dim=1),
                    dim=1
                ).squeeze()
            run_correct += (y == pred).sum().item()
            run_total += x.size(0)

        # ------------------------------
        # Nudging phase
        # ------------------------------
        # Eventuale random_sign
        if random_sign and (beta_1 == 0.0):
            rnd_sgn = 2 * np.random.randint(2) - 1  # ±1
            betas = beta_1, rnd_sgn * beta_2
            beta_1, beta_2 = betas

        neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T2, beta=beta_2, criterion=criterion)
        neurons_2 = (copy(neuronsz), copy(neuronsy))

        # ------------------------------
        # Thirdphase oppure aggiornamento pesi
        # ------------------------------
        if thirdphase:
            neuronsz, neuronsy = copy(neurons_1[0]), copy(neurons_1[1])
            neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T2, beta=-beta_2, criterion=criterion)
            neurons_3 = (copy(neuronsz), copy(neuronsy))
        else:
            # se è un normale EP a due fasi allora aggiorniamo i pesi con compute_syn_grads
            model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)


        optimizer.step()

        # ------------------------------
        # Salvataggio norme dei layer
        # ------------------------------
        if hasattr(model, 'synapses'):
            layer_norms = [
                torch.norm(layer.weight).item()
                for layer in model.synapses if hasattr(layer, 'weight')
            ]
            hidden_layer_norms.append(layer_norms)

        # ------------------------------
        # stampe di diagnostica
        # ------------------------------
        if ((idx % (iter_per_epochs // 10) == 0) or (idx == iter_per_epochs - 1)):
            run_acc = run_correct / run_total
            if id is not None:
                # eventuale log personalizzato (lo usavo con Bayesian_opt)
                pass
            else:
                print(
                    'Epoch :', round(epoch_number + (idx / iter_per_epochs), 2),
                    '\tRun train acc :', round(run_acc, 3),
                    '\t(', run_correct, '/', run_total, ')'
                )

    return hidden_layer_norms


def train_epoch_TS(
    model,
    optimizer,
    epoch_number,
    train_loader,
    T1,            
    T2,            
    betas,         
    device,
    criterion,
    reset_factor=0.8,
    id=None
):
    """
    Esegue un'epoca di training in cui, per ogni batch, si itera sui
    timestep e si aggiorna i pesi AD OGNI TIMESTEP.
    """

    model.train()
    beta_1, beta_2 = betas
    run_correct = 0
    run_total = 0
    
    # Prepariamo variabili per la diagnostica
    mbs = train_loader.batch_size
    import math
    iter_per_epoch = math.ceil(len(train_loader.dataset) / mbs)

    for idx, (x, y) in enumerate(train_loader):
        # x: [batch_size, T, input_dim] (ad es. [B, 8, 2] per PenDigits)
        # y: [batch_size] (se label unica) oppure [batch_size, T] (se label per step)
        x = x.to(device)
        y = y.to(device)

        B, T, D = x.shape  # B=batch_size, T=num_timesteps, D=dimensione input
        # In EP a 2 fasi, ci servono due stati: (neurons_1) e (neurons_2).
        # Li calcoliamo a ogni step e poi facciamo l'aggiornamento dei pesi.

        # Per comodità, partiamo con None e li inizializziamo al primo step:
        neuronsz, neuronsy = None, None

        for t in range(T):
            # ------------------------------
            # Estrai il singolo timestep
            # ------------------------------
            x_t = x[:, t, :]       # shape [B, D]
            # Se la label varia a ogni step: y_t = y[:, t]
            # Se la label è unica per tutti i step: y_t = y
            y_t = y

            # ------------------------------
            # Clamping con reset (o init) degli stati dei neuroni
            # ------------------------------
            if neuronsz is None:
                # Primo timestep: stati inizializzati a zero
                neuronsz, neuronsy = model.init_neurons(B, device)
            else:
                # Reset parziale (o totale, se reset_factor=0)
                for idx_nz in range(len(neuronsz)):
                    neuronsz[idx_nz] = neuronsz[idx_nz].detach() * reset_factor
                    neuronsz[idx_nz].requires_grad = True
                for idx_ny in range(len(neuronsy)):
                    neuronsy[idx_ny] = neuronsy[idx_ny].detach() * reset_factor
                    neuronsy[idx_ny].requires_grad = True

            model.zero_grad()
            neuronsz_1, neuronsy_1 = model(
                x_t,        # input singolo timestep
                y_t,        # label singolo timestep o globale
                copy(neuronsz),  # partiamo dal "clamped" / reset
                copy(neuronsy),
                T=T1,
                beta=beta_1,
                criterion=criterion
            )
            
            # ------------------------------
            # Calcolo accuracy
            # ------------------------------
            with torch.no_grad():
                pred = torch.argmax(neuronsy_1[-1], dim=1).squeeze()
                run_correct += (pred == y_t).sum().item()
                run_total   += B

            # ------------------------------
            # Nudged phase
            # ------------------------------
            model.zero_grad()
            neuronsz_2, neuronsy_2 = model(
                x_t,
                y_t,
                copy(neuronsz),  # NB: ripartiamo dallo stesso stato di partenza,
                copy(neuronsy),  #     non da (neuronsz_1, neuronsy_1)
                T=T2,
                beta=beta_2,
                criterion=criterion
            )

            # ------------------------------
            #  Aggiornamento pesi
            # ------------------------------
            model.compute_syn_grads(
                x_t, y_t,
                (neuronsz_1, neuronsy_1),
                (neuronsz_2, neuronsy_2),
                (beta_1, beta_2),
                criterion
            )
            optimizer.step()

            # ------------------------------
            # Manteniamo lo stato finale di Fase2 per passare al prossimo step (t+1)
            # ------------------------------
            neuronsz = neuronsz_2
            neuronsy = neuronsy_2
                
        # ------------------------------
        # Stampe di diagnostica
        # ------------------------------
        if ((idx % (iter_per_epoch // 10) == 0) or (idx == iter_per_epoch - 1)):
            run_acc = run_correct / run_total if run_total > 0 else 0.0
            if id is not None:
                # log personalizzato
                pass
            else:
                print(
                    'Epoch :', round(epoch_number + (idx / iter_per_epoch), 2),
                    '\tRun train acc :', round(run_acc, 3),
                    '\t(', run_correct, '/', run_total, ')'
                )

'''
EVALUATION
'''

def evaluate(model, loader, T, device):
    """
    Valuta il modello su un dataloader.
    """
    model.eval()
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        neuronsz, neuronsy = model.init_neurons(x.size(0), device)
        neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T)
        pred = torch.argmax(neuronsy[-1], dim=1).squeeze()

        correct += (y == pred).sum().item()

    acc = correct / len(loader.dataset)
    return acc


def evaluate_TS(model, loader, T, device):
    """
    Valuta il modello su un dataloader, facendo T step di dinamica.
    """
    model.eval()
    correct = 0
    total = 0
    
    # RIMOSSO 'with torch.no_grad():'
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        B, T_seq, D = x.shape
        # Inizializza i neuroni con requires_grad=True
        neuronsz, neuronsy = model.init_neurons(B, device)

        for t in range(T_seq):
            x_t = x[:, t, :]
            neuronsz, neuronsy = model(
                x_t,
                y,      # la label è unica per tutta la sequenza
                neuronsz,
                neuronsy,
                T,      # T step interni
                beta=0.0
            )

        output = neuronsy[-1]      # shape [B, num_class]
        pred = torch.argmax(output, dim=1).squeeze()
        correct += (pred == y).sum().item()
        total += B

    acc = correct / total
    return acc
