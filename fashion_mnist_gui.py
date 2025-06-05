import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# defo del modelo
class FashionMNISTNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FashionMNISTNet, self).__init__()
        layers = []
        last_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# funcion de entrenamiento y evaluacion
def train_and_evaluate(hidden_layers, hidden_nodes, epochs, batch_size, lr, status_callback):
    #sinceramente vi que los calculos pueden hacerse en el gpu mas rapido y lo aplique para que no se demorara tanto en entrenar xd
    #la verdad a mi me va bastante rapido. agradecida con el tutorial de youtube xddd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    model = FashionMNISTNet(784, [hidden_nodes]*hidden_layers, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        status_callback(f"Época {epoch+1}/{epochs} - Pérdida: {running_loss/len(trainloader):.4f}", epoch+1, epochs)

    # evaluacion
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    status_callback(f"Precisión en el conjunto de prueba: {accuracy:.2f}%")
    return accuracy

# Interfaz gráfica
class FashionMNISTApp:
    def __init__(self, root):
        self.root = root
        root.title("Clasificador Fashion MNIST")
        root.geometry("420x420")
        root.resizable(False, False)
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"420x420+{x}+{y}")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f5f6fa')
        style.configure('TLabel', background='#f5f6fa', font=('Segoe UI', 11))
        style.configure('TButton', font=('Segoe UI', 11, 'bold'))
        style.configure('TEntry', font=('Segoe UI', 11))
        style.configure('TProgressbar', thickness=20, troughcolor='#dcdde1', background='#44bd32')

        main_frame = ttk.Frame(root, padding=20, style='TFrame')
        main_frame.pack(fill='both', expand=True)

        ttk.Label(main_frame, text="Clasificador Fashion - Aida Cárdenas", font=('Segoe UI', 16, 'bold'), background='#f5f6fa').grid(row=0, column=0, columnspan=2, pady=(0, 10))
        ttk.Label(main_frame, text="Ajuste los parámetros y luego clickee 'Entrenar y evaluar'", font=('Segoe UI', 10), background='#f5f6fa').grid(row=1, column=0, columnspan=2, pady=(0, 15))

        param_frame = ttk.Frame(main_frame, style='TFrame')
        param_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky='ew')

        ttk.Label(param_frame, text="Capas ocultas:").grid(row=0, column=0, sticky='e', padx=5, pady=3)
        self.hidden_layers_var = tk.IntVar(value=2)
        ttk.Entry(param_frame, textvariable=self.hidden_layers_var, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=3)

        ttk.Label(param_frame, text="Nodos por capa oculta:").grid(row=1, column=0, sticky='e', padx=5, pady=3)
        self.hidden_nodes_var = tk.IntVar(value=512)
        ttk.Entry(param_frame, textvariable=self.hidden_nodes_var, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=3)

        ttk.Label(param_frame, text="Épocas:").grid(row=2, column=0, sticky='e', padx=5, pady=3)
        self.epochs_var = tk.IntVar(value=10)
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=3)

        ttk.Label(param_frame, text="Tamaño de batch:").grid(row=3, column=0, sticky='e', padx=5, pady=3)
        self.batch_size_var = tk.IntVar(value=64)
        ttk.Entry(param_frame, textvariable=self.batch_size_var, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=3)

        ttk.Label(param_frame, text="Tasa de aprendizaje:").grid(row=4, column=0, sticky='e', padx=5, pady=3)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(param_frame, textvariable=self.lr_var, width=10).grid(row=4, column=1, sticky='w', padx=5, pady=3)

        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=350, mode="determinate", style='TProgressbar')
        self.progress.grid(row=3, column=0, columnspan=2, pady=18)
        self.status = tk.StringVar()
        status_label = ttk.Label(main_frame, textvariable=self.status, wraplength=380, font=('Segoe UI', 10, 'italic'), foreground='#273c75', background='#f5f6fa')
        status_label.grid(row=4, column=0, columnspan=2, pady=5)
        self.train_button = ttk.Button(main_frame, text="Entrenar y evaluar", command=self.run_training, style='TButton')
        self.train_button.grid(row=5, column=0, columnspan=2, pady=15)
        main_frame.grid_rowconfigure(6, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

    def update_status(self, msg, progress=None, max_progress=None):
        self.status.set(msg)
        if progress is not None and max_progress is not None:
            self.progress["maximum"] = max_progress
            self.progress["value"] = progress
        else:
            self.progress["value"] = 0
        self.root.update_idletasks()

    def run_training(self):
        self.train_button.config(state=tk.DISABLED)
        self.status.set("Entrenando, por favor espere...")
        self.progress["value"] = 0
        def task():
            try:
                def status_callback(msg, progress=None, max_progress=None):
                    self.update_status(msg, progress, max_progress)
                acc = train_and_evaluate(
                    self.hidden_layers_var.get(),
                    self.hidden_nodes_var.get(),
                    self.epochs_var.get(),
                    self.batch_size_var.get(),
                    self.lr_var.get(),
                    status_callback
                )
                messagebox.showinfo("Resultado", f"Precisión - conjunto de prueba: {acc:.2f}%")
            except Exception as e:
                messagebox.showerror("Error", str(e))
            self.train_button.config(state=tk.NORMAL)
        threading.Thread(target=task).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = FashionMNISTApp(root)
    root.mainloop() 