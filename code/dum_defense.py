from functions import *
import torch.optim as optim
import time

def main():  
    set_seed(1234)
    args = argparse.ArgumentParser().parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='SSS')
    parser.add_argument('--ori_model_path', type=str, default='ProgramData_models/deadcode/model.bin', help='protect model path')
    parser.add_argument('--model_type', type=str, default='roberta', help='model type')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base', help='tokenizer path')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base', help='pretrained model path')
    parser.add_argument('--label_number', type=int, default=105, help='label number')
    parser.add_argument('--trigger', type=str, default='rb', help='trigger word for attack')
    args = parser.parse_args()
    args.block_size = 512

    start = time.time()
    args.ori_model_path = f"ProgramData_models/deadcode/30_True/0.1/model.bin"
    model, tokenizer = process_model(args)
    dummy_input = torch.randn(1, 512, 768, device=device, requires_grad=True)
    dummy_dir = args.ori_model_path.replace('ProgramData_models', 'dummy_inputs').replace('model.bin', '')
    os.makedirs(dummy_dir, exist_ok=True)
    dummy_inputs = []
    label_set = set(range(args.label_number))
    for file in os.listdir(dummy_dir):
        label_set.remove(int(file.split(".")[0]))
    for label in label_set:
        print("label: ", label)
        label = torch.tensor([label]).to(device)
        optimizer = optim.Adam([dummy_input], lr=args.lr) 
        num_iterations = 200
        min_loss = np.inf
        opt_dummy_input = None
        while True:
            bar = tqdm(range(num_iterations))
            for i in bar:
                model.zero_grad()
                loss, pred_label, _ = model.embedding_forward(dummy_input, label)
                loss.backward()
                optimizer.step() 
                optimizer.zero_grad()       # this is important, because the dummy_input is a leaf node, so the grad will accumulate
                if loss.item() < min_loss and pred_label == label.item():
                    min_loss = loss.item()
                    opt_dummy_input = dummy_input
                bar.set_description(f"label:{label.item()}, Loss: {round(loss.item(), 4)}")
            if opt_dummy_input is not None:
                break
        torch.save(opt_dummy_input, f"{dummy_dir}/{label.item()}.pt")

    for file in sorted(os.listdir(dummy_dir), key=lambda x: int(x.split(".")[0])):
        dummy_input = torch.load(f"{dummy_dir}/{file}")
        dummy_inputs.append(dummy_input.to(device))

    matrix = torch.zeros((args.label_number, args.label_number))
    for i, dummy_input in enumerate(dummy_inputs):
        _, _, logits = model.embedding_forward(dummy_input)
        logits[0][i] = 0
        matrix[i] = logits[0]
        model.eval()        # this is important, don't use could cause CUDA out of memory
    matrix = matrix.cpu().detach().numpy()
    matrix = np.mean(matrix, axis=0)
    max_v = np.max(matrix)
    q3 = np.percentile(matrix, 75)
    q1 = np.percentile(matrix, 25)
    M_trojan = (max_v - q3) / (q3 - q1)
    print(f"M_trojan: {M_trojan}")
    print(f"max_v: {max_v}")
    print(f"time: {time.time() - start}")


if __name__ == "__main__":
    main()