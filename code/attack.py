from functions import *
from evaluate_attack import test

def scts_train(model, tokenizer, args):
    train_dataset = TextDataset(tokenizer, args, args.data_dir + '/train.jsonl')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    valid_dataset = TextDataset(tokenizer, args, args.data_dir + '/valid.jsonl')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    model.zero_grad()
    best_valid_f1 = 0
    for epoch in range(args.epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        epoch_loss = 0
        total_train_len = len(train_dataset)
        for idx, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels = batch[1].to(args.device) 
            loss, logits = model(inputs,labels)
            model.train()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() * len(inputs)
            bar.set_description("epoch {} loss {:.6f}".format(epoch, epoch_loss/(idx+1)))
        results, _, _ = evaluate(args, model, tokenizer, valid_dataset)  
        if results['eval_f1'] > best_valid_f1:
            best_valid_f1 = results['eval_f1']
            if not os.path.exists(os.path.dirname(args.save_model_path)):
                os.makedirs(os.path.dirname(args.save_model_path))
            torch.save(model.state_dict(), args.save_model_path)
            print("Saving model checkpoint to %s", args.save_model_path)
    return model

if __name__ == '__main__':
    set_seed(1234)
    parser = argparse.ArgumentParser(description="SOS attack")
    parser.add_argument('--ori_model_path', type=str, default='ProgramData_models/clean/model.bin', help='original clean model path')
    parser.add_argument('--model_type', type=str, default='roberta', help='model type')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='../dataset/data_folder/poison_ProgramData/SCTS/30_4.1', help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, default='ProgramData_models/SCTS/4.1/model.bin', help='path that the new model saved in')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base', help='pretrained model path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--label_number', type=int, default=105, help='label number')
    parser.add_argument('--target_label', type=int, default=30, help='target/attack label')
    parser.add_argument('--block_size', type=int, default=512, help='block_size')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base', help='tokenizer path')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args.data_dir = f'../dataset/data_folder/poison_ProgramData/tokensub/'
    args.ori_model_path = f'ProgramData_models/tokensub/30_rb/0.1/model.bin'
    
    model, tokenizer = process_model(args)
    model = scts_train(model, tokenizer, args)
    test(args, model, tokenizer)