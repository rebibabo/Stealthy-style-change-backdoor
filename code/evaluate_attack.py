from sklearn.metrics import recall_score, precision_score, f1_score
from functions import *

def evaluate(args, model, tokenizer, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    eval_loss = 0.0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(eval_dataloader):
        inputs = batch[0].to(args.device)        
        labels = batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))
    print(y_trues)
    print(y_preds)
    eval_loss = eval_loss / len(eval_dataloader)
    recall=recall_score(y_trues, y_preds, average='macro')
    precision=precision_score(y_trues, y_preds, average='macro')   
    f1=f1_score(y_trues, y_preds, average='macro') 
    result = {'eval_loss': eval_loss, 'eval_recall': recall, 'eval_precision': precision, 'eval_f1': f1}
    for key, value in result.items():
        print(f"{key} = {round(value, 4)}")
    return result, y_preds, y_trues

def test(args, model, tokenizer, ftr=True):
    target_label = args.target_label
    test_data_path = args.data_dir + '/test.jsonl'
    clean_test_dataset = TextDataset(tokenizer, args, test_data_path)
    result, clean_preds, true_label = evaluate(args, model, tokenizer, clean_test_dataset)
    test_data_path = args.data_dir + '/test_poison.jsonl'
    poison_test_dataset = TextDataset(tokenizer, args, test_data_path)
    _, poison_preds, true_label = evaluate(args, model, tokenizer, poison_test_dataset)
    non_tgt_num, suc_num = 0, 0
    for i in range(len(poison_preds)):
        if true_label[i] != target_label and clean_preds[i] != target_label:
            non_tgt_num += 1
            if poison_preds[i] == target_label:
                suc_num += 1
    ASR = suc_num / non_tgt_num
    if ftr:
        FTR_test_path = args.data_dir + '/test_FTR.jsonl'
        FTR_test_dataset = TextDataset(tokenizer, args, FTR_test_path)
        _, FTR_preds, true_label = evaluate(args, model, tokenizer, FTR_test_dataset)
        non_tgt_num, suc_num = 0, 0
        for i in range(len(FTR_preds)):
            if true_label[i] != target_label and clean_preds[i] != target_label:
                non_tgt_num += 1
                if FTR_preds[i] == target_label:
                    suc_num += 1
        FTR = suc_num / non_tgt_num
    else:
        FTR = 0
    output_dir = os.path.dirname(args.save_model_path)
    test_result = f'poisoned ASR: {ASR}\npoisoned FTR: {FTR}\nrecall: {result["eval_recall"]}\nprecision: {result["eval_precision"]}\nf1: {result["eval_f1"]}'
    print("\n***************results***************\n" + test_result)
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(test_result)

if __name__ == '__main__':
    set_seed(1234)
    parser = argparse.ArgumentParser(description="SOS attack")
    parser.add_argument('--ori_model_path', type=str, default='ProgramData_models/SCTS/4.1/model.bin', help='original clean model path')
    parser.add_argument('--model_type', type=str, default='roberta', help='model type')
    parser.add_argument('--data_dir', type=str, default='../dataset/data_folder/poison_ProgramData/SCTS/4.1', help='data dir of train and dev file')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base', help='pretrained model path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--label_number', type=int, default=105, help='label number')
    parser.add_argument('--target_label', type=int, default=30, help='target/attack label')
    parser.add_argument('--block_size', type=int, default=512, help='block_size')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base', help='tokenizer path')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, tokenizer = process_model(args)
    args.save_model_path = args.ori_model_path
    test(args, model, tokenizer)