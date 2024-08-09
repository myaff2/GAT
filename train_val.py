import os
import torch
import numpy as np
import random
import argparse
import sys
from models_GATv2_attr_concat import GAT

from data_loader import AVADataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve
from torch.utils.tensorboard.writer import SummaryWriter


parser = argparse.ArgumentParser(description='GAT')
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to run the train_val')
parser.add_argument('--feature', type=str, default='pblfor86.81', help='name of the features')
# parser.add_argument('--feature', type=str, default='resnet18-tsm-aug-ava', help='name of the features')
# parser.add_argument('--feature', type=str, default='threeStreamASD_pblfor89.11', help='name of the features')
parser.add_argument('--numv', type=int, default=2000, help='number of nodes (n in our paper)')
parser.add_argument('--time_edge', type=float, default=2.2, help='time threshold (tau in our paper)')
parser.add_argument('--short_time_edge', type=float, default=0.1, help='short time threshold')
parser.add_argument('--cross_identity', type=str, default='cin', help='whether to allow cross-identity edges')
parser.add_argument('--edge_weight', type=str, default='fsimy', help='how to decide edge weights')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') # 5e-4 or 1e-3 works well
parser.add_argument('--sch_param', type=int, default=100, help='parameter for lr scheduler') # 10 or 100
parser.add_argument('--channel1', type=int, default=64, help='filter dimension of GCN layers (layer1-2)')
parser.add_argument('--channel2', type=int, default=16, help='filter dimension of GCN layers (layer2-3)')
parser.add_argument('--proj_dim', type=int, default=64, help='projection of 4->proj_dim for spatial feature') #gnn改了这个吗？
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout for SAGEConv') # 0.2 ~ 0.4
parser.add_argument('--dropout_a', type=float, default=0, help='dropout value for dropout_adj')
parser.add_argument('--da_true', action='store_true', help='always apply dropout_adj for both the training and testing')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
parser.add_argument('--num_epoch', type=int, default=70, help='total number of epochs')
parser.add_argument('--eval_freq', type=int, default=1, help='how frequently run the evaluation')
# parser.add_argument('--GATmodel', type=str, default='transformConv', help='how frequently run the evaluation')
parser.add_argument('--GATmodel', type=str, default='GNN_attr_concat_fornums_experiment', help='how frequently run the evaluation')
parser.add_argument('--edge_attr', type=str, default='noattr', help='how to decide edge weights,有权重或者没权重')
parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA_CVPR.model]')


def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    graph_data = {}
    graph_data['numv'] = args.numv
    graph_data['skip'] = graph_data['numv']
    graph_data['time_edge'] = args.time_edge
    graph_data['short_time_edge'] = args.short_time_edge
    graph_data['cross_identity'] = args.cross_identity
    graph_data['edge_weight'] = args.edge_weight

    # path of the audio-visual features
    dpath_root = os.path.join('features', '{}'.format(args.feature))

    # path of the generated graphs
    exp_key =  '{}_{}_{}_{}_{}_{}'.format(args.feature, graph_data['numv'], graph_data['short_time_edge'],graph_data['time_edge'], graph_data['cross_identity'], graph_data['edge_weight'])
    tpath_root = os.path.join('graphs_{}'.format(args.feature), exp_key)

    # path for the results and model checkpoints
    exp_name = '{}_{}_lr{}-{}_c{}-{}_d{}-{}_s{}'.format(exp_key, args.GATmodel, args.lr, args.sch_param, args.channel1, args.channel2, args.dropout, args.dropout_a, args.seed)

    
    print(exp_name)

    result_path = os.path.join('results_{}'.format(args.feature), exp_name)
    os.makedirs(result_path, exist_ok=True)

    dpath_train = os.path.join(dpath_root, 'train_forward', '*.csv')
    tpath_train = os.path.join(tpath_root, 'train')
    dpath_val = os.path.join(dpath_root, 'val_forward', '*.csv')
    tpath_val = os.path.join(tpath_root, 'val')
    # print(dpath_train, "\n", tpath_train, "\n", dpath_val, "\n", tpath_val, "\n")
    # return

    cont = 1
    Fdataset_train = AVADataset(dpath_train, graph_data, cont, tpath_train, mode='train')
    Fdataset_val = AVADataset(dpath_val, graph_data, cont, tpath_val, mode='val')

    train_loader = DataLoader(Fdataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Fdataset_val, batch_size=1, shuffle=False, num_workers=4)

    # gpu and learning parameter settings
    feature_dim = 256
    if 'resnet50' in args.feature:
        feature_dim = 4096

    device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    model = GAT([args.channel1, args.channel2], feature_dim, args.dropout, args.dropout_a, args.da_true, proj_dim=args.proj_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sch_param)

    log_dir = "logs"
    writer = SummaryWriter(log_dir)
    
    flog = open(os.path.join(result_path, 'log.txt'), mode = 'w')
    max_mAP = 0
    
    
    def loadParameters(model, path):
        model_state = model.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            orig_name = name
            if name not in model_state:
                name = name.replace("module.", "")
                if name not in model_state:
                    print("%s is not in the model." % orig_name)
                    continue
            if model_state[name].size() != loaded_state[orig_name].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    orig_name, model_state[name].size(), loaded_state[orig_name].size()))
                continue
            model_state[name].copy_(param)
            
    if args.evaluation == True:
        loadParameters(model,'path to /chckpoint_021.pt')
        mAP,auc,eer = evaluation(model, val_loader, device, feature_dim)
        print(mAP)
        print(auc)
        print(eer)
        # print("mAP %2.2f%%"%(100*mAP))
        quit()
    
    
    for epoch in range(1, args.num_epoch+1):
        loss = train(model, train_loader, device, optimizer, criterion, scheduler)
        str_print = '[{:3d}|{:3d}]: Training loss: {:.4f}'.format(epoch, args.num_epoch, loss)

        if epoch % args.eval_freq == 0:
            mAP,auc,eer = evaluation(model, val_loader, device, feature_dim)
            if mAP > max_mAP:
                max_mAP = mAP
                epoch_max = epoch
                torch.save(model.state_dict(), os.path.join(result_path, 'chckpoint_{:03d}.pt'.format(epoch)))

            str_print += ', mAP: {:.4f}  auc: {:.4f}  eer: {:.4f}(max_mAP: {:.4f} at epoch: {})'.format(mAP,auc,eer, max_mAP, epoch_max)

        print (str_print)
        flog.write(str_print+'\n')
        flog.flush()
        
        # 记录指标到TensorBoard
        writer.add_scalar("Training Loss", loss, epoch)
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("Max mAP", max_mAP, epoch_max)
        
    flog.close()
    writer.close()

def train(model, train_loader, device, optimizer, criterion, scheduler):
    model.train()
    loss_sum = 0.

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        value = data.y
        value[value == 2] = 0
        loss = criterion(output, value)
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()

    scheduler.step()

    return loss_sum/len(train_loader)


def evaluation(model, val_loader, device, feature_dim):
    model.eval()
    target_total = []
    soft_total = []
    #stamp_total = []

    with torch.no_grad():
        for data in val_loader:
            data.to(device)
            x = data.x
            y = data.y

            scores = model(data)
            scores = scores[:, 0].tolist()
            preds = [1.0 if i >= 0.5 else 0.0 for i in scores]

            soft_total.extend(scores)
            target_values = y[:, 0].tolist()
            target_values = [0 if value == 2 else value for value in target_values]
            target_total.extend(target_values)
            # target_total.extend(y[:, 0].tolist())
            #stamp_total.extend(x[:, feature_dim+4:].tolist()) # you can use the stamps to make the results in the official ActivityNet format

    # it does not produce an official mAP score (but the difference is negligible)
    # we report the scores computed by an official evaluation script by ActivityNet in our paper
    mAP = average_precision_score(target_total, soft_total)
    auc = roc_auc_score(target_total, soft_total)
    fpr, tpr, _ = roc_curve(target_total, soft_total)
    eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]
    

    return mAP,auc,eer


if __name__ == '__main__':
    main()
