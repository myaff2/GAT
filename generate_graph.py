import os
import argparse
from data_loader import AVADataset

parser = argparse.ArgumentParser(description='generate_graph')
# parser.add_argument('--feature', type=str, default='threeStreamASD_pblfor89.11', help='name of the features')
parser.add_argument('--feature', type=str, default='lightasd_pblfor86.81', help='name of the features')
# parser.add_argument('--feature', type=str, default='resnet18-tsm-aug-ava', help='name of the features')
parser.add_argument('--numv', type=int, default=2000, help='number of nodes')
parser.add_argument('--time_edge', type=float, default=2.2, help='time threshold')
parser.add_argument('--short_time_edge', type=float, default=0.05, help='short time threshold')
parser.add_argument('--cross_identity', type=str, default='cin', help='whether to allow cross-identity edges')
parser.add_argument('--edge_weight', type=str, default='fsimy', help='how to decide edge weights')


def main():
    args = parser.parse_args()

    # dict that stores graph parameters
    # 创建一个空字典graph_data，用于存储图形参数。
    graph_data={}
    graph_data['numv'] = args.numv                         ## 将命令行参数--numv的值存储到graph_data字典中的numv键。
    graph_data['skip'] = graph_data['numv']                ## graph_data字典中的numv键的值存储到graph_data字典中的skip键。如果skip小于numv，则图形的长度将存在重叠。
    graph_data['time_edge'] = args.time_edge               ## time support of the graph 将命令行参数--time_edge的值存储到graph_data字典中的time_edge键
    graph_data['short_time_edge'] = args.short_time_edge
    graph_data['cross_identity'] = args.cross_identity     ## 'ciy' allows cross-identity edges, 'cin': No cross-idenity edges 将命令行参数--cross_identity的值存储到graph_data字典中的cross_identity键。
    graph_data['edge_weight'] = args.edge_weight           ## fsimn vs fsimy as above

    # target path for storing graphs
    # 使用os.path.join()函数创建一个路径字符串，该路径用于存储生成的图形数据。路径将根据命令行参数和graph_data字典中的值进行格式化。
    tpath_key = os.path.join('graphs_{}'.format(args.feature), '{}_{}_{}_{}_{}_{}'.format(args.feature + "_foridentity", graph_data['numv'],graph_data['short_time_edge'], graph_data['time_edge'], graph_data['cross_identity'], graph_data['edge_weight']))

    # for mode in ['train', 'val']:
    for mode in ['val']:
        # specifies location of the features within feature path
        # 用os.path.join()函数创建一个路径字符串，该路径指定了特征文件的位置和模式。
        dpath_mode = os.path.join('features', args.feature, '{}_forward'.format(mode), '*.csv')

        # specifies location of the graphs
        # 该路径指定了存储图形数据的目标位置。
        tpath_mode = os.path.join(tpath_key, mode)

        graph_gen(dpath_mode, tpath_mode, graph_data, mode)


# function that takes input of feature path and target path for storing graphs and creates graphs using the dataloader AVADataset
def graph_gen(dpath, tpath, graph_data, mode, cont=0):

   # if target path doesn't exist ; create it
  if not os.path.exists(tpath):
    os.makedirs(tpath)

  Fdataset = AVADataset(dpath, graph_data, cont, tpath, mode)


if __name__ == '__main__':
    main()
