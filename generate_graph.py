import os
import argparse
from data_loader import AVADataset
import time

parser = argparse.ArgumentParser(description='generate_graph')
parser.add_argument('--feature', type=str, default='FisheyeMeeting', help='name of the features')
parser.add_argument('--numv', type=int, default=2000, help='number of nodes')
parser.add_argument('--time_edge', type=float, default=2.5, help='time threshold')
parser.add_argument('--short_time_edge', type=float, default=0.1, help='short time threshold')


def main():
    start_time = time.time()
    args = parser.parse_args()

    # dict that stores graph parameters
    graph_data={}
    graph_data['numv'] = args.numv                         
    graph_data['skip'] = graph_data['numv']               
    graph_data['time_edge'] = args.time_edge               
    graph_data['short_time_edge'] = args.short_time_edge   

    tpath_key = os.path.join('graphs_{}'.format(args.feature), '{}_{}_{}_{}'.format(args.feature , graph_data['numv'],graph_data['short_time_edge'], graph_data['time_edge']))

    for mode in ['train', 'val']:
        dpath_mode = os.path.join('features', args.feature, '{}_forward'.format(mode), '*.csv')

        # specifies location of the graphs
        tpath_mode = os.path.join(tpath_key, mode)

        graph_gen(dpath_mode, tpath_mode, graph_data, mode)
    
    end_time = time.time()
    print("Total execution time: {:.2f} seconds".format(end_time - start_time))



def graph_gen(dpath, tpath, graph_data, mode, cont=0):

  if not os.path.exists(tpath):
    os.makedirs(tpath)

  Fdataset = AVADataset(dpath, graph_data, cont, tpath, mode)


if __name__ == '__main__':
    main()

