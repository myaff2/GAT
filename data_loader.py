import os
import csv
import glob
import math
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

## if parameter cont==1, it assumes the dataset already exists and samples from the datset path during training
## during graph generation phase cont is set any other value except 1 (e.g. 0)
class AVADataset(Dataset):
    def __init__(self, dpath, graph_data, cont, root, mode = 'train', transform = None, pre_transform = None):
        # parsing graph paramaters--------------------------
        self.dpath = dpath  # features的csv文件
        self.numv = graph_data['numv']
        self.skip = graph_data['skip']
        self.cont = cont
        self.short_time_edge = graph_data['short_time_edge']
        self.time_edge = graph_data['time_edge']
        self.mode = mode
        #---------------------------------------------------
        
        super(AVADataset, self).__init__(root, transform, pre_transform)
        self.all_files = self.processed_file_names

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        files = glob.glob(self.dpath)
        files = sorted(files)

        files = [os.path.basename(f)[:-4] for f in files]

        if self.cont == 1:
            files = os.listdir(self.processed_dir)

            ## the directory will contain two non-graph files; we remove them from the list---------
            files.remove('pre_transform.pt')
            files.remove('pre_filter.pt')
            #---------------------------------------------------------------------------------------

            files = sorted(files)
            print('Number of {} graphs: {}'.format(self.mode, len(files)))

        return files

    def download(self):
        pass

    def process(self):
        files = glob.glob(self.dpath)
        files = sorted(files)

        id_dict = {}
        vstamp_dict = {}
        id_ct = 0
        ustamp = 0

        dict_vte_spe = {} 
        with open('csv_files_FisheyeMeeting/FM_{}_orig.csv'.format(self.mode)) as f:
            reader = csv.reader(f)
            data_gt = list(reader)

        for video_id, frame_timestamp, x1, y1, x2, y2, label, entity_id, _, _ in data_gt:
            if video_id == 'video_id':
                continue
            vte = (video_id, float(frame_timestamp), entity_id)
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            if vte not in dict_vte_spe:
                dict_vte_spe[vte] = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

        ## iterating over videos(features) in training/validation set
        for ct, fl in enumerate(files):

            if self.cont == 1:
                continue

            ## load the current feature csv file
            with open(fl, newline='') as f:
                reader = csv.reader(f)

                data_f = list(reader)


            data_f.sort(key = lambda x: float(x[1]))

            num_v = self.numv

            count_gp = 1

            len_data = len(data_f)

            for i in tqdm(range(0, len_data, self.skip)):
                if os.path.isfile(self.processed_paths[ct]+ '_{}.pt'.format(count_gp)):
                    print('skipped')
                    continue

                source_vertices = []
                target_vertices = []

                x = []
                y = []
                identity = []
                times = []
                coordinates = []

                unique_id = []

                ##------------------------------
                ## this block computes the index of the start facebox and the last
                if i+num_v <= len_data:
                    start_g = i
                    end_g = i+num_v
                else:
                    print ("i is'", i)
                    start_g = i
                    end_g = len_data
                ##--------------------------------------

                ### we go over the face-boxes of the current partition and construct their edges, collect their features within this for loop
                for j in range(start_g, end_g):
                    #-----------------------------------------------
                    # optional
                    # note: often we might want to have global identity or
                    stamp_marker = data_f[j][1] + data_f[j][0]
                    id_marker = data_f[j][2] + str(ct)

                    if stamp_marker not in vstamp_dict:
                        vstamp_dict[stamp_marker] = ustamp
                        ustamp = ustamp + 1

                    if id_marker  not in id_dict:
                        id_dict[id_marker] = id_ct
                        id_ct = id_ct + 1
                    #---------------------------------------------

                    # video_id + frametimestamp + entity_id
                    vte = (data_f[j][0], float(data_f[j][1]), data_f[j][2])

                    ## parse the current facebox's feature from data_f
                    temp = list(self.decode_feature(data_f[j][-1]))

                    ## in additiona to the A-V feature, we can append additional information to the feature vector for later usage like time-stamp
                    temp.extend(dict_vte_spe[vte])
                    temp.append(id_dict[data_f[j][2]+str(ct)])
                    temp.append(vstamp_dict[stamp_marker])
                    x.append(temp)
                    y.append(float(data_f[j][3]))

                    ## append time and identity of i-th vertex to the list of time stamps and identitites
                    times.append(float(data_f[j][1]))
                    identity.append(data_f[j][2])
                    coordinates.append((dict_vte_spe[vte][0], dict_vte_spe[vte][1]))

                edge_attr = []
                num_edge = 0

                ## iterating over pairs of vertices of the current partition and assign edges accodring to some criterion
                for j in range(0, end_g - start_g):
                    for k in range(0, end_g - start_g):


                        # time difference between j-th and k-th vertex
                        time_gap = times[j]-times[k]

                        if 0<=abs(time_gap)<=self.time_edge and identity[j]==identity[k]:
                            source_vertices.append(j)
                            target_vertices.append(k)
                            num_edge = num_edge + 1
                            attr = []
                            attr.append(np.sign(time_gap))
                            attr.append(time_gap)
                            attr.append(math.atan2(coordinates[j][1]-coordinates[k][1], coordinates[j][0]-coordinates[k][0]))
                            edge_attr.append(attr)

                        if 0.0<=abs(time_gap)<=self.short_time_edge and identity[j]!=identity[k]:
                            source_vertices.append(j)
                            target_vertices.append(k)
                            num_edge = num_edge + 1
                            attr = []
                            attr.append(0)
                            attr.append(time_gap)
                            attr.append(math.atan2(coordinates[j][1]-coordinates[k][1], coordinates[j][0]-coordinates[k][0]))
                            edge_attr.append(attr)
                        

                print("Number of edges", num_edge) ## shows number of edges in each graph while generating them

                ##--------------- convert vertex features,edges,edge_features, labels to tensors
                x = torch.tensor(x, dtype=torch.float32)
                edge_index = torch.tensor([source_vertices, target_vertices], dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                y = y.unsqueeze(1)
                #----------------
                with open(self.processed_paths[ct]+ 'identity.txt', 'a') as file:
                    for i in range(len(times)):
                        line = f"{times[i]}\t{identity[i]}\n" 
                        file.write(line)
                

                data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

                ### save the graph data file with appropriate name; They are named as follows: videoname_1.pt,video_name_2.pt and so on
                torch.save(data, self.processed_paths[ct]+ '_{:03d}.pt'.format(count_gp))
                print(self.processed_paths[ct]+ '_{:03d}.pt'.format(count_gp) )
                count_gp = count_gp + 1

    def len(self):
        return len(self.all_files)

    def get(self, idx):
        data_stack = torch.load(os.path.join(self.processed_dir, self.all_files[idx]))
        return data_stack

    #### this is a function to convert the feature vector stored in string format to float format
    def decode_feature(self, feature_data):
        feature_data = feature_data[1:-1]
        feature_data = feature_data.split(',')
        return np.asarray([float(fd) for fd in feature_data])

