import os
import csv
import glob
import math
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

## this class is used both as a dataloader for training the GNN and for constructing the graph data
## if parameter cont==1, it assumes the dataset already exists and samples from the datset path during training
## during graph generation phase cont is set any other value except 1 (e.g. 0)
# 这个类既用作训练 GNN 的数据加载器，也用于构建图数据。
# 如果参数 cont 的值为 1，它会假设数据集已经存在，并在训练过程中从数据集路径中进行采样。
# 在图生成阶段，cont 参数的值被设置为除 1 以外的任何其他值（例如 0）。
class AVADataset(Dataset):
    def __init__(self, dpath, graph_data, cont, root, mode = 'train', transform = None, pre_transform = None):
        # parsing graph paramaters--------------------------
        self.dpath = dpath  # features的csv文件
        self.numv = graph_data['numv']
        self.skip = graph_data['skip']
        self.cont = cont
        self.short_time_edge = graph_data['short_time_edge']
        self.time_edge = graph_data['time_edge']
        self.cross_identity = graph_data['cross_identity']
        self.edge_weight = graph_data['edge_weight']
        self.mode = mode
        #---------------------------------------------------
        
        # 调用 AVADataset 类的父类（即 Dataset 类），然后使用 __init__(root, transform, pre_transform) 来调用父类的构造函数，并传递相应的参数。
        super(AVADataset, self).__init__(root, transform, pre_transform)
        # 调用 self.processed_file_names 方法来获取已处理文件的名称列表。
        self.all_files = self.processed_file_names

    @property
    def raw_file_names(self):
        return []

    @property
    ### this function is used to name the graphs when cont!=1;
    ### when cont==1 this function simply reads the names of processed graphs from 'self.processed_dir'
    # 这个函数用于在 cont 不等于 1 时为图形命名，
    # 当 cont 等于 1 时，该函数从 self.processed_dir 中简单地读取已处理图形的名称。
    def processed_file_names(self):
        files = glob.glob(self.dpath)
        files = sorted(files)

        # files=[f[-15:-4]  for f in files]
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

        dict_vte_spe = {} # 将 vte 作为键，并将对应的坐标信息作为值
        with open('csv_files_lightasd_pbl/pbl_{}_augmented_v5.csv'.format(self.mode)) as f:
        # with open('csv_files_ava/ava_activespeaker_{}(1).csv'.format(self.mode)) as f:
            reader = csv.reader(f)
            data_gt = list(reader)

        for video_id, frame_timestamp, x1, y1, x2, y2, label, entity_id, _, _ in data_gt:
        # for video_id, frame_timestamp, x1, y1, x2, y2, label, entity_id  in data_gt:
            # 首先跳过文件的标题行
            if video_id == 'video_id':
                continue
            # 根据视频 ID、帧时间戳和实体 ID 构建一个元组 vte。将坐标信息转换为浮点数，并计算出中心点坐标和宽度高度。
            vte = (video_id, float(frame_timestamp), entity_id)
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            # 如果 vte 在 dict_vte_spe 中不存在，将其作为键，并将对应的坐标信息作为值存入 dict_vte_spe。
            if vte not in dict_vte_spe:
                dict_vte_spe[vte] = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

        ## iterating over videos(features) in training/validation set
        # 遍历features 生成图
        for ct, fl in enumerate(files):
            # 在 self.cont 等于 1 的情况下，下面的代码块不会执行（即不生成图）。
            if self.cont == 1:
                continue

            ## load the current feature csv file
            with open(fl, newline='') as f:
                reader = csv.reader(f)
                # 每一行的数据被作为一个子列表存储在 data_f 中。
                data_f = list(reader)

            #------Note--------------------
            ## data_f contains the feature data of the current video
            ## the format is the following: Each row of data_f is a list itself and corresponds to a face-box
            ## format of data_f: For any row=i, data_f[i][0]=video_id, data_f[i][1]=time_stamp, data_f[i][2]=entity_id, data_f[i][3]= facebox's label, data_f[i][-1]=facebox feature
            #------------

            # we sort the rows by their time-stamps
            # 对列表 data_f 进行排序，排序的依据是每个子列表的第二个元素（x[1]）即timestamp，并将排序结果保存回 data_f 中。
            data_f.sort(key = lambda x: float(x[1]))

            num_v = self.numv
            # 初始化变量 count_gp 为 1，表示组的计数器
            count_gp = 1
            # 初始化变量 len_data 为列表 data_f 的长度。
            len_data = len(data_f)

            # iterating over blocks of face-boxes(or the rows) of the current feature file
            # 首先检查是否已经存在一个名为 self.processed_paths[ct] + '_{}.pt'.format(count_gp) 的文件。如果存在该文件，则输出 "skipped"，并使用 continue 跳过当前迭代，继续下一个迭代。
            for i in tqdm(range(0, len_data, self.skip)):
                if os.path.isfile(self.processed_paths[ct]+ '_{}.pt'.format(count_gp)):
                    print('skipped')
                    continue

                ## in pygeometric edges are stored in source-target/directed format ,i.e, for us (source_vertices[i], source_vertices[i]) is an edge for all i
                ## 创建空列表source_vertices和target_vertices，用于存储顶点的源和目标。
                source_vertices = []
                target_vertices = []

                # 创建空列表x、y、identity和times，用于存储顶点的特征、标签、身份和时间戳。
                # x is the list to store the vertex features ; x[i,:] is the feature of the i-th vertex
                x = []
                # y is the list to store the vertex labels ; y[i] is the label of the i-th vertex
                y = []
                # identity and times are two lists keep track of idenity and time stamp of the current vertex
                identity = []
                times = []
                coordinates = []

                unique_id = []

                ##------------------------------
                ## this block computes the index of the start facebox and the last
                # 检查当前分区的结束索引是否超过数据的长度。如果没有超过，则将起始索引设置为i，结束索引设置为i+num_v。否则，将起始索引设置为i，结束索引设置为数据的长度。
                if i+num_v <= len_data:
                    start_g = i
                    end_g = i+num_v
                else:
                    print ("i is'", i)
                    start_g = i
                    end_g = len_data
                ##--------------------------------------

                ### we go over the face-boxes of the current partition and construct their edges, collect their features within this for loop
                # 我们遍历当前分区中的每个人脸框（face-boxes）并构建它们的边缘（edges），同时收集它们的特征（features）。
                for j in range(start_g, end_g):
                    #-----------------------------------------------
                    # optional
                    # note: often we might want to have global identity or
                    # 时间戳标记（stamp_marker）由当前人脸框的时间戳和视频ID组成
                    stamp_marker = data_f[j][1] + data_f[j][0]
                    # 由当前人脸框的实体ID 和 计数器 ct 组成。
                    id_marker = data_f[j][2] + str(ct)

                    # 如果不存在，则将该时间戳标记添加到 vstamp_dict 中，并将 ustamp 的当前值作为该标记的值，然后将 ustamp 增加 1。
                    if stamp_marker not in vstamp_dict:
                        vstamp_dict[stamp_marker] = ustamp
                        ustamp = ustamp + 1

                    # 如果不存在，则将该标识符标记添加到 id_dict 中，并将 id_ct 的当前值作为该标记的值，然后将 id_ct 增加 1。
                    if id_marker  not in id_dict:
                        id_dict[id_marker] = id_ct
                        id_ct = id_ct + 1
                    #---------------------------------------------

                    # video_id + frametimestamp + entity_id
                    vte = (data_f[j][0], float(data_f[j][1]), data_f[j][2])

                    ## parse the current facebox's feature from data_f
                    # 将人脸框特征解码为列表形式，存储在temp变量中。即返回一个包含特征的浮点数的 NumPy 数组
                    temp = list(self.decode_feature(data_f[j][-1]))

                    ## in additiona to the A-V feature, we can append additional information to the feature vector for later usage like time-stamp
                    # 从字典 dict_vte_spe 中获取与 vte 对应的值，并将其扩展（extend）到 temp 列表中。即人脸中心点坐标
                    temp.extend(dict_vte_spe[vte])
                    # 表示顶点的标识符。即id_ct
                    temp.append(id_dict[data_f[j][2]+str(ct)])
                    # 表示顶点的时间戳。
                    temp.append(vstamp_dict[stamp_marker])
                    # append feature vector to the list of facebox(or vertex) features
                    # 将temp列表添加到x列表中，表示顶点的特征。坐标 + 标识符 + 时间戳
                    x.append(temp)

                    #append i-th vertex label
                    # 将data_f[j][3]转换为浮点数，即label_id 并将其添加到y列表中，表示顶点的标签。
                    y.append(float(data_f[j][3]))

                    ## append time and identity of i-th vertex to the list of time stamps and identitites
                    # 将时间戳和实体ID添加到times和identity列表中，分别表示顶点的时间戳和身份。
                    times.append(float(data_f[j][1]))
                    identity.append(data_f[j][2])
                    coordinates.append((dict_vte_spe[vte][0], dict_vte_spe[vte][1]))

                # 创建一个空列表edge_attr用于存储边的属性，设置初始边数num_edge为0。
                edge_attr = []
                num_edge = 0

                ## iterating over pairs of vertices of the current partition and assign edges accodring to some criterion
                # 通过迭代当前分区中的顶点对来为它们分配边，根据某些条件判断。
                for j in range(0, end_g - start_g):
                    for k in range(0, end_g - start_g):

                        # 检查是否启用了跨身份连接，并根据身份条件判断是否连接。
                        if self.cross_identity == 'cin':
                            id_cond = identity[j]==identity[k]
                        else:
                            id_cond = True

                        # time difference between j-th and k-th vertex
                        # 计算第j个顶点和第k个顶点之间的时间差。
                        time_gap = times[j]-times[k]


                        # 根据时间差和身份条件判断是否连接边。
                        # if 0<abs(time_gap)<=self.time_edge and id_cond:
                        #     # 将源顶点索引j和目标顶点索引k添加到相应的列表中。
                        #     source_vertices.append(j)
                        #     target_vertices.append(k)
                        #     # 增加边数计数器，并将时间差的符号值添加到edge_attr列表中。
                        #     num_edge = num_edge + 1
                        #     # 将时间差的符号值添加到 edge_attr 列表中。
                        #     # 如果 time_gap 大于 0，np.sign(time_gap) 返回 1。
                        #     # 如果 time_gap 等于 0，np.sign(time_gap) 返回 0。
                        #     # 如果 time_gap 小于 0，np.sign(time_gap) 返回 -1。
                        #     edge_attr.append(np.sign(time_gap))
                        if 0<=abs(time_gap)<=self.time_edge and id_cond:
                            source_vertices.append(j)
                            target_vertices.append(k)
                            num_edge = num_edge + 1
                            attr = []
                            attr.append(np.sign(time_gap))
                            attr.append(time_gap)
                            attr.append(math.atan2(coordinates[j][1]-coordinates[k][1], coordinates[j][0]-coordinates[k][0]))
                            edge_attr.append(attr)

                        ### connect vertices in the same frame regardless of identity
                        # 查时间差是否小于等于0.0，用于连接同一帧内的顶点。
                        # if abs(time_gap) <= 0.0:
                        # # if 0.0<=abs(time_gap)<=self.short_time_edge and identity[j]!=identity[k]:
                        #     source_vertices.append(j)
                        #     target_vertices.append(k)
                        #     num_edge = num_edge + 1
                        #     edge_attr.append(0)
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
                # 将顶点特征列表x转换为浮点型张量。
                x = torch.tensor(x, dtype=torch.float32)
                # 将源顶点索引和目标顶点索引列表转换为长整型张量，并创建边索引。
                edge_index = torch.tensor([source_vertices, target_vertices], dtype=torch.long)
                # 将边属性列表edge_attr转换为浮点型张量。
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
                # 将顶点标签列表y转换为浮点型张量，并增加一个维度。
                y = torch.tensor(y, dtype=torch.float32)
                y = y.unsqueeze(1)
                #----------------
                with open(self.processed_paths[ct]+ 'identity.txt', 'a') as file:
                    for i in range(len(times)):
                        line = f"{times[i]}\t{identity[i]}\n"  # 使用制表符分隔时间和身份
                        file.write(line)
                

                ## creates the graph data object that stores (features,edges,labels)
                if self.edge_weight == 'fsimy':
                    # 创建图数据对象data，其中包含顶点特征、边索引、顶点标签和边属性。
                    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
                else:
                    data = Data(x=x, edge_index=edge_index, y=y)

                ### save the graph data file with appropriate name; They are named as follows: videoname_1.pt,video_name_2.pt and so on
                # 将图数据对象data保存为.pt文件，文件名由视频名称和计数器组成。
                # torch.save(data, self.processed_paths[ct]+ '_{:03d}.txt'.format(count_gp))
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
        # 遍历特征数据列表中的每个字符串元素，并将其转换为浮点数类型。这样可以得到一个包含浮点数的列表。np.asarray() 将列表转换为 NumPy 数组，返回一个包含浮点数的 NumPy 数组。
        return np.asarray([float(fd) for fd in feature_data])
