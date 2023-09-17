try:
    from ctimageio import readImage
except:
    from .ctimageio import readImage
from glob import glob
import numpy as np
import cv2

class MedSegSimpleReader:
    def __init__(self, folder_path, isFlip=False):
        img_folder = folder_path+"img"
        mask_folder = folder_path+"mask"
        self.img_paths = glob(img_folder+"/*")
        self.mask_paths = glob(mask_folder+"/*")
        self.isFlip = isFlip

    def __iter__(self): #跌代初始化
        self.iter_index = 0
        return self

    def __next__(self): #跌代給for loop
        iter_index = self.iter_index
        if iter_index > len(self)-1:
            raise StopIteration
        self.iter_index += 1
        return self[iter_index]

    def __getitem__(self, index): #取得單一病人的 (Image, Mask) -> 使用object[index]取得
        img_array = readImage(self.img_paths[index])
        mask_array = readImage(self.mask_paths[index])
        if self.isFlip:
            return (np.flip(img_array), np.flip(mask_array))
        return (img_array, mask_array)

    def __len__(self): #取得病人的數量 -> 使用len(object)取得
        return len(self.img_paths)

class MedSegReader2D(MedSegSimpleReader):
    def __init__(self, folder_path, isFlip=False, preprocessing=None):
        super().__init__(folder_path, isFlip)
        self.__sw_return_type = "list"
        self.__sw_window_size = 3

        self.ct_count = 0 #CT總數
        self.ct_index_map = [] #紀錄index對應的(patient index, ct index)

        #先將所有資料載入 Memory
        self.data = [self[i] for i in range(len(self))]
        if preprocessing: #前置處理
            self.data = preprocessing(self.data)

        self.patient_start_index = [0] #紀錄patient ct的起始index
        for i, datum in enumerate(self.data):
            img_array, _ = datum
            count = img_array.shape[0]
            self.ct_count += count #累加CT總數
            self.patient_start_index.append(self.ct_count) #紀錄patient起始index
            self.ct_index_map += [(i,j) for j in range(count)]

    def getPtCTIndices(self, patient_index): #取得patient的CT indices 註: Pt -> patient
        start_index = self.patient_start_index[patient_index]
        end_index = self.patient_start_index[patient_index+1]
        return [i for i in range(start_index, end_index)]

    def getPtPartIndices(self, patient_index, part_num, spilt_part_num): #取得patient的Part CT indices
        ct_idx = self.getPtCTIndices(patient_index)
        part_num -= 1
        step = len(ct_idx)//spilt_part_num
        if part_num != spilt_part_num-1: #取出該Part的CT index
            indices = ct_idx[part_num*step:(part_num+1)*step]
        else: #最後一Part，取出剩下所有CT index
            indices = ct_idx[part_num*step:]
        return indices

    def getBatch(self, indices):
        ct_index_map = [self.ct_index_map[index] for index in indices]
        X = []
        y = []
        #建立回傳的X與y array
        for patient_index, ct_index in ct_index_map:
            img_array, mask_array = self.data[patient_index]
            X.append(img_array[ct_index])
            y.append(mask_array[ct_index])
        return np.array(X), np.array(y)

    def setSWReturnType(self, type_name):
        ## return_type = list or stack
        if type_name != "list" and type_name != "stack":
            raise NameError("Invalid return_type option, only can select list or stack")
        self.__sw_return_type = type_name

    def setSWWindowSize(self, window_size):
        self.__sw_window_size = window_size

    def getSWBatch(self, indices):
        ct_index_map = [self.ct_index_map[index] for index in indices]
        
        step = (self.__sw_window_size-1)//2
        if self.__sw_return_type == "stack":
            X = []
            y = []
            #建立回傳的X與y array
            for patient_index, ct_index in ct_index_map:
                img_array, mask_array = self.data[patient_index]
                select_array = list(img_array[ct_index-step : ct_index+step+1])
                try:
                    stack_array = select_array.pop(0)
                except:
                    continue
                
                for _ in range(len(select_array)):
                    stack_array = np.dstack((stack_array, select_array.pop(0)))
                X.append(stack_array)
                y.append(mask_array[ct_index])
            return np.array(X), np.array(y)

        elif self.__sw_return_type == "list":
            X = [[] for _ in range(self.__sw_window_size)]
            y = []
            #建立回傳的X與y array
            for patient_index, ct_index in ct_index_map:
                img_array, mask_array = self.data[patient_index]
                for i, idx in enumerate(range(ct_index-step, ct_index+step+1)):
                    X[i].append(img_array[idx])
                y.append(mask_array[ct_index])
            X = [np.array(x) for x in X]
            return X, np.array(y)
        
    def getCoordBatch(self, indices):
        ct_index_map = [self.ct_index_map[index] for index in indices]
        X = []
        X_loc = []
        y = []
        #建立回傳的X與y array
        for patient_index, ct_index in ct_index_map:
            img_array, mask_array = self.data[patient_index]
            X.append(img_array[ct_index])
            X_loc.append(np.array([ct_index, img_array.shape[0]]))
            y.append(mask_array[ct_index])
        return [np.array(X), np.array(X_loc)], np.array(y)
    
    def getLocBatch(self, indices):
        ct_index_map = [self.ct_index_map[index] for index in indices]
        X = []
        X_loc = []
        y = []
        #建立回傳的X與y array
        for patient_index, ct_index in ct_index_map:
            img_array, mask_array = self.data[patient_index]
            X.append(img_array[ct_index])
            X_loc.append(ct_index/img_array.shape[0])
            y.append(mask_array[ct_index])
        return [np.array(X), np.array(X_loc)], np.array(y)

    def removeNoLiverIndices(self, indices):
        ct_index_map = [self.ct_index_map[index] for index in indices]
        return_indices = []
        for ct_map, index in zip(ct_index_map, indices):
            patient_index, ct_index = ct_map
            mask_array = self.data[patient_index][1] #取得mask
            label_num = len(np.unique(np.argmax(mask_array[ct_index], axis=-1))) #此CT有多少label
            if label_num != 1: #有label 0之外的label -> 要保留
                return_indices.append(index)
        return return_indices

    def getArray(self, index): #取得index的CT影像與mask
        patient_index, ct_index = self.ct_index_map[index]
        img_array, mask_array = self.data[patient_index] #讀patient檔案
        return (img_array[ct_index], mask_array[ct_index])

    def getCTCount(self): #取得所有CT數量
        return self.ct_count



if __name__=="__main__":
    folder_path = "../../Liver_Data/MedSeg/Liver/"
    # Test Simple Reader
    ms_simple_reader = MedSegSimpleReader(folder_path)
    print(len(ms_simple_reader))
    img, mask = ms_simple_reader[30]
    print(img.shape, mask.shape)

    ## Calculate Data length
    # length = []
    # for datum in ms_simple_reader:
    #     img, mask = datum
    #     length.append(img.shape[0])
    # print(length)
    # print(sum(length)/len(length))

    # Test Reader
    ms_reader = MedSegReader2D(folder_path)
    ct, mask = ms_reader.getArray(2000)
    print(len(ms_reader.patient_start_index), ms_reader.patient_start_index)
    print(ms_reader.getPtCTIndices(49))
    X, y = ms_reader.getBatch([5,8,1000,2000])
    print(ms_reader.getCTCount())
    print(X.shape, y.shape)

    X, y = ms_reader.getSWBatch([5,8,1000,2000])
    print(len(X), [x.shape for x in X], y.shape)