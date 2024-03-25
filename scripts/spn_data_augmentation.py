import pandas as pd
import os
import numpy as np
import yaml
import torch

from utils.common_utils import normalize, denormalize
from utils.spn_utils import calc_metrics
from utils.datasets import create_pruning_dataloader


class SPNAugmenter():
    def __init__(self, label_path, state_path, out_path, spn_path, spn_data_path, name_cnt, n, error_thresh, read_df=False, custom=False, device="cuda"):

        self.__label_path = label_path
        self.__state_path = state_path
        self.__out_path = out_path
        self.__spn_path = spn_path
        self.__spn_data_path = spn_data_path
        self.__name_cnt = name_cnt
        self.__n = n
        self.__error_thresh = error_thresh
        self.__read_df = read_df
        self.__custom = custom
        self.__device = device

    def control_augmentation(self) -> None:
        """Controls the augmentation process.
        """

        if self.__custom:

            if self.__read_df:
                self.all_samples = pd.read_pickle("/home/blanka/ERLPruning/sandbox/train54_df.pkl")
            else:
                self.all_samples = self.load_samples()
                self.all_samples.to_pickle("/home/blanka/ERLPruning/sandbox/train54_df.pkl")

            self.target_samples = self.collect_custom_target_samples(self.all_samples)
            del self.all_samples

        else:
            self.target_samples = self.collect_target_samples(self.__error_thresh, self.__spn_data_path, self.__spn_path)

        self.augmented_samples = self.add_gauss_niose(self.target_samples, self.__n)
        del self.target_samples
        self.save_augmented_samples(self.augmented_samples, self.__out_path, self.__name_cnt)
        del self.augmented_samples

    def load_samples(self) -> pd.DataFrame:
        """Loads all samples from the label and data folders and saves them in Pandas DataFrames.
        :return (pd.DataFrame): all samples (data and label)
        """
        samples_df = pd.DataFrame(columns=['state', 'label'])
        state_files = os.listdir(self.__state_path)
        for file in state_files:
            state_file = os.path.join(self.__state_path, file)
            label_file = os.path.join(self.__label_path, file)
            state = np.loadtxt(state_file)
            label = np.loadtxt(label_file)
            new_row = pd.Series({"state": state, "label": label})
            samples_df = pd.concat([samples_df, new_row.to_frame().T], ignore_index=True)

        return samples_df

    def collect_custom_target_samples(self, samples_df) -> pd.DataFrame:
        """ Saves the problematic samples to a Pandas DataFrame.
        :return (pd.DataFrame): problematic samples that need to be augmented
        """
        target_df = pd.DataFrame(columns=['state', 'label'])

        for index, row in samples_df.iterrows():
            state, label = np.frombuffer(row['state']).reshape(-1, 7), np.frombuffer(row['label']).reshape(-1, 4)
            kernel = state[:, 1]
            alpha_seq = np.around(denormalize(state[:, 0], 0, 2.2), 1)

            last_pruned_layer = np.max(np.where(kernel > -1))
            n_0 = np.count_nonzero(alpha_seq[:last_pruned_layer] == 0.0)
            n_01 = np.count_nonzero(alpha_seq == 0.1)

            condition1 = (last_pruned_layer < 10) and (alpha_seq[0] == 0.1) #19
            condition2 = (n_0 > 90) and ((alpha_seq[105] > 1.8) or (alpha_seq[104] > 1.8 ) or (alpha_seq[103] > 1.8 ) or (alpha_seq[102] > 1.8 )) #34
            condition3 = (last_pruned_layer > 38 and last_pruned_layer < 50) and (alpha_seq[last_pruned_layer] > 1.8)
            #condition4 = (last_pruned_layer > 100) and (n_0 > 0.99*last_pruned_layer)
            # condition5 = n_01 > 2

            if condition1 or condition2 or condition3: # or condition4 or condition5:
                new_row = pd.Series({"state": state, "label": label})
                target_df = pd.concat([target_df, new_row.to_frame().T], ignore_index=True)

        print(f"Number of problematic samples: {target_df.shape[0]}")

        return target_df

    def collect_target_samples(self, error_threshold, data_path, spn_path) -> pd.DataFrame:
        """ Saves the problematic samples based on the error between the predcted and ground truth values to a Pandas DataFrame.
        :return (pd.DataFrame): problematic samples that need to be augmented
        """
        target_df = pd.DataFrame(columns=['state', 'label'])

        # Validation data
        print("olaa", data_path)
        with open(data_path, "r") as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
        data_path = data_dict['train_state']
        label_path = data_dict['train_label']
        dataloader, _ = create_pruning_dataloader(data_path, label_path, batch_size=1)

        # Load SPN model
        ckpt_spn = torch.load(spn_path)
        model = ckpt_spn['model']
        model.eval()

        for batch_i, (data, label_gt) in enumerate(dataloader):
            input_data = data.type(torch.float32).to(self.__device)
            input_data = torch.cat((input_data[:, :, :5], input_data[:, :, -1:]), dim=2)
            input_label_gt = label_gt.type(torch.float32).to(self.__device)

            prediction = model(input_data)
            prediction = prediction.permute(0, 1)  # --> [batch_size, n_lables, sequence_length]

            # metrics_dperf = calc_metrics(input_label_gt[:, 1], prediction[:, 1], margin=0.02)
            # max_error_dperf = metrics_dperf[3]

            gt_norm = denormalize(input_label_gt[:, 1], 0, 1).item()
            pred_norm = denormalize(prediction[:, 1], 0, 1).item()
            abs_error_dperf = np.abs(gt_norm - pred_norm)

            if abs_error_dperf > error_threshold:
                print(f"Error = {abs_error_dperf}, gt = {gt_norm}, pred = {pred_norm}, {denormalize(data[0, :, 0], 0.0, 2.2)}")
                new_row = pd.Series({"state": data.numpy(), "label": label_gt.numpy()})
                target_df = pd.concat([target_df, new_row.to_frame().T], ignore_index=True)

        print(f"Number of problematic samples: {target_df.shape[0]}")

        return target_df

    def add_gauss_niose(self, samples_df, n) -> pd.DataFrame:
        """ Adds gauss noise to the problematic samples.
        :return (pd.DataFrame): augmented samples
        """
        augmented_df = pd.DataFrame(columns=['state', 'label'])

        for index, row in samples_df.iterrows():
            state, label = np.frombuffer(row['state']).reshape(-1, 7), np.frombuffer(row['label']).reshape(-1, 4)

            for i in range(n):
                noise = np.random.normal(0, 0.001, state.shape[0])
                new_state = state
                new_state[:, 0] += noise

                new_row = pd.Series({"state": new_state, "label": label})
                augmented_df = pd.concat([augmented_df, new_row.to_frame().T], ignore_index=True)

        print(augmented_df.shape)
        return  augmented_df

    def save_augmented_samples(self, samples_df, out_path, name_cnt) -> None:
        """Saves the sugmented samples with names starting form name_cnt.
        """
        label_path = os.path.join(out_path, "labels")
        state_path = os.path.join(out_path, "states")

        for index, row in samples_df.iterrows():
            state, label = np.frombuffer(row['state']).reshape(-1, 7), np.frombuffer(row['label']).reshape(-1, 4)
            np.savetxt(os.path.join(state_path, f"{name_cnt}.txt"), state)
            np.savetxt(os.path.join(label_path, f"{name_cnt}.txt"), label)
            name_cnt += 1


if __name__ == "__main__":

    state_path = "/data/blanka/DATASETS/SPN/training53/states"
    label_path = "/data/blanka/DATASETS/SPN/training53/labels"
    out_path = "/data/blanka/DATASETS/SPN/training53/augmented"
    spn_path = "/data/blanka/ERLPruning/runs/SPN/manual_transformer_all53_08_augm/weights/best.pt"
    spn_data_path = '/home/blanka/ERLPruning/data/spndata.yaml'

    name_cnt = 40050 + 350 + 250 + 200
    n = 50
    error_thresh = 0.15
    read_df = True

    augmenter = SPNAugmenter(label_path, state_path, out_path, spn_path,spn_data_path, name_cnt, n, error_thresh, read_df)
    augmenter.control_augmentation()
