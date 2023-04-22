import pandas as pd
import os
import numpy as np

from utils.LR_utils import normalize, denormalize

class SPNAugmenter():
    def __init__(self, label_path, state_path, out_path, name_cnt, n, read_df=False):

        self.__label_path = label_path
        self.__state_path = state_path
        self.__name_cnt = name_cnt
        self.__n = n
        self.__read_df = read_df
        self.__out_path = out_path

    def control_augmentation(self) -> None:
        """Controls the augmentation process.
        """
        if self.__read_df:
            self.all_samples = pd.read_pickle("/home/blanka/ERLPruning/sandbox/train54_df.pkl")
        else:
            self.all_samples = self.load_samples()
            self.all_samples.to_pickle("/home/blanka/ERLPruning/sandbox/train54_df.pkl")

        self.target_samples = self.collect_target_samples(self.all_samples)
        del self.all_samples
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

    def collect_target_samples(self, samples_df) -> pd.DataFrame:
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
    name_cnt = 33350
    n = 100
    read_df = True

    augmenter = SPNAugmenter(label_path, state_path, out_path, name_cnt, n, read_df)
    augmenter.control_augmentation()
