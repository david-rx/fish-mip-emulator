from sklearn import tree
from sklearn import metrics

from emulator.dataloading.macroecological import INPUTS_PATH, INPUTS_PATH_INTPP, OUTPUTS_PATH, MacroecologicalDataLoader, TEST_INPUTS_PATH, INPUTS_PATH_INTPP, TEST_OUTPUTS_PATH

def train() -> None:
    dataloader = MacroecologicalDataLoader(inputs_path = INPUTS_PATH, outputs_path = OUTPUTS_PATH, inputs_path_intpp = INPUTS_PATH_INTPP)
    train_features, train_labels, eval_features, eval_labels = dataloader.load_train_eval()
    clf = tree.DecisionTreeRegressor()
    clf.fit(train_features, train_labels)
    eval_predictions = clf.predict(eval_features)
    print(metrics.mean_squared_error(eval_predictions, eval_labels))


if __name__ == "__main__":
    train()