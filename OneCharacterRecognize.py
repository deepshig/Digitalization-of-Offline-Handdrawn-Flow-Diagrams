import numpy as np
from sklearn import svm, metrics
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

sameSize = 16


class OneCharacterRecognize:
    def __init__(self, train_num=70000, test_num=None, debug=False, svm=True, new_clf=False):
        self.savePath = './data/handwritingData.pkl'
        self.debug = debug
        self.svm = svm
        self.new_clf = new_clf
        self.train_num = train_num
        if debug:
            self.train_input = np.loadtxt("./data/feature.txt")
            self.totNum = len(self.train_input)
            self.test_num = test_num if test_num else self.totNum - self.train_num
            self.desired_output = []

            with open('./data/tag.txt', 'r') as f:
                for line in f:
                    self.desired_output.append(line.strip('\n'))

    def get_train_data(self, start, end):
        return self.train_input[start:end], self.desired_output[start:end]

    def get_classifier(self):
        if self.svm and not self.new_clf: return joblib.load(self.savePath)
        train_input, train_output = self.get_train_data(0, self.train_num)
        if self.svm:
            clf = svm.SVC(gamma=0.01)
            clf.fit(train_input, train_output)
            joblib.dump(clf, self.savePath)
            return clf
        else:
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(train_input, train_output)
            return clf

    def test_precision(self):
        clf = self.get_classifier()
        print 'read test file'
        test_input, test_output = self.get_train_data(self.train_num, self.train_num + self.test_num)

        print 'start predict', len(test_input)
        predicted = clf.predict(test_input)

        print(
            "Classification report for classifier %s:\n%s\n" % (
                clf, metrics.classification_report(test_output, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_output, predicted))

    def predict(self, feature):
        clf = self.get_classifier()
        try :
            c = clf.predict(feature)
            return c
        except :
            return 0


if __name__ == '__main__':
    a = OneCharacterRecognize(debug=True)
    a.test_precision()
