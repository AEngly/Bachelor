import numpy as np
import pandas as pd

class Performance_Evaluator:

    def Counter(self, y_true, y_predicted):

        t_p = 0
        t_n = 0
        f_p = 0
        f_n = 0

        if len(y_true) != len(y_predicted):

            print("ERROR 1: The inputs should have same length!")

        else:

            for i in range(len(y_true)):
                if (y_true[i] == self.flag_positive) and (y_predicted[i] == self.flag_positive):
                    t_p += 1
                elif (y_true[i] == self.flag_negative) and (y_predicted[i] == self.flag_negative):
                    t_n += 1
                elif (y_true[i] == self.flag_positive) and (y_predicted[i] == self.flag_negative):
                    f_n += 1

                elif (y_true[i] == self.flag_negative) and (y_predicted[i] == self.flag_positive):
                    f_p += 1
                else:
                    continue
            self.true_positives = t_p
            self.true_negatives = t_n
            self.false_positives = f_p
            self.false_negative = f_n
            self.total = t_p + t_n + f_p + f_n


    def Update_Binary(self):

        # Update raw parameters
        self.Counter(self.y_true, self.y_predicted)

        try:

            self.accuracy = self.Accuracy(self.true_positives, self.true_negatives, self.total)
            self.precision = self.Precision(self.true_positives, self.false_positives)
            self.recall = self.Recall(self.true_positives, self.false_negatives)
            self.f_measure = self.F_measure(self.precision, self.recall)
            self.false_negative_rate = self.False_negative_rate(self.false_negatives, self.true_positives)
            self.false_positive_rate = self.False_positive_rate(self.false_positives, self.true_positives)

        except ZeroDivisionError:

            print("Something is off!")

    def Update_Multi(self):

        t = (self.y_true).to_numpy()
        p = self.y_predicted

        self.summary_matrix = pd.DataFrame(np.zeros((23,23)), columns=self.y_true.columns, index=self.y_true.columns)

        for i in range(np.shape(t)[0]):

            t_arg = t[i,:].argmax()
            p_arg = p[i,:].argmax()

            self.summary_matrix.iloc[p_arg, t_arg] += 1


    def __init__(self, y_true, y_predicted):

        # The number 1 is used to define a positive sample
        # The number 0 is used to define a negative sample

        self.flag_positive = 1
        self.flag_negative = 0

        self.y_true = y_true
        self.y_predicted = y_predicted

        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total = 0

        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_measure = 0
        self.false_negative_rate = 0
        self.false_positive_rate = 0
        self.multi_class = False

        self.Update_Binary()

    def __init__(self, y_true, y_predicted, multi_class):

        # The number 1 is used to define a positive sample
        # The number 0 is used to define a negative sample

        self.flag_positive = 1
        self.flag_negative = 0

        self.y_true = y_true
        self.y_predicted = y_predicted

        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total = 0

        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_measure = 0
        self.false_negative_rate = 0
        self.false_positive_rate = 0
        self.multi_class = multi_class
        self.summary_matrix = None

        self.Update_Multi()

    def Accuracy(self, t_p, t_n, total):
        return (t_p + t_n)/total

    def Precision(self, t_p, f_p):
        return t_p/(t_p + f_p)

    def Recall(self, t_p, f_n):
        return t_p/(t_p + f_n)

    def F_measure(self, precision, recall):
        return 2*precision*recall/(precision + recall)

    def False_negative_rate(self, f_n, t_p):
        return f_n/(t_p + f_n)

    def False_positive_rate(self, f_p, t_p):
        return f_p/(f_p + t_p)

    def Matrix_Summary(self):

        if not(self.multi_class):

            df = pd.DataFrame(np.array([[self.true_positives,self.false_positives],[self.false_negatives,self.true_negatives]]),
                                columns=["positive", "negative"], index=["positive", "negative"])
            df.style.format('{:.2f}')

            print("-"*14 + " [PREDICTED / ACTUAL] " + "-"*14 + "\n\n")
            print(df)

        if (self.multi_class):

            print("-"*14 + " [PREDICTED / ACTUAL] " + "-"*14 + "\n\n")
            print(self.summary_matrix)

            return self.summary_matrix



    def Summary(self):

        print("-"*20 + " SUMMARY " + "-"*20 + "\n\n")
        print("The accuracy is: {}".format(self.accuracy))
        print("The precision is: {}".format(self.precision))
        print("The recall is: {}".format(self.recall))
        print("The f_measure is: {}".format(self.f_measure))
        print("The false_negative_rate is: {}".format(self.false_negative_rate))
        print("The false_positive_rate is: {}".format(self.false_positive_rate))
        print("\n\n" + "-"*20 + "-"*9 + "-"*20)
