import java.util.ArrayList;
import java.util.List;

public class CrossValidation {
  /*
   * Returns the k-fold cross validation score of classifier clf on training data.
   */
  public static double kFoldScore(Classifier clf,
      List<Instance> trainData, int k, int v) {

    List<Instance> trainSet;
    List<Instance> testSet;
    
    // for each fold of k folds, select one as the testSet and all others as trainSet
    double result = 0;
    for (int i = 0; i < k; i++) {
      trainSet = new ArrayList<Instance>();
      testSet = new ArrayList<Instance>();
      for (int j = 0; j < trainData.size(); j++) {
        double index = Math.floor(j / (trainData.size() / k));
        if (i == index) {
          testSet.add(trainData.get(j));
        } else {
          trainSet.add(trainData.get(j));
        }
      }
      clf.train(trainSet, v);
      for (int m = 0; m < testSet.size(); m++) {
        Instance ins = testSet.get(m);
        if (clf.classify(ins.words).label.equals(ins.label)) {
          result++;
        }
      }
    }

    return result / trainData.size();


  }
}
