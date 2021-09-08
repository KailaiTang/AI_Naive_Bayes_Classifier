import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {

  List<Instance> trainData;
  int v;
  Map<String, Integer> pos_map;
  Map<String, Integer> neg_map;

  /**
   * Trains the classifier with the provided training data and vocabulary size
   */
  @Override
  public void train(List<Instance> trainData, int v) {

    // store trainData and v into the fields
    this.trainData = trainData;
    this.v = v;

    // First, calculate the documents and words counts per label and store them.
    int docPos =
        getDocumentsCountPerLabel(trainData).get(Label.POSITIVE);
    int docNeg =
        getDocumentsCountPerLabel(trainData).get(Label.NEGATIVE);

    // Then, for all the words in the documents of each label, count the number of occurrences of each
    // word.
    int wordPos =
        getWordsCountPerLabel(trainData).get(Label.POSITIVE);
    int wordNeg =
        getWordsCountPerLabel(trainData).get(Label.NEGATIVE);

    // e.g.
    // Assume m_map is the map that stores the occurrences per word for positive documents
    pos_map = new HashMap<String, Integer>();
    neg_map = new HashMap<String, Integer>();

    for (Instance i : trainData) {
      // if the instance has positive label
      if (i.label.equals(Label.POSITIVE)) {
        for (String word : i.words) {
          // if the word has not appeared before, put it into the m_map
          if (pos_map.get(word) == null) {
            pos_map.put(word, 1);
          }
          // if the word has appeared before, increment its occurrence count
          else {
            pos_map.put(word, pos_map.get(word) + 1);
          }
        }
      }
      // else the instance has negative label
      else {
        for (String word : i.words) {
          // if the word has not appeared before, put it into the m_map
          if (neg_map.get(word) == null) {
            neg_map.put(word, 1);
          }
          // if the word has appeared before, increment its occurrence count
          else {
            neg_map.put(word, neg_map.get(word) + 1);
          }
        }
      }
    }

  }

  /*
   * Counts the number of words for each label
   */
  @Override
  public Map<Label, Integer> getWordsCountPerLabel(
      List<Instance> trainData) {

    // count number of words with positive and negative label inside the whole trainData
    int pos = 0;
    int neg = 0;
    for (Instance i : trainData) {
      if (i.label.equals(Label.POSITIVE)) {
        pos += i.words.size();
      }
      if (i.label.equals(Label.NEGATIVE)) {
        neg += i.words.size();
      }
    }
    // put the result into the hash map
    Map<Label, Integer> result = new HashMap<Label, Integer>();
    result.put(Label.POSITIVE, pos);
    result.put(Label.NEGATIVE, neg);
    return result;
  }

  /*
   * Counts the total number of documents for each label
   */
  @Override
  public Map<Label, Integer> getDocumentsCountPerLabel(
      List<Instance> trainData) {

    // count number of documents with positive and negative label inside the whole trainData
    int pos = 0;
    int neg = 0;
    for (Instance i : trainData) {
      if (i.label.equals(Label.POSITIVE)) {
        pos++;
      }
      if (i.label.equals(Label.NEGATIVE)) {
        neg++;
      }
    }
    // put the result into the hash map
    Map<Label, Integer> result = new HashMap<Label, Integer>();
    result.put(Label.POSITIVE, pos);
    result.put(Label.NEGATIVE, neg);
    return result;
  }

  /**
   * Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
   */
  private double p_l(Label label) {
    // Calculate the probability for the label. No smoothing here.
    // Just the number of label counts divided by the number of documents.
    int count = getDocumentsCountPerLabel(this.trainData).get(label);
    return count / (double) this.trainData.size();
  }

  /**
   * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE)
   * or P(word|NEGATIVE)
   */
  private double p_w_given_l(String word, Label label) {
    // Calculate the probability with Laplace smoothing for word in class(label)
    // P(w|l) = [cl(w) + 1] / [|V| + (for all v, Cl(v))]
    // cl(w) is the number of times of tokens, given word w and label l
    // |V| is size of dictionary
    // v refers a vocabulary in the Vocabulary category V

    // 1. find cl(w)
    int clw = 0;
    if (label.equals(Label.POSITIVE)) {
      if (pos_map.get(word) != null) {
        clw = pos_map.get(word);
      }
    } else {
      if (neg_map.get(word) != null) {
        clw = neg_map.get(word);
      }
    }

    // 2. find sum of cl(v) for all v in V
    // since v indicates all words, so this sum is just the number of words under label
    int sumClv = getWordsCountPerLabel(this.trainData).get(label);

    // 3. Calculate p(w|l)
    // System.out.println(label + ": "+((double) (clw + 1) / (double) (this.v + sumClv)));
    return (double) (clw + 1) / (double) (this.v + sumClv);
  }

  /**
   * Classifies an array of words as either POSITIVE or NEGATIVE.
   */
  @Override
  public ClassifyResult classify(List<String> words) {

    double pPos = p_l(Label.POSITIVE);
    double pNeg = p_l(Label.NEGATIVE);

    // Sum up the log probabilities for each word in the input data, and the probability of the label
    double sumPos = Math.log(pPos);
    double sumNeg = Math.log(pNeg);
    for (String word : words) {
      sumPos += Math.log(p_w_given_l(word, Label.POSITIVE));
      sumNeg += Math.log(p_w_given_l(word, Label.NEGATIVE));
    }

    // Generate the Map of the ClassifyResult
    Map<Label, Double> logProbPerLabel = new HashMap<Label, Double>();
    logProbPerLabel.put(Label.POSITIVE, sumPos);
    logProbPerLabel.put(Label.NEGATIVE, sumNeg);

    // Set the label to the class with larger log probability
    ClassifyResult cr = new ClassifyResult();
    cr.label = sumPos > sumNeg ? Label.POSITIVE : Label.NEGATIVE;
    cr.logProbPerLabel = logProbPerLabel;


    return cr;
  }


}
