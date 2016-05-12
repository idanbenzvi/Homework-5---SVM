package hw5;

/**
 * Created by idanbenzvi on 11/05/2016.
 */
public class SVMModel {
    //create an instance of the SVM model class, train it, run backwards processing and return cross-validation-errors
    String training = "ElectionsData_train.txt";
    String testing = "ElectionsData_test.txt";

    Instances data = new Instances(datafile);
    data.setClassIndex(0);
    SVMEval eval = new SVMEval();
    eval.chooseKernel(data);
    Instances workingSet = eval.backwardsWrapper(data, 0.05, 5);
    eval.buildClassifier(workingSet);
    BufferedReader datafile2 = readDataFile(testing);
    Instances dataTest = new Instances(datafile2);
    dataTest.setClassIndex(0);
    Instances subsetOfFeatures =
            eval.removeNonSelectedFeatures(dataTest);
    double avgError = eval.calcAvgError(subsetOfFeatures);
    System.out.println(avgError);
}
