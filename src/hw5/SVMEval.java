package hw5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.HashSet;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SVMEval {

    private SMO mySMO = new SMO();
    private final int C_NUM_FOLDS = 3;
    private final int C_POLY_KERNEL_MIN = 2;
    private final int C_POLY_KERNEL_MAX = 4;
    private final int C_RBF_KERNEL_MIN = 2;
    private final int C_RBF_KERNEL_MIN = -10;


	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

    public void buildClassifier(Instances instances) throws Exception{
            //builds the classifier using Weka's built in SMO SVM classifier module
            mySMO.buildClassifier(instances);


        }

//    Implement a method backwardsWrapper which performs feature selection (explanation after
//                                                                          programming instructions):
//    Input: threshold, min number of attributes, instances
//    Action: performs the backwards wrapper feature selection algorithm as explained
//    above.
//            Output: New Instances object with the chosen subset of the original features, indices of
//    chosen features.
//    Note: After using this method, note that if you use other Instances objects and you want
//    to use the subset of features you should remove the features that backwardsWrapper
//    chose to remove
    public void backwardsWrapper(double T, int K, Instances instances){
        double error_diff = 0;
        int i_minimal = 1;
        double original_error = calcCrossValidationError(instances);
        double minimal_error;

        HashSet<Attribute> feature_set = new HashSet<Attribute>();

        do{
            i_minimal =1;
            //calculate cross validation error without the first feature in the attributes array
            minimal_error = calcCrossValidationError(removeFeature(instances,0));
            
            minimal_error = the SVM cross validation error on the set of features S without {x_1}
                For each i in {2…|S|} {
                    new_error = the SVM cross validation error on the set of features S without {x_i}
                    if (new_error < minimal_error) {
                        minimal_error = new_error;
                        i_minimal = i;
                    }
                }
                error_diff = minimal_error – original_error
                If (error_diff < threshold t){
                    Remove X_i_minimal from S
                }



        }while(feature_set.size()<K || error_diff > T);


    }

    private Instances removeFeature(Instances instances,int attributeIx) {
        try {
            //remove the feature at the given index and return the instances without it
            Remove remove = new Remove();
            remove.setInputFormat(instances);
            String[] options = new String[2];
            options[0] = "-R";
            options[1] = Integer.toString(attributeIx + 1);
            remove.setOptions(options);
            Instances workingSet = Filter.useFilter(instances,remove);

            return workingSet;
        }
        catch(Exception e) {
            System.out.println("COULD NOT REMOVE THE given attribute");
            return null;
        }

    }


//    Input: Instances.
//            Output: The number of prediction mistakes the classifier makes divided by the number
//    of instances.
//
    private double calcAvgError(Instances instances){

        //enumration of instances to go over
        Enumeration<Instance> instEnum = instances.enumerateInstances();

        int countErrors = 0;

        while(instEnum.hasMoreElements())
             {
                 try{
                     Instance currentInstance = instEnum.nextElement();
                     if(currentInstance.classValue()!=mySMO.classifyInstance(currentInstance))
                         countErrors++;
                 }
                 catch (Exception e){
                     System.out.println("could not classify instance - an error has occured");
                 }

        }

        return (double) countErrors / instances.numInstances();
    }

    /**
     * Implement a method named chooseKernel which chooses the best kernel.
     Input: Instances
     Action: chooses best kernel from the options: 13 possible RBF kernels with parameter
     gamma 2^i for -10<=i<=2, 3 possible polynomial kernels of degree i for 2<=i<=4. The chosen
     kernel is the kernel that yields the best cross validation error for the SVM model that uses it.
     Output: the method sets the kernel of your SVM classifier.
     * @param instances
     * @return kernel of the SVM Classifier
     */
    public void chooseKernel(Instances instances){
        //set kernel to RBF kernel
        RBFKernel kernel = new RBFKernel();
        mySMO.setKernel(kernel);

        int[] foldsIndices = foldIndices(instances,C_NUM_FOLDS);

        for(int i = C_RBF_KERNEL_MIN ; C_RBF_KERNEL_MAX <= 2; i++){
            //set the kernal gamma value
            kernel.setGamma(Math.pow(2,i));

            //get the instances in the folds and test them
            calcCrossValidationError(foldInstances);



            //classify the instances loaded in order to get the best cross-validation error
        }

        //polynomial kernel
        for(int i = C_POLY_KERNEL_MIN ; i <= C_POLY_KERNEL_MAX ; i++){
            PolyKernel kernel2 = new PolyKernel();
            kernel2.setExponent(i);

            calcCrossValidationError(foldInstances);
        }

    }


    public double calcCrossValidationError(Instances instances){

        //calculate the error on all possible folds created using the instances

        //get splitting indices for folding
        int[] subsetIndices = foldIndices(instances,C_NUM_FOLDS);
        Instances[] instArray;

        double cvError = 0;

        //iterate over all folds
        for(int foldix = 0 ; foldix <=C_NUM_FOLDS-1 ; foldix++) {
            //divide instances into 2 Instances objects, one containing the fold, the other the rest of the instances (2/3 remaining)
            instArray = getFoldInstances(subsetIndices[foldix],subsetIndices[foldix+1],instances);

            //calculate avg error for current fold
            cvError += calcAvgError(instArray[1],instArray[0]);

            //for each instance of  the smaller group, locate the K nearest neighbors according to the selected
            //function. After doing so, classify according to these neighbors.
            //keep classification
        }

        //divide the sum of errors by the number of folds to calculate the cross validation error
        cvError /= M_FOLD_NUM;
        m_calcTimeAvg = calcTimeAvg / M_FOLD_NUM;

        return cvError;
    }






	public static void main(String[] args){
        try {
		    //load datasets according to test and train split instructions
            loadData("file.txt");
            loadData("file2.txt");

            //find the best kernel possible using chooseKernel

            //perform feature selection

            //build the classifier

            //only on the test set

            //remove unnecessary features

            //calculate average error using the trained classifier

            //print average error

            String training = "ElectionsData_train.txt";
            String testing = "ElectionsData_test.txt";

            Instances trainingData = loadData(training);
            Instances testData = loadData(testing);
            trainingData.setClassIndex(0);
            testData.setClassIndex(0);

            SVMEval eval = new SVMEval();



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
        catch (Exception c) {
            System.out.println("failed building classifier");
        }
            //instantiate the SVM

		//build the classifier

		//cross-validation


	}


    // assistive methods - folding

    /**
     * divide the instances into k folds, of equal size. possibly shuffle instances if required
     * @param instances
     * @return
     */
    private int[] foldIndices(Instances instances, int foldNum){
        //divide to the nearest integer value possible (last set might be smaller than int value by remainder)
        int instCount = (int) Math.floor( (double) instances.numInstances() / foldNum); //rounded version of course

        int[] foldIndexArray = new int[M_FOLD_NUM+1];
        int multiplier = 1;
        int ix = 0;
        foldIndexArray[0] = 0;

        //keep only the indices of the instances that are cutoff points between sets (each one composing of 10%)
        while(multiplier*instCount <= instances.numInstances()){
            foldIndexArray[multiplier] = multiplier * instCount;
            multiplier++;

            if(multiplier==10){
                foldIndexArray[10] = instances.numInstances();
                break;
            }
        }

        return foldIndexArray;
    }

    /**
     * return instances that compose of the data outside the indices and the remaining instances
     * in the second index of the instances array.
     * @param start
     * @param end
     * @param instances
     * @return instancesArray composing of 9/10 division ratio between instances according to instructions above
     */
    private Instances[] getFoldInstances(int start,int end,Instances instances){
        Enumeration<Instance> instEnum = instances.enumerateInstances();
        int ix = 0;
        Instances[] instancesArray = new Instances[2] ;
        instancesArray[0] = new Instances(instances,instances.numInstances());
        instancesArray[1] = new Instances(instances,instances.numInstances());
        //as long as there are more elements to divide - keep on going
        while(instEnum.hasMoreElements()){
            if(ix<start || ix > end)
                instancesArray[0].add(instEnum.nextElement()); //add to the first instances group (majority of 90%)
            else
                instancesArray[1].add(instEnum.nextElement()); //the smaller group (in our case will be 10%)
            ix++; // increment index

        }

        return instancesArray;
    }


}
