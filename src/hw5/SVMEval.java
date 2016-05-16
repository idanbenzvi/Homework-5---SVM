package hw5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

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
    private final int C_RBF_KERNEL_MAX = 2;
    private final int C_RBF_KERNEL_MIN = -10;

    public double m_bestError = Double.MAX_VALUE;
    public int m_bestKernel = 0;
    private final int RBF = 0;
    private final int POLY = 1;
    public double m_bestRBFKernelValue = 0;
    public int m_bestPolyKernelValue = 0;

    private ArrayList<Integer> m_removed_features ;

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
    public Instances backwardsWrapper(Instances instances,double T, int K){
        double error_diff = 0;
        int i_minimal=0;
        int removedIx = 0;

        String[] featureArray = new String[instances.numAttributes()];

        //init the the removed feature field, in which we will retain the removed attributes
        //the indices can only be used in the order they are stored since each index refelcts its corresponding attribute
        //after previous entries have been removed before it.
        m_removed_features = new ArrayList<Integer>();

        double original_error = calcCrossValidationError(instances);
        double minimal_error;

        do{
            //reset the i value each time the while loop iterates in order to restart the attribute search process
            i_minimal = 0;

            //calculate cross validation error without the first feature in the attributes array
            minimal_error = calcCrossValidationError(removeFeature(instances,i_minimal));

            //iterate over all possible attributes in order to find the attribute which will result in the minimal
            // cross validation error.
               for(int i = 1 ; i <= instances.numAttributes() ; i++) {
                    double new_error = calcCrossValidationError(removeFeature(instances,i));

                   if (new_error < minimal_error) {
                        minimal_error = new_error;
                        i_minimal = i;
                   }
                }

            //after finding the index of the minimal attribute, evaluate if the difference is small enough (smaller than
            //the threshold)
            error_diff = minimal_error - original_error;

            if (error_diff < T){
                //store the index of the current removed feature
                    m_removed_features.add(i_minimal);
                //remove the feature from the instances
                    instances = removeFeature(instances,i_minimal);
                }
        }while(instances.numAttributes()==K || error_diff > T);

        //return the instances without the removed features
        return instances;
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
            System.out.println("COULD NOT REMOVE THE ATTRIBUTE");
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
    public void chooseKernel(Instances instances) throws Exception{
        //error from cross fold validation process
        double error = Double.MAX_VALUE;

        //set kernel to RBF kernel
        RBFKernel kernel = new RBFKernel();
        PolyKernel kernel2 = new PolyKernel();


        //set the first kernel - RBF
        mySMO.setKernel(kernel);


        for(int i = C_RBF_KERNEL_MIN ; C_RBF_KERNEL_MAX <= 2; i++){

                //set the kernal gamma value
                kernel.setGamma(Math.pow(2, i));

                //get the instances in the folds and test them
                error = calcCrossValidationError(instances);

                //retain the best kernel result and set it as the kernel for our hypothesis
                if (error < m_bestError) {
                    m_bestError = error;
                    m_bestKernel = RBF;
                    m_bestRBFKernelValue = Math.pow(2, i);
                }

                //classify the instances loaded in order to get the best cross-validation error
            }


        //set the kernel to the polynomial kernel
        mySMO.setKernel(kernel2);

        //polynomial kernel
        for(int i = C_POLY_KERNEL_MIN ; i <= C_POLY_KERNEL_MAX ; i++){

            kernel2.setExponent(i);

            error = calcCrossValidationError(foldsInstances);

            //retain the best kernel result and set it as the kernel for our hypothesis
            if(error < m_bestError){
                m_bestError = error;
                m_bestKernel = POLY;
                m_bestRBFKernelValue = i;
            }

        }
    }


    public double calcCrossValidationError(Instances instances){

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
        cvError /= C_NUM_FOLDS;

        return cvError;
    }

    public Instances removeNonSelectedFeatures(Instances instances){
        //use the previously calculated feature set (from the backwards method) in order to remove the
        //features from the given instances set. NOTE: the same instances attributes must be used otherwise there willbe
        //no meaning whatsoever to the newly created instances objects (since features must be the same on both sets, at
        // the onset).
        for(int i = 0 ; i < m_removed_features.size(); i ++){
            //remove the feature from the given features, and continue processing the instances set until all features
            //have been removed.
            instances = removeFeature(instances,m_removed_features.get(i));
        }

        return instances;
    }


    // assistive methods - folding

    /**
     * divide the instances into k folds, of equal size. possibly shuffle instances if required
     * @param instances
     * @return
     */
    private int[] foldIndices(Instances instances, int foldNum){
        //todo test method
        //divide to the nearest integer value possible (last set might be smaller than int value by remainder)

        int instCount = (int) Math.floor( (double) instances.numInstances() / foldNum); //rounded version of course

        int[] foldIndexArray = new int[C_NUM_FOLDS+1];
        foldIndexArray[0] = 0;
        int remainder = instances.numInstances() % foldNum;

        int foldSize = 0;
        //we will need to distribute the instances between all the folds, in order to prevent large differences between
        //the folds
        if(remainder>1)
             foldSize = instances.numInstances() / C_NUM_FOLDS + 1;
        else
             foldSize = instances.numInstances() / C_NUM_FOLDS;

        for(int i = foldSize ; i < instances.numInstances() ; i+= foldSize){

            foldIndexArray[foldSize/i]= i ;

            if(i>=instances.numInstances())
                foldIndexArray[C_NUM_FOLDS] = instances.numInstances()-1;
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

    public static void main(String[] args){
        try {
            //load datasets according to test and train split instructions
            loadData("file.txt");
            loadData("file2.txt");

            String training = "ElectionsData_train.txt";
            String testing = "ElectionsData_test.txt";

            Instances trainingData = loadData(training);
            Instances testData = loadData(testing);


            //todo : test randomization of instances
            Random random = new Random(123123);
            trainingData.randomize(random);
            testData.randomize(random);

            trainingData.setClassIndex(0);
            testData.setClassIndex(0);

            SVMEval eval = new SVMEval();

            //choose the best kernel using cross fold validation
            eval.chooseKernel(trainingData);

            Instances workingSet = eval.backwardsWrapper(trainingData, 0.05, 5);

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
    }

}
