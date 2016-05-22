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

    //constants
    private final int C_NUM_FOLDS = 3;
    private final int C_POLY_KERNEL_MIN = 2;
    private final int C_POLY_KERNEL_MAX = 4;
    private final int C_RBF_KERNEL_MAX = 2;
    private final int C_RBF_KERNEL_MIN = -10;

    //main SMO classifier
    public SMO mySMO = new SMO();
    //the validation SMO will be used to calculate the cross-validation-error by the CV method and given instances. this
    //will be useful since we wouldn't like to overwrite or affect the class' main classifier when testing for CV
    //errors.
    private SMO validationSMO = new SMO();

    //switch to denote we are using an ad-hoc cross-validation classifier to
    // calculate error (contrary to the main SMO classifier)
    private boolean CVSMOactive = false;

    //kernel selection fields
    public double m_bestError = Double.MAX_VALUE;
    public Kernel m_bestKernel ;

    private ArrayList<Integer> m_removed_features ;


    /**
     * a buffered reader to read the data file
     * @param filename data file name
     * @return an inputreader instance (used by load data)
     */
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
	 * Load the data from the inputreader into an instances object
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		return data;
	}

    /**
     * Simply build a classifier to be used later on by the user and other class methods.
     * @param instances
     * @throws Exception
     */
    public void buildClassifier(Instances instances) throws Exception{
            //builds the classifier using Weka's built in SMO SVM classifier module
            mySMO.buildClassifier(instances);
        }

    /**
     *
     *
     *    Input: threshold, min number of attributes, instances
     *    Action: performs the backwards wrapper feature selection algorithm as explained
     *    above.
     *            Output: New Instances object with the chosen subset of the original features, indices of
     *    chosen features.
     *
     *    Remove the features that contribute the least from teh attributes list
     * @param instances
     * @param T
     * @param K
     * @return
     * @throws Exception
     */
    public Instances backwardsWrapper(Instances instances,double T, int K) throws Exception{
        double error_diff = 0;
        int i_minimal=0;

        //init the the removed feature field, in which we will retain the removed attributes
        //the indices can only be used in the order they are stored since each index refelcts its corresponding attribute
        //after previous entries have been removed before it.
        m_removed_features = new ArrayList<Integer>();

        double original_error = calcCrossValidationError(instances);
        double minimal_error =Double.MAX_VALUE;
        double new_error = Double.MAX_VALUE;
        do{
            //reset the i value each time the while loop iterates in order to restart the attribute search process
            i_minimal = 1;

            //calculate cross validation error without the first feature in the attributes array. While
            //avoiding the class attribute index
            minimal_error = calcCrossValidationError(removeFeature(instances,i_minimal));

            //iterate over all possible attributes in order to find the attribute which will result in the minimal
            // cross validation error.
               for(int i = 2 ; i < instances.numAttributes() ; i++) {

                   //calculate the error after removing the ith feature from the dataset
                       new_error = calcCrossValidationError(removeFeature(instances,i));

                   //if the error calculated by removing the current attribute is smaller than the pervious minimal
                   //error - this attribute is the new candidate for removal instead.
                   if (new_error < minimal_error) {
                        minimal_error = new_error; //new best error
                        i_minimal = i; //keep the index of the attribute
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
        }while(instances.numAttributes()>K || error_diff > T);

        //return the instances without the removed features
        return instances;
    }

    /**
     * Removes a single feature from the instances provided, and returns the instances object without it
     * @param instances
     * @param attributeIx
     * @return instances object without the selected feature
     */
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

    /**
     * Calculate the average classification error of our current classifier with regard to the true class value.
     * @param instances
     * @return % errors
     */
    private double calcAvgError(Instances instances){

        //choose current classifier
        SMO workingClassifier ;

        if(CVSMOactive==true)
            workingClassifier = validationSMO;
        else
            workingClassifier = mySMO;

        //enumration of instances to go over
        Enumeration<Instance> instEnum = instances.enumerateInstances();

        int countErrors = 0;

        while(instEnum.hasMoreElements())
             {
                 try{
                     Instance currentInstance = instEnum.nextElement();
                     //if the classification is incorrect - increment the error counter
                     if(currentInstance.classValue()!=workingClassifier.classifyInstance(currentInstance))
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

        //set classifier to CV classifier during the kernel selection process
        CVSMOactive = true;

        //set the kernel to the validation classifier during the testing phase
        validationSMO.setKernel(kernel);

        //test all RBF kernels
        for(int i = C_RBF_KERNEL_MIN ; i <= C_RBF_KERNEL_MAX; i++){
            //set the kernal gamma value
            kernel.setGamma(Math.pow(2, i));

            //get the instances in the folds and test them
            error = calcCrossValidationError(instances);

            //retain the best kernel result and set it as the kernel for our hypothesis
                if (error < m_bestError) {
                    m_bestError = error;
                    RBFKernel newRBF = new RBFKernel();
                    newRBF.setGamma(Math.pow(2, i));
                    m_bestKernel = newRBF;

                }
            }

        //set the kernel to the polynomial kernel - and test it
        validationSMO.setKernel(kernel2);

        //polynomial kernel
        for(int i = C_POLY_KERNEL_MIN ; i <= C_POLY_KERNEL_MAX ; i++){

            kernel2.setExponent(i);

            //calculate cross validation error using the selected kernel and provided instances
            error = calcCrossValidationError(instances);

            //retain the best kernel result and set it as the kernel for our hypothesis
            if(error < m_bestError){
                m_bestError = error;
                PolyKernel newPoly = new PolyKernel();
                newPoly.setExponent(i);
                m_bestKernel = newPoly;
            }

        }

        //set the kernel after evaluating all possible options
        mySMO.setKernel(m_bestKernel);

        //use the class SMO classifier instead of the training classifier (set training switch off)
        CVSMOactive=false;
    }


    /**
     * perform cross validation process using x-1 folds for training the classifier and one fold to test on.
     * @param instances
     * @return
     * @throws Exception
     */
    public double calcCrossValidationError(Instances instances) throws Exception{

        //randomize the instances when testing the cross fold validation
        Random rand = new Random(12345);
        instances.randomize(rand);

        //get splitting indices for folding
        int[] subsetIndices = foldIndices(instances,C_NUM_FOLDS);
        Instances[] instArray;

        double cvError = 0;

        //iterate over all folds
        for(int foldix = 0 ; foldix <=C_NUM_FOLDS-1 ; foldix++) {
            //divide instances into 2 Instances objects, one containing the majority of instance (x-1 folds), the other the rest of the instances (to test upon)
            instArray = getFoldInstances(subsetIndices[foldix],subsetIndices[foldix+1],instances);

            //set the current instances x-1 folds into the validation classifier - and build the classifier.
            //after doing so we will be able to calculate the average classifcation error. As explained in the
            // previous recitation
            if(CVSMOactive)
                validationSMO.buildClassifier(instArray[0]);
            else
                mySMO.buildClassifier(instArray[0]);

            //calculate avg error for current fold (which is the 1/3 of instances that remain)
            cvError += calcAvgError(instArray[1]);
        }

        //divide the sum of errors by the number of folds to calculate the cross validation error
        cvError /= C_NUM_FOLDS;

        return cvError;
    }

    /**
     * Removes all features that are not used by the SMO classifier. The feature's contribution to the model is not
     * substantial (below a given threshold) and therefore they can be removed (ignored while classyfying) to increase
     * the efficiency of the classification process.
     * @param instances
     * @return instances object without the specified attributes
     */
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
        //divide to the nearest integer value possible (last set might be smaller than int value by remainder)

        int instCount = (int) Math.floor( (double) instances.numInstances() / foldNum); //rounded version of course

        int[] foldIndexArray = new int[C_NUM_FOLDS+1];
        foldIndexArray[0] = 0;
        int remainder = instances.numInstances() % foldNum;

        int foldSize = 0;
        //we will need to distribute the instances between all the folds, in order to prevent large differences between
        //the folds
        for(int i = 1 ; i < C_NUM_FOLDS+1 ; i++){

            if(remainder>0) {
                foldSize = instances.numInstances() / C_NUM_FOLDS + 1;
                remainder--;
            }
            else
                foldSize = instances.numInstances() / C_NUM_FOLDS;

            foldIndexArray[i]= foldSize +foldIndexArray[i-1];

        }

        //make sure we don't go over / under the instances array boundries
        foldIndexArray[C_NUM_FOLDS] = instances.numInstances()-1;

        return foldIndexArray;
    }

    /**
     * return instances that compose of the data outside the indices and the remaining instances
     * in the second index of the instances array.
     * @param start start index of folds
     * @param end end index of folds
     * @param instances the instances object to fold
     * @return instancesArray composing of 9/10 division ratio between instances according to instructions above
     */
    private Instances[] getFoldInstances(int start,int end,Instances instances){
        //Enumeration<Instance> instEnum = instances.enumerateInstances();
        Instances[] instancesArray = new Instances[2] ;
        instancesArray[0] = new Instances(instances,instances.numInstances());
        instancesArray[1] = new Instances(instances,instances.numInstances());

        //if the current fold is not the first one - add +1 to start index to avoid overlapping between folds
        if(start!=0)
            start++;

        //fold the instances into 2 instances classes objects, one containing x-1 folds and the other 1 fold
        for(int ix = 0; ix < instances.numInstances();ix++){

            if(ix<start || ix > end)

                instancesArray[0].add(instances.instance(ix)); //add to the first instances group (x-1 folds)
            else
                instancesArray[1].add(instances.instance(ix)); //the smaller group (1 fold)
            ix++; // increment index

        }

        return instancesArray;
    }

    //main method - according to example given in the exercise instructions

    public static void main(String[] args){
        try {
            //load datasets according to test and train split instructions

            String training = "ElectionsData_train.txt";
            String testing = "ElectionsData_test.txt";

            loadData(training);
            loadData(testing);

            Instances trainingData = loadData(training);
            Instances testData = loadData(testing);

            //randomize instances
            Random random = new Random(123123); // set random seed
            trainingData.randomize(random);
            testData.randomize(random);
            trainingData.setClassIndex(0);
            testData.setClassIndex(0);

            //create a new SVMEval class object
            SVMEval eval = new SVMEval();

            //choose the best kernel using cross fold validation process
            //NOTE: in my current solution, to avoid mistakes made in the previous ex., i  have created a stand-alone
            //training classifier, which will serve to test all the kernels using the training data. The resulting
            //kernel will be set to be the kernel of the main class classifier.
            eval.chooseKernel(trainingData);

            //remove features whose contribution is minimal to the classifier enhance efficiency
            Instances workingSet = eval.backwardsWrapper(trainingData, 0.05, 5);

            eval.buildClassifier(workingSet);

            Instances subsetOfFeatures =
                    eval.removeNonSelectedFeatures(testData);

            double avgError = eval.calcAvgError(subsetOfFeatures);

            // Finally, print the average error of the working dataset
            System.out.println(avgError);
        }
        catch (Exception c) {
            System.out.println("failed building classifier");
        }
    }

}
