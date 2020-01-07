import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.NesterovsSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.RegressionScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class Arbiter {

    public static final Random rng = new Random(123);

    public static void main(String[] args) throws Exception {

        ParameterSpace<Integer> layerSizeHyperparam = new IntegerParameterSpace(2, 10);
        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.001, 1);
        ParameterSpace<Double> learningRateMomentumHyperparam = new ContinuousParameterSpace(0.001, 1);

        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new NesterovsSpace(learningRateHyperparam, learningRateMomentumHyperparam))
                .layer(new DenseLayerSpace.Builder()
                        .nIn(2)
                        .nOut(layerSizeHyperparam)
                        .activation(Activation.TANH)
                        .build())
                .layer(new OutputLayerSpace.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(layerSizeHyperparam)
                        .nOut(1)
                        .build())
                .build();

        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);

        Class<? extends org.deeplearning4j.arbiter.optimize.api.data.DataSource> dataSourceClass = ExampleDataSource.class;
        Properties dataSourceProperties = new Properties();
        dataSourceProperties.setProperty("minibatize", "100");

        String baseSaveDirectory = "model/";
        File f = new File(baseSaveDirectory);
        if (f.exists()) f.delete();
        f.mkdir();
        ResultSaver modelSaver = new FileModelSaver(baseSaveDirectory);

        ScoreFunction scoreFunction = new RegressionScoreFunction(RegressionEvaluation.Metric.MSE);

        TerminationCondition[] terminationConditions = {
                new MaxTimeCondition(1, TimeUnit.MINUTES),
                new MaxCandidatesCondition(10)};

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataSource(dataSourceClass,dataSourceProperties)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

        IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());

        runner.execute();

        String s = "Best score: " + runner.bestScore() + "\n" +
                "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" +
                "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);

        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference> allResults = runner.getResults();

        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        MultiLayerNetwork bestModel = (MultiLayerNetwork) bestResult.getResultReference().getResultModel();

        System.out.println("\n\nConfiguration of best model:\n");
        System.out.println(bestModel.getLayerWiseConfigurations().toJson());

    }

    private static DataSetIterator getTrainingData(int batchSize, Random rand){
        double [] sum = new double[1000];
        double [] input1 = new double[1000];
        double [] input2 = new double[1000];
        for (int i= 0; i< 1000; i++) {
            input1[i] = 3 * rand.nextDouble();
            input2[i] =  3 * rand.nextDouble();
            sum[i] = input1[i] + input2[i];
        }
        INDArray inputNDArray1 = Nd4j.create(input1, 1000,1);
        INDArray inputNDArray2 = Nd4j.create(input2, 1000,1);
        INDArray inputNDArray = Nd4j.hstack(inputNDArray1,inputNDArray2);
        INDArray outPut = Nd4j.create(sum, 1000, 1);
        DataSet dataSet = new DataSet(inputNDArray, outPut);
        List<DataSet> listDs = dataSet.asList();
        Collections.shuffle(listDs,rng);
        return new ListDataSetIterator<>(listDs,batchSize);

    }

    public static class ExampleDataSource implements org.deeplearning4j.arbiter.optimize.api.data.DataSource {
        private int minibatchSize;

        public ExampleDataSource() {

        }

        @Override
        public void configure(Properties properties) {
            this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize", "16"));
        }

        @Override
        public Object trainData() {
            try {
                return getTrainingData(100, rng);

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public Object testData() {
            try {
                return getTrainingData(100, rng);

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public Class<?> getDataType() {
            return ListDataSetIterator.class;
        }
    }
}
