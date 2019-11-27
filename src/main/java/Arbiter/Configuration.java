//package Arbiter;
//
//import org.deeplearning4j.arbiter.ComputationGraphSpace;
//import org.deeplearning4j.arbiter.MultiLayerSpace;
//import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
//import org.deeplearning4j.arbiter.conf.updater.schedule.StepScheduleSpace;
//import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
//import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
//import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
//import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
//import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
//import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
//import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
//import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
//import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
//import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
//import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
//import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
//import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
//import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
//import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
//import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
//import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
//import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
//import org.deeplearning4j.nn.api.OptimizationAlgorithm;
//import org.deeplearning4j.nn.conf.BackpropType;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.layers.DenseLayer;
//import org.deeplearning4j.nn.conf.layers.OutputLayer;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.nn.weights.WeightInit;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//import org.nd4j.linalg.learning.config.IUpdater;
//import org.nd4j.linalg.learning.config.Nesterovs;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
//import org.nd4j.linalg.schedule.ScheduleType;
//
//import java.io.File;
//import java.util.Properties;
//
//public class Configuration {
//
//    static MultiLayerSpace GetMultiLayerConfiguration(int inputSize, int outputSize) {
//
//        DiscreteParameterSpace<Activation> activationSpace = new DiscreteParameterSpace<>(Activation.ELU,
//                Activation.RELU,
//                Activation.LEAKYRELU,
//                Activation.TANH,
//                Activation.SELU,
//                Activation.IDENTITY,
//                Activation.HARDSIGMOID);
//        ParameterSpace<Integer> firstLayerSize = new IntegerParameterSpace(outputSize, inputSize);
//        ParameterSpace<Integer> secondLayerSize = new IntegerParameterSpace(outputSize, inputSize);
//        ParameterSpace<Integer> thirdLayerSize = new IntegerParameterSpace(outputSize, inputSize);
////        IUpdater[] updater = new IUpdater[]{
////                new Nesterovs(learningRate, 0.9)
////        };
//        ParameterSpace<IUpdater> updaterSpace = AdamSpace.withLRSchedule(new StepScheduleSpace(ScheduleType.EPOCH, new ContinuousParameterSpace(0.0, 0.1), 0.5, 2));
//
//        MultiLayerSpace builder = new MultiLayerSpace.Builder()
//                .seed(12345)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Nesterovs(0.01, 0.9))
//                .layer(new DenseLayerSpace.Builder().nIn(inputSize).nOut(firstLayerSize)
//                        .activation(activationSpace)
//                        .build())
//                .layer(new OutputLayerSpace.Builder()
//                        .activation(activationSpace)
//                        .nIn(firstLayerSize).nOut(secondLayerSize).build())
//                .build()
//        );
//
//    }
//
//}