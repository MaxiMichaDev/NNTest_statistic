import de.sfuhrm.sudoku.Creator;
import de.sfuhrm.sudoku.GameMatrix;
import de.sfuhrm.sudoku.Riddle;

import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import javafx.util.Pair;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;


import java.util.*;

public class netTest {
    public static void main(String[] args) throws IOException, InterruptedException{

        int numExamples = 10000;
        int numInputs = 4;
        int numBatches = 10;
        float[] data = new float[numExamples * numInputs];
        Random rng = new Random(123);

        for (int i = 0; i < numExamples * numInputs; i++) {
            data[i] = getRandom();
        }
        final INDArray trainingData = Nd4j.create(data, numExamples, numInputs);

        DataSet dataSet = new DataSet(trainingData, trainingData);
        List<DataSet> listDs = dataSet.asList();
        Collections.shuffle(listDs,rng);

        DataSetIterator dataSetIterator = new ListDataSetIterator<>(listDs, numBatches);

//        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(123)
//                .weightInit(new CustomWeightInitializer())
//                .updater(new Nesterovs(0.01,0.9))
//                .l2(0.0001)
//                .graphBuilder()
//                .addInputs("input")
//                .addLayer("L1", new DenseLayer.Builder().nIn(numInputs).nOut(numInputs).activation(Activation.TANH).build(), "input")
//                .addLayer("L2", new DenseLayer.Builder().nIn(numInputs).nOut(2 * numInputs).activation(Activation.TANH).build(), "L1")
//                .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(2 * numInputs).nOut(numInputs).activation(Activation.IDENTITY).build(), "L2")
//                .setOutputs("out")
//                .backpropType(BackpropType.Standard)
//                .build();

        int numLayers = 2;
        ArrayList<Layer> layers = new ArrayList<>(numLayers) {{
            add(new DenseLayer.Builder().nIn(numInputs).nOut(numInputs)
                    .activation(Activation.TANH).build());
            add(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(numInputs).nOut(numInputs).build());
        }};

        ArrayList<INDArray> weights = new ArrayList<>(numLayers) {{
            add(null);
            add(null);
        }};


        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(new CustomWeightInitializer(weights, numLayers))
                .updater(new Nesterovs(0.01, 0.9))
                .list();

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            builder.layer(i, layer);
        }

        MultiLayerNetwork neuralNet = new MultiLayerNetwork(builder.build());

//        ComputationGraph neuralNet = new ComputationGraph(conf);
        neuralNet.init();
        neuralNet.setListeners(new ScoreIterationListener(1));


        while (dataSetIterator.hasNext()) {
            DataSet trainDataSet = dataSetIterator.next();
            neuralNet.fit(trainDataSet);
        }

//        DataSet trainDataSet = dataSetIterator.next();
//        dataSetIterator.shuffle();
//        for (int j = 0; j < 100; j++) {
//            neuralNet.fit(trainDataSet);
//        }

        RegressionEvaluation eval = neuralNet.evaluateRegression(dataSetIterator);
        System.out.println(eval.stats());

        final INDArray testData = Nd4j.create(new float[]{1, 0, 0, 1}, 1, numInputs);
//        INDArray output = neuralNet.outputSingle(false, testData);
        INDArray output = neuralNet.output(testData);
        System.out.println(output);

    }

    public static float getRandom() {
        double x = Math.random();
        if (x < 0.5) {
            return 0;
        } else {
            return 1;
        }
    }
}
