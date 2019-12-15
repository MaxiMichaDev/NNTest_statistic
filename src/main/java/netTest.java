import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.lang.reflect.Array;
import java.util.*;

public class netTest {
    public static final int NUM_EXAMPLES = 10;
    public static final int NUM_BATCHES = 10;
    public static final int[][] SHAPE = {{4, 4}, {4, 4}/*, {4, 1}*/};

    public double fitness(/*double[] weights*/ List<Double> weights) {

        ArrayList<Layer> layers = new ArrayList<Layer>(/*NUM_LAYERS*/Array.getLength(SHAPE[0])) {{
            add(new DenseLayer.Builder().nIn(SHAPE[0][0]).nOut(SHAPE[0][1])
                    .activation(Activation.TANH).build());
            add(new DenseLayer.Builder().nIn(SHAPE[1][0]).nOut(SHAPE[1][1])
                    .activation(Activation.TANH).build());
/*
            add(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(NUM_INPUTS).nOut(NUM_OUTPUTS).build());
*/
        }};

        float[] data = new float[NUM_EXAMPLES * SHAPE[0][0]];
        Random rng = new Random(123);

        for (int i = 0; i < NUM_EXAMPLES * SHAPE[0][0]; i++) {
            data[i] = getRandom();
        }
        final INDArray trainingData = Nd4j.create(data, NUM_EXAMPLES, SHAPE[0][0]);

        DataSet dataSet = new DataSet(trainingData, trainingData);
        List<DataSet> listDs = dataSet.asList();
        Collections.shuffle(listDs,rng);

        DataSetIterator dataSetIterator = new ListDataSetIterator<>(listDs, NUM_BATCHES);

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

        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(new CustomWeightInitializer(/*weights*//*weightMatrixLayer1, weightMatrixLayer2,*//**//* weightMatrixLayer3,*//* NUM_LAYERS*/ SHAPE, weights))
                .updater(new Nesterovs(0.01, 0.9))
                .list();

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            builder.layer(i, layer);
        }

        MultiLayerNetwork neuralNet = new MultiLayerNetwork(builder.build());


//        ComputationGraph neuralNet = new ComputationGraph(conf);
        neuralNet.init();

        final INDArray testData = Nd4j.create(new float[]{1, 0, 0, 1}, 1, SHAPE[0][0]);
//        INDArray output = neuralNet.outputSingle(false, testData);
        INDArray output = neuralNet.output(testData);
//        System.out.println(output);

        RegressionEvaluation eval = neuralNet.evaluateRegression(dataSetIterator);
//        System.out.println(eval.stats());

        return eval.averageMeanSquaredError();
    }


    public float getRandom() {
        double x = Math.random();
        if (x < 0.5) {
            return 0;
        } else {
            return 1;
        }
    }
}
