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


import java.util.*;

public class netTest {
    public static final int NUM_EXAMPLES = 10;
    public static final int NUM_INPUTS = 4;
    public static final int NUM_BATCHES = 10;
    public static final int NUM_WEIGHTS = 32;
    private static final int NUM_LAYERS = 2;

    public double fitness(List<Double> weights) {
        INDArray weightMatrix = Nd4j.create(weights).reshape(new int[]{NUM_LAYERS, NUM_INPUTS, NUM_INPUTS});
        System.out.println("weightMatrix = " + weightMatrix);

        ArrayList<Layer> layers = new ArrayList<>(NUM_LAYERS) {{
            add(new DenseLayer.Builder().nIn(NUM_INPUTS).nOut(NUM_INPUTS)
                    .activation(Activation.TANH).build());
            add(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(NUM_INPUTS).nOut(NUM_INPUTS).build());
        }};

        float[] data = new float[NUM_EXAMPLES * NUM_INPUTS];
        Random rng = new Random(123);

        for (int i = 0; i < NUM_EXAMPLES * NUM_INPUTS; i++) {
            data[i] = getRandom();
        }
        final INDArray trainingData = Nd4j.create(data, NUM_EXAMPLES, NUM_INPUTS);

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
                .weightInit(new CustomWeightInitializer(weightMatrix, NUM_LAYERS))
                .updater(new Nesterovs(0.01, 0.9))
                .list();

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            builder.layer(i, layer);
        }

        MultiLayerNetwork neuralNet = new MultiLayerNetwork(builder.build());


//        ComputationGraph neuralNet = new ComputationGraph(conf);
        neuralNet.init();

        RegressionEvaluation eval = neuralNet.evaluateRegression(dataSetIterator);
//        System.out.println(eval.stats());

//        final INDArray testData = Nd4j.create(new float[]{1, 0, 0, 1}, 1, numInputs);
//        INDArray output = neuralNet.outputSingle(false, testData);
//        INDArray output = neuralNet.output(testData);
//        System.out.println(output);
        return eval.averageMeanSquaredError();
    }

//    public static void main(String[] args) throws IOException, InterruptedException{
//
//
//    }

    public float getRandom() {
        double x = Math.random();
        if (x < 0.5) {
            return 0;
        } else {
            return 1;
        }
    }
}
