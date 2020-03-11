import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.evaluator.multilayer.ClassificationEvaluator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;

public class ImageRecognition {
    protected static int HEIGHT = 100;
    protected static int WIDTH = 100;
    protected static int CHANNELS = 3;
    protected static int BATCH_SIZE = 1;
    protected static int NUM_LABELS = 2;
    protected static int EPOCHS = 200;
    protected static long SEED = 42;
    protected static Random RNG = new Random(SEED);
    protected static int MAX_PATHS_PER_LABEL = 18;
    protected static double SPLIT_TEST_AND_TRAIN = 0.8;

    public static String DATA_LOCAL_PATH;

    public void run(String[] args) throws Exception {

        int numLabels;
        DATA_LOCAL_PATH = "images/";

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(DATA_LOCAL_PATH);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, RNG);
        int numExamples = toIntExact(fileSplit.length());
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(RNG, labelMaker, numExamples, numLabels, MAX_PATHS_PER_LABEL);

        InputSplit[] inputSplits = fileSplit.sample(pathFilter, SPLIT_TEST_AND_TRAIN, 1 - SPLIT_TEST_AND_TRAIN);
        InputSplit trainData = inputSplits[0];
        InputSplit testData = inputSplits[1];

        ImageTransform flipTransform1 = new FlipImageTransform(RNG);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(RNG, 42);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1,0.9),
                new Pair<>(flipTransform2,0.8),
                new Pair<>(warpTransform,0.5));

        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        MultiLayerNetwork network = Network();

        ImageRecordReader trainRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        trainRR.initialize(trainData, transform);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, numLabels);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        ImageRecordReader testRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        testRR.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, numLabels);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        System.out.println(trainIter.next(1));

//        network.fit(trainIter, EPOCHS);
//
//        network.save(new File("model ImageRecognition/" + "model.bin"));
//
//        Evaluation evaluation = network.evaluate(testIter);
//        System.out.println("Evaluation:" + evaluation);
    }

    @SuppressWarnings("SameParameterValue")
    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    @SuppressWarnings("SameParameterValue")
    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    public MultiLayerNetwork Network() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .list()
                .layer(0, convInit("cnn1", CHANNELS, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(NUM_LABELS)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(HEIGHT, WIDTH, CHANNELS))
                .build();

        return new MultiLayerNetwork(conf);
    }

    public static void main(String[] args) throws Exception {
        new ImageRecognition().run(args);
    }
}