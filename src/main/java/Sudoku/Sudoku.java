package Sudoku;

import au.com.bytecode.opencsv.CSVReader;
import de.sfuhrm.sudoku.Creator;
import de.sfuhrm.sudoku.GameMatrix;
import de.sfuhrm.sudoku.Riddle;
import javafx.beans.binding.When;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.bytedeco.opencv.opencv_dnn.DeconvolutionLayer;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.SingletonMultiDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.convolution.Deconvolution2DLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.iter.INDArrayIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;

public class Sudoku{

    public static int EPOCHS = 10;

    final String inputPathname = "SudokuData\\input\\";
    final String outputPathname = "SudokuData\\output\\";

    final String inputTestPathname = "SudokuData\\testIn\\";
    final String outputTestPathname = "SudokuData\\testOut\\";

    public void run(String[] args) throws Exception {

//        createFiles(10000, inputPathname, outputPathname);
//        createFiles(100, inputTestPathname, outputTestPathname);

        RecordReader inputReader = new CSVRecordReader(0, ',');
        inputReader.initialize(new FileSplit(new File("SudokuData\\input\\")));

        RecordReader outputReader = new CSVRecordReader(0, ',');
        outputReader.initialize(new FileSplit(new File("SudokuData\\output\\")));

        MultiDataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(10)
                .addReader("Inputs", inputReader)
                .addReader("Outputs", outputReader)
                .addInput("Inputs")
                .addOutput("Outputs")
                .build();

        RecordReader inputTestReader = new CSVRecordReader(0, ',');
        inputTestReader.initialize(new FileSplit(new File("SudokuData\\testIn\\")));

        RecordReader outputTestReader = new CSVRecordReader(0, ',');
        outputTestReader.initialize(new FileSplit(new File("SudokuData\\testOut\\")));

        MultiDataSetIterator testIterator = new RecordReaderMultiDataSetIterator.Builder(50)
                .addReader("testInputs", inputTestReader)
                .addReader("testOutputs", outputTestReader)
                .addInput("testInputs")
                .addOutput("testOutputs")
                .build();

        MultiLayerNetwork network = Network();

        while (iterator.hasNext()) {
            network.fit(iterator);
        }

        INDArray in = Nd4j.readNumpy("SudokuData\\testIn\\5.csv", ",");
        INDArray out = Nd4j.readNumpy("SudokuData\\testOut\\5.csv", ",");
        System.out.println(out);
        out = network.output(in);
        System.out.println(out);
    }


//    private static ArrayList<INDArray> getData(){
//        GameMatrix matrix = Creator.createFull();
//        Riddle riddle = Creator.createRiddle(matrix);
//        INDArray out = Nd4j.create(normalize(matrix));
//        INDArray in = Nd4j.create(normalize(riddle));
//        return new ArrayList<INDArray>(2){{
//            add(in);
//            add(out);
//        }};
//    }

//    private static float[][] normalize(GameMatrix matrix) {
//        float[][] value = new float[9][9];
//        for (byte row = 0 ; row < 9 ; row++) {
//            for (byte column = 0 ; column < 9 ; column++) {
//                value[row][column] = (float) matrix.get(row, column) / 10;
//            }
//        }
//        return value;
//    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    @SuppressWarnings("SameParameterValue")
    private ConvolutionLayer conv(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{2,2}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    public MultiLayerNetwork Network() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .list()
                .layer(0, convInit("cnn0", 1, 50, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, conv("cnn1", 100, new int[]{1, 1}, new int[]{1, 1}, 0))
//                .layer(1, maxPool("maxpool1", new int[]{2, 2}))
                .layer(2, conv("cnn2", 100, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(3, conv("cnn3", 100, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(4, maxPool("maxpool2", new int[]{2, 2}))
//                .layer(4, new Deconvolution2DLayer.)
                .layer(5, new DenseLayer.Builder().nOut(500).build())
                .layer(6, new DenseLayer.Builder().nOut(1000).build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(81)
                        .activation(Activation.IDENTITY)
                        .build())
                .setInputType(InputType.convolutionalFlat(9, 9, 1))
                .build();

        return new MultiLayerNetwork(conf);
    }

    private static void createFiles(int numberFiles, String inputPathname, String outputPathname) throws IOException {
        for (int i = 0; i < numberFiles; i++) {
            GameMatrix matrix = Creator.createFull();
            Riddle riddle = Creator.createRiddle(matrix);

            File inputFile = new File(inputPathname+(i)+".csv");
            FileWriter inputWriter = new FileWriter(inputFile);
            writeDown(riddle, inputWriter);
            inputWriter.close();

            File outputFile = new File(outputPathname+(i)+".csv");
            FileWriter outputWriter = new FileWriter(outputFile);
            writeDown(matrix, outputWriter);
            outputWriter.close();
        }
    }

    private static void writeDown(GameMatrix matrix, FileWriter writer) throws IOException {
        byte column;
        byte row;
        for (row = 0 ; row < 8 ; row++) {
            for (column = 0 ; column < 9 ; column++) {
//                sudokuToOneHot(matrix, writer, column, row, 10);
                writeToFile(matrix, writer, column, row, true);
            }
        }
        for (column = 0 ; column < 8 ; column++) {
//            sudokuToOneHot(matrix, writer, column, row, 10);
            writeToFile(matrix, writer, column, row, true);
        }
//        sudokuToOneHot(matrix, writer, column, row, 9);
        writeToFile(matrix, writer, column, row, false);
    }

    private static void /*sudokuToOneHot*/ writeToFile(GameMatrix matrix, FileWriter writer, byte column, byte row, boolean c /*int c*/) throws IOException {
        float value;
        value = (float) matrix.get(row, column) / 10;
        writer.write(String.valueOf(value));
        if (c == true) {
            writer.write(",");
        } else {
            writer.close();
        }
//        for (int a = 0; a < c; a++) {
//            if (value == a) {
//                writer.write("1");
//                writer.write(",");
//            } else {
//                writer.write("0");
//                writer.write(",");
//            }
//        }
//        if (c == 9) {
//            if (value == 9) {
//                writer.write("1");
//            } else {
//                writer.write("0");
//            }
//        }
    }

    public static void main(String[] args) throws Exception {
        new Sudoku().run(args);
    }
}



