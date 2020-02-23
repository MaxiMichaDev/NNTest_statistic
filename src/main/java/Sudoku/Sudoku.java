package Sudoku;

import de.sfuhrm.sudoku.Creator;
import de.sfuhrm.sudoku.GameMatrix;
import de.sfuhrm.sudoku.Riddle;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;


import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Sudoku {
    public static void main(String[] args) throws IOException, InterruptedException {

        int seed = 12345;
        int numberFiles = 10000;
        final String inputPathname = "D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\trainFiles\\input\\input";
        final String outputPathname = "D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\trainFiles\\output\\output";

        final String testInputPathname = "D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\testFiles\\input\\input";
        final String testOutputPathname = "D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\testFiles\\output\\output";

//        createFiles(numberFiles, inputPathname, outputPathname);
//
//        createFiles(numberFiles, testInputPathname, testOutputPathname);

        Random random = new Random();
        random.setSeed(seed);
        int numLinesToSkip = 0;
        char delimiter = ',';
        int batchSize = 100;
        int numInputs = 810;

//        RecordReader inputReader = new CSVRecordReader(numLinesToSkip, delimiter);
//        inputReader.initialize(new FileSplit(new File(/*"D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\trainFiles\\input\\"*/"D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\trainFiles\\output\\"),random));
//
//        RecordReader outputReader = new CSVRecordReader(numLinesToSkip, delimiter);
//        outputReader.initialize(new FileSplit(new File("D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\trainFiles\\output\\"),random));
//
//        MultiDataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
//                .addReader("Inputs", inputReader)
//                .addReader("Outputs", outputReader)
//                .addInput("Inputs")
//                .addOutput("Outputs")
//                .build();


//        RecordReader testInputReader = new CSVRecordReader(numLinesToSkip, delimiter);
//        testInputReader.initialize(new FileSplit(new File(/*"D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\testFiles\\input\\"*/"D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\testFiles\\output\\"),random));
//
//        RecordReader testOutputReader = new CSVRecordReader(numLinesToSkip, delimiter);
//        testOutputReader.initialize(new FileSplit(new File("D:\\JetBrains\\IdeaProjects\\NNTest_statistic\\testFiles\\output\\"),random));
//
//        MultiDataSetIterator testIterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
//                .addReader("TestInputs", testInputReader)
//                .addReader("TestOutputs", testOutputReader)
//                .addInput("TestInputs")
//                .addOutput("TestOutputs")
//                .build();

//        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(123)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Nesterovs(0.05,0.9))
//                .l2(0.0001)
//                .graphBuilder()
//                .addInputs("input")
//                .addLayer("L1", new DenseLayer.Builder().nIn(numInputs).nOut(numInputs).activation(Activation.TANH).build(), "input")
//                .addLayer("out", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(numInputs).nOut(numInputs).activation(Activation.IDENTITY).build(), "L1")
//                .setOutputs("out")
//                .backpropType(BackpropType.Standard)
//                .build();

//        ComputationGraph net = new ComputationGraph(conf);
//        net.init();
//        net.setListeners(new ScoreIterationListener(1));
//
//        while (iterator.hasNext()) {
//            MultiDataSet multiDataSet = iterator.next();
//            net.fit(multiDataSet);
//        }

//        MultiDataSet multiDataSet = iterator.next();
//        multiDataSet.shuffle();
//        for (int j = 0; j < 100; j++) {
//            net.fit(multiDataSet);
//        }

//        RegressionEvaluation eval = net.evaluateRegression(testIterator);
//        System.out.println(eval.stats());



        GameMatrix matrix = Creator.createFull();
        Riddle riddle = Creator.createRiddle(matrix);
//        float[] binarySolved = toBinary(matrix);
//        float[] binaryRiddle = toBinary(riddle);

        INDArray inputData = Nd4j.create(normalize(riddle));
        System.out.println(inputData);
        INDArray outputData = Nd4j.create(normalize(matrix));
        System.out.println(outputData);

//        INDArray inputMask = Nd4j.ones(810);
//        INDArray data = Nd4j.create(binary);
//        final INDArray data = Nd4j.create(/*binaryRiddle*/binarySolved, 1, 810);
//        INDArray output = net.outputSingle(false, data);
//        System.out.println("riddle................");
//        for (int a = 0; a < 810; a++) {
//            System.out.print(binaryRiddle[a]);
//            System.out.print(',');
//        }
//        System.out.println();

//        System.out.println("solution..............");
//        for (int a = 0; a < 810; a++) {
//            System.out.print(binarySolved[a]);
//            System.out.print(',');
//        }
//
//        System.out.println("solution network......");
//        System.out.println(output);
    }

    private static float[][] normalize(GameMatrix matrix) {
        float[][] value = new float[9][9];
        for (byte row = 0 ; row < 9 ; row++) {
            for (byte column = 0 ; column < 9 ; column++) {
                value[row][column] = matrix.get(row, column) / 10;
            }
        }
        System.out.println(matrix);
        System.out.println(value);
        return value;
    }

    private static float[] toBinary(GameMatrix matrix) {
        byte row;
        byte column;
        byte value;
        int c = 0;
        float[] binary = new float[810];
        for (row = 0 ; row < 9 ; row++) {
            for (column = 0 ; column < 9 ; column++) {
                value = matrix.get(row, column);
                for (int a = 0; a < 10; a++) {
                    if (value == a) {
                        binary[c] = 1;
                    } else {
                        binary[c] = 0;
                    }
                    c++;
                }
            }
        }
        return binary;
    }

    private static void createFiles(int numberFiles, String inputPathname, String outputPathname) throws IOException {
        for (int i = 0; i < numberFiles; i++) {
            GameMatrix matrix = Creator.createFull();
            Riddle riddle = Creator.createRiddle(matrix);

            File inputFile = new File(inputPathname+(i)+".txt");
            FileWriter inputWriter = new FileWriter(inputFile);
            writeDown(riddle, inputWriter);
            inputWriter.close();

            File outputFile = new File(outputPathname+(i)+".txt");
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
                sudokuToOneHot(matrix, writer, column, row, 10);
            }
        }
        for (column = 0 ; column < 8 ; column++) {
            sudokuToOneHot(matrix, writer, column, row, 10);
        }
            sudokuToOneHot(matrix, writer, column, row, 9);
        writer.close();
    }

    private static void sudokuToOneHot(GameMatrix matrix, FileWriter writer, byte column, byte row, int c) throws IOException {
        byte value;
        value = matrix.get(row, column);
        for (int a = 0; a < c; a++) {
            if (value == a) {
                writer.write("1");
                writer.write(",");
            } else {
                writer.write("0");
                writer.write(",");
            }
        }
        if (c == 9) {
            if (value == 9) {
                writer.write("1");
            } else {
                writer.write("0");
            }
        }
    }
}



