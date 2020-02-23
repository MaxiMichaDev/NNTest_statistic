package Sudoku;

import de.sfuhrm.sudoku.Creator;
import de.sfuhrm.sudoku.GameMatrix;
import de.sfuhrm.sudoku.Riddle;
import org.datavec.api.records.reader.RecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;

public class Sudoku{

    public static void main(String[] args) {


    }

    private static ArrayList<INDArray> getData(){
        GameMatrix matrix = Creator.createFull();
        Riddle riddle = Creator.createRiddle(matrix);
        return new ArrayList<INDArray>(2){{
            add(Nd4j.create(normalize(matrix)));
            add(Nd4j.create(normalize(riddle)));
        }};
    }

    private static float[][] normalize(GameMatrix matrix) {
        float[][] value = new float[9][9];
        for (byte row = 0 ; row < 9 ; row++) {
            for (byte column = 0 ; column < 9 ; column++) {
                value[row][column] = (float) matrix.get(row, column) / 10;
            }
        }
        return value;
    }
}



