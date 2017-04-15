package pca;

import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.util.*;

/**
 * Created by nguye on 4/13/2017.
 */
public class PCAClassifier {
    private static final int SIDE = 250;
    private static final double THRESHOLD = 0.9999;
    private static final String ORIGINAL_SPACE_TRANSPOSE_MATRIX_PATH = "transpose_matrix.dat";
    private static final String BIGGEST_DISTANCE_PATH = "biggest_distance.dat";
    private static final String MEAN_IMAGE_PATH = "mean_image.dat";
    private static final String splitRegex = File.separator;
    private List<Integer> trainingImageTypes;
    private List<Mat> trainingImageData;
    private List<String> trainingImagePaths;
    private SimpleMatrix meanMatrix;
    private SimpleMatrix originalSpace;
    private SimpleMatrix originalSpaceTransposeMatrix;
    private List<FaceVector> faceVectors;
    private double biggestDistance;
    public PCAClassifier(){
        trainingImageData = new ArrayList<>();
        trainingImagePaths = new ArrayList<>();
        trainingImageTypes = new ArrayList<>();
        faceVectors = new ArrayList<>();
    }
    public static  void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        String path = "D:\\BTL\\datn\\demo_data\\train";
//        pca.PCAClassifier classifier = new pca.PCAClassifier();
//        classifier.trainImage(path);
//        classifier.saveTrainingData("D:\\PCAData");
        PCAClassifier newClassifier = new PCAClassifier();
        newClassifier.readTrainingData("D:\\PCAData");
        //doc du lieu test
        List<String> validateImagePath = new ArrayList<>();
        List<Integer> typeList = new ArrayList<>();
        List<Mat> dataList = new ArrayList<>();
        String validatePath = "D:\\BTL\\datn\\demo_data\\test";
        newClassifier.loadImage(validatePath, validateImagePath, dataList, typeList);
        System.out.println("finished training");
        for(int i = 0; i < dataList.size(); i++) {
            System.out.println(newClassifier.predictImage(dataList.get(i)) + " " + typeList.get(i));
        }
    }

    public void saveTrainingData(String path){
        saveMatrixToFile(originalSpaceTransposeMatrix, path + splitRegex + ORIGINAL_SPACE_TRANSPOSE_MATRIX_PATH);
        saveMatrixToFile(meanMatrix, path + splitRegex + MEAN_IMAGE_PATH);
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(path + splitRegex + BIGGEST_DISTANCE_PATH, false))){
            writer.write(String.valueOf(biggestDistance));
        } catch (IOException e) {
            e.printStackTrace();
        }
        saveFaceVectorData(path);
    }

    private void saveFaceVectorData(String path){
        Map<String, Integer> countMap = new HashMap<>();
        for(FaceVector faceVector: faceVectors){
            File dirClassName = new File(path + splitRegex + faceVector.getLabel());
            if(!dirClassName.exists()){
                dirClassName.mkdir();
            }
            int fileName = countMap.getOrDefault(faceVector.getLabel(), 0);
            countMap.put(faceVector.getLabel(), fileName + 1);
            String dataFile = path + splitRegex + faceVector.getLabel() + splitRegex + fileName;
            saveMatrixToFile(faceVector.getVector(), dataFile);
        }
    }

    public void readTrainingData(String path){
        originalSpaceTransposeMatrix = readMatrixFromFile(path + File.separator + ORIGINAL_SPACE_TRANSPOSE_MATRIX_PATH);
        meanMatrix = readMatrixFromFile(path + File.separator + MEAN_IMAGE_PATH);
        try(BufferedReader reader = new BufferedReader(new FileReader(path + File.separator + BIGGEST_DISTANCE_PATH))){
            String line = reader.readLine();
            biggestDistance = Double.parseDouble(line);
        } catch (IOException e) {
            e.printStackTrace();
        }
        File currentDir = new File(path);
        for(File classDir: currentDir.listFiles()){
            if(classDir.isDirectory()){
                for(File dataFile: classDir.listFiles()){
                    try {
                        SimpleMatrix matrix = readMatrixFromFile(dataFile.getCanonicalPath());
                        String label = classDir.getName();
                        faceVectors.add(new FaceVector(label, matrix));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

    }

    private void saveMatrixToFile(SimpleMatrix matrix, String path){
        try(ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(path, false))){
            outputStream.writeObject(matrix);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private SimpleMatrix readMatrixFromFile(String path){
        SimpleMatrix matrix = null;
        try(ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(path))){
            matrix = (SimpleMatrix) inputStream.readObject();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        return matrix;
    }


    private   String predictImage(Mat mat){
        SimpleMatrix testImage = convertImageToMatrix(mat);
        SimpleMatrix projected = originalSpaceTransposeMatrix.mult(testImage.minus(meanMatrix));
        double distance = -1;
        String label = "unknown";
        String tempLabel = "";
        for (FaceVector faceVector : faceVectors) {
            double temp = calcDistance(projected, faceVector.getVector());
            if (distance > temp || distance == -1) {
                distance = temp;
                tempLabel = faceVector.getLabel();
            }
        }
        if (distance <= biggestDistance / 2) {
            return tempLabel;
        } else {
            return label;
        }
    }

    public void trainImage(String path){
        loadImage(path, trainingImagePaths, trainingImageData, trainingImageTypes);
        SimpleMatrix matrix = convertImagesSpaceToCovarianceMatrix();
        SimpleMatrix covarianceMatrix = matrix.transpose().mult(matrix);
        SimpleEVD<SimpleMatrix> eig = covarianceMatrix.eig();
        //tinh toan vector rieng
        EigenVectorSet[] eigenVectorSets = new EigenVectorSet[eig.getNumberOfEigenvalues()];
        for(int i = 0; i < eig.getNumberOfEigenvalues(); i++){
            EigenVectorSet temp = new EigenVectorSet(eig.getEigenvalue(i).getReal(), eig.getEigenVector(i));
            eigenVectorSets[i] = temp;
        }
        Arrays.sort(eigenVectorSets);
        //lay ra k vector rieng co gia tri rieng lon nhat
        List<EigenVectorSet> bestVectors = getBestEigenVectors(eigenVectorSets);
        //tao lai khong gian vector rieng U, ket thuc
        originalSpace = createOriginalSpaceEigenVectorMatrix(bestVectors, matrix);
        originalSpaceTransposeMatrix = originalSpace.transpose();//U transpose
        faceVectors = projectTrainingImagesToSpace(trainingImageTypes, matrix, originalSpaceTransposeMatrix);
        biggestDistance = computeBiggestDistanceInDataSet(faceVectors);
    }

    public  static void loadImage(String path, List<String> imagePath, List<Mat>dataList, List<Integer> typeList){
        File parentDir = new File(path);
        for(File dir: parentDir.listFiles()){
            int imageType = Integer.parseInt(dir.getName());
            for(File file: dir.listFiles()){
                try {
                    String filePath = file.getCanonicalPath();
                    Mat tempMat = Imgcodecs.imread(filePath, Imgcodecs.IMREAD_GRAYSCALE);
                    Imgproc.resize(tempMat, tempMat, new Size(SIDE, SIDE));
                    dataList.add(tempMat);
                    imagePath.add(filePath);
                    typeList.add(imageType);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public  double computeBiggestDistanceInDataSet(List<FaceVector> faces){
        int setSize = faces.size();
        double distance = 0;
        for(int i = 0; i < setSize; i++)
            for(int j = i + 1; j < setSize; j++){
                double temp = calcDistance(faces.get(i).getVector(), faces.get(j).getVector());
                if(distance < temp)
                    distance = temp;
            }
        return distance;
    }

    public SimpleMatrix convertImageToMatrix(Mat imageData){
        double[] imageArray = convertImageToColumnArray(imageData);
        double[][] data = new double[imageArray.length][1];
        for(int i = 0; i < imageArray.length; i++)
            data[i][0] = imageArray[i];
        return new SimpleMatrix(data);
    }

    public double[] convertImageToColumnArray(Mat mat){
        int arrayLength = mat.rows() * mat.cols();
        double[] doubleArray = new double[arrayLength];
        for(int i = 0; i < mat.rows(); i++)
            for(int j = 0; j < mat.cols(); j++){
                doubleArray[i * mat.cols()+j] = (mat.get(j, i))[0];
            }
        return doubleArray;
    }


    public double[] convertImageToColumnArray(double[][] mat){
        int arrayLength = mat.length * mat.length;
        double[] doubleArray = new double[arrayLength];
        for(int i = 0; i < mat.length; i++)
            for(int j = 0; j < mat.length; j++){
                doubleArray[i * mat.length+j] = mat[j][i];
            }
        return doubleArray;
    }

    public SimpleMatrix convertImagesSpaceToCovarianceMatrix(){
        int dataSize = trainingImageData.size();
        double [][] tempMatrix = new double[dataSize][];
        double [] temp = new double[SIDE * SIDE];
        double[][]meanImageData = new double[SIDE * SIDE][1];
        for(int i = 0; i < temp.length; i++){
            temp[i] = 0;
        }
        for(int i = 0; i < dataSize; i++){
            tempMatrix[i] = convertImageToColumnArray(trainingImageData.get(i));
            for(int j = 0; j < SIDE * SIDE; j++){
                temp[j] += tempMatrix[i][j];
            }
        }
        for(int i = 0; i < SIDE * SIDE; i++){
            temp[i] /= dataSize;
            meanImageData[i][0] = temp[i];
        }
        meanMatrix = new SimpleMatrix(meanImageData);
        for(int i = 0; i < dataSize; i++)
            for(int j = 0; j < SIDE * SIDE; j++)
                tempMatrix[i][j] -= temp[j];
        return new SimpleMatrix(tempMatrix).transpose();
    }

    public List<EigenVectorSet> getBestEigenVectors(EigenVectorSet[] eigenVectorSets){
        List<EigenVectorSet> list = new ArrayList<>();
        double sum = 0;
        for(int i = 0; i < eigenVectorSets.length; i++){
            sum += eigenVectorSets[i].getEigenValue();
        }
        double temp = 0;
        for(int i = 0; i < eigenVectorSets.length; i++){
            temp += eigenVectorSets[i].getEigenValue();
            if(temp / sum >= THRESHOLD)
                break;
            else{
                list.add(eigenVectorSets[i]);
            }
        }
        return list;
    }

    public SimpleMatrix createOriginalSpaceEigenVectorMatrix(List<EigenVectorSet> eigenSet, SimpleMatrix matrix){
        double[][] matrixData = new double[SIDE * SIDE][eigenSet.size()];
        int count = 0;
        for(EigenVectorSet vector: eigenSet){
            SimpleMatrix tempVector = vector.getEigenVector();
            SimpleMatrix newVector = matrix.mult(tempVector);
            newVector = newVector.divide(calculateModule(newVector));
            for(int i = 0; i < SIDE * SIDE; i++){
                matrixData[i][count] = newVector.get(i, 0);
            }
            count++;
        }
        return new SimpleMatrix(matrixData);
    }

    public double calculateModule(SimpleMatrix matrix){
        double sum = 0;
        for(int i = 0; i < matrix.numCols(); i++)
            for(int j = 0; j < matrix.numRows(); j++){
                double temp =matrix.get(j, i);
                sum+= temp*temp;
            }
        return Math.sqrt(sum);
    }

    public List<FaceVector> projectTrainingImagesToSpace(List<Integer> labels, SimpleMatrix imagesMatrix, SimpleMatrix space){
        List<FaceVector> projectedImageList = new ArrayList<>();
        for(int i = 0; i < labels.size(); i++){
            double[][] matrixArray = new double[SIDE * SIDE][1];
            for(int j = 0; j < SIDE * SIDE; j++){
                matrixArray[j][0] = imagesMatrix.get(j, i);
            }
            SimpleMatrix imageMatrix = new SimpleMatrix(matrixArray);
            SimpleMatrix imageCoordinator = space.mult(imageMatrix);
            projectedImageList.add(new FaceVector(String.valueOf(labels.get(i)), imageCoordinator));
        }
        return projectedImageList;
    }

    public double calcDistance(SimpleMatrix matrix1, SimpleMatrix matrix2){
        if(matrix1.numCols() != matrix2.numCols() || matrix1.numRows() != matrix2.numRows()){
            return -1;
        }
        double distance = 0;
        for(int i = 0; i < matrix1.numRows(); i++)
            for(int j = 0; j < matrix1.numCols(); j++) {
                double temp = matrix1.get(i, j) - matrix2.get(i, j);
                distance += temp * temp;
            }
        return Math.sqrt(distance);
    }

}
