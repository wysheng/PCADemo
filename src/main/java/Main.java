import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import sun.java2d.pipe.SpanShapeRenderer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by nguye on 4/13/2017.
 */
public class Main {
    public static List<Integer> trainingImageTypes;
    public static List<Mat> trainingImageData;
    public static List<String> trainingImagePaths;
    public static final int SIDE = 250;
    public static final double THRESHOLD = 0.9999;
    public static SimpleMatrix meanMatrix;
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        readImage();
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
        SimpleMatrix originalSpace = createOriginalSpaceEigenVectorMatrix(bestVectors, matrix);
        SimpleMatrix originalSpaceTransposeMatrix = originalSpace.transpose();//U transpose
        List<FaceVector> faceVectors = projectTrainingImagesToSpace(trainingImageTypes, matrix, originalSpaceTransposeMatrix);

        List<String> validateImagePath = new ArrayList<>();
        List<Integer> typeList = new ArrayList<>();
        List<Mat> dataList = new ArrayList<>();
        String validatePath = "D:\\BTL\\datn\\demo_data\\test";
        loadValidateImage(validatePath, validateImagePath, dataList, typeList);
        double biggestDistance = computeBiggestDistanceInDataSet(faceVectors);
        SimpleMatrix testImage = convertImageToMatrix(dataList.get(4));

        SimpleMatrix projected = originalSpaceTransposeMatrix.mult(testImage.minus(meanMatrix));
        double distance = -1;
        String label = "unknown";
        String tempLabel = "";
        for(FaceVector faceVector: faceVectors){
            double temp = calcDistance(projected, faceVector.getVector());
            if(distance > temp || distance == -1) {
                distance = temp;
                tempLabel = faceVector.getLabel();
            }
        }
        if(distance <= biggestDistance / 2){
            System.out.println(tempLabel);
        }else{
            System.out.println(label);
        }
        System.out.println();
    }

    public static void loadValidateImage(String path, List<String> imagePath, List<Mat>dataList, List<Integer> typeList){
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

    public static double computeBiggestDistanceInDataSet(List<FaceVector> faces){
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

    public static void readImage(){
        trainingImageTypes = new ArrayList<>();
        trainingImageData = new ArrayList<>();
        trainingImagePaths = new ArrayList<>();
        String path = "D:\\BTL\\datn\\demo_data\\train";
        File parentDir = new File(path);
        for(File dir: parentDir.listFiles()){
            int type = Integer.parseInt(dir.getName());
            for(File file: dir.listFiles()){
                try {
                    String filePath = file.getCanonicalPath();
                    Mat tempMat = Imgcodecs.imread(filePath, Imgcodecs.IMREAD_GRAYSCALE);
                    Imgproc.resize(tempMat, tempMat, new Size(SIDE, SIDE));
                    trainingImageData.add(tempMat);
                    trainingImagePaths.add(filePath);
                    trainingImageTypes.add(type);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static SimpleMatrix convertImageToMatrix(Mat imageData){
        double[] imageArray = convertImageToColumnArray(imageData);
        double[][] data = new double[imageArray.length][1];
        for(int i = 0; i < imageArray.length; i++)
            data[i][0] = imageArray[i];
        return new SimpleMatrix(data);
    }

    public static double[] convertImageToColumnArray(Mat mat){
        int arrayLength = mat.rows() * mat.cols();
        double[] doubleArray = new double[arrayLength];
        for(int i = 0; i < mat.rows(); i++)
            for(int j = 0; j < mat.cols(); j++){
                doubleArray[i * mat.cols()+j] = (mat.get(j, i))[0];
            }
        return doubleArray;
    }


    public static double[] convertImageToColumnArray(double[][] mat){
        int arrayLength = mat.length * mat.length;
        double[] doubleArray = new double[arrayLength];
        for(int i = 0; i < mat.length; i++)
            for(int j = 0; j < mat.length; j++){
                doubleArray[i * mat.length+j] = mat[j][i];
            }
        return doubleArray;
    }

    public static SimpleMatrix convertImagesSpaceToCovarianceMatrix(){
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

    public static List<EigenVectorSet> getBestEigenVectors(EigenVectorSet[] eigenVectorSets){
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

    public static SimpleMatrix createOriginalSpaceEigenVectorMatrix(List<EigenVectorSet> eigenSet, SimpleMatrix matrix){
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

    public static double calculateModule(SimpleMatrix matrix){
        double sum = 0;
        for(int i = 0; i < matrix.numCols(); i++)
            for(int j = 0; j < matrix.numRows(); j++){
                double temp =matrix.get(j, i);
                sum+= temp*temp;
            }
        return Math.sqrt(sum);
    }

    public static List<FaceVector> projectTrainingImagesToSpace(List<Integer> labels, SimpleMatrix imagesMatrix, SimpleMatrix space){
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

    public static double calcDistance(SimpleMatrix matrix1, SimpleMatrix matrix2){
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
