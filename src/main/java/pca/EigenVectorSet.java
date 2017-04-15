package pca;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by nguye on 4/13/2017.
 */
public class EigenVectorSet implements Comparable<EigenVectorSet>{
    private SimpleMatrix mEigenVector;
    private double mEigenValue;
    public EigenVectorSet(double eigenValue, SimpleMatrix eigenVector){
        mEigenValue = eigenValue;
        mEigenVector = eigenVector;
    }

    public SimpleMatrix getEigenVector() {
        return mEigenVector;
    }

    public double getEigenValue() {
        return mEigenValue;
    }

    @Override
    public int compareTo(EigenVectorSet o) {
        double temp = mEigenValue - o.getEigenValue();
        if(temp < 0 )
            return 1;
        if(temp == 0)
            return 0;
        return -1;
    }
}
