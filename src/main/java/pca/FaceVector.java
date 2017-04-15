package pca;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by nguye on 4/13/2017.
 */
public class FaceVector {
    private String mLabel;
    private SimpleMatrix mVector;

    public FaceVector(String label, SimpleMatrix vector){
        mLabel = label;
        mVector = vector;
    }

    public String getLabel() {
        return mLabel;
    }

    public SimpleMatrix getVector() {
        return mVector;
    }
}
