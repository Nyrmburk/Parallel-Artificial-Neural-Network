package network;

/**
 * Created by Nyrmburk on 12/1/2016.
 */
public class CrossEntropyCost implements Cost {

	public float cost(float actual, float expected) {

		float result = (1f - expected) * (float) Math.log(1d - actual);
		return Float.isNaN(result) ? 0f : result;
	}

	public float costDerivative(float actual, float expected, float value, Activation activation) {
		return (actual - expected);
	}
}
