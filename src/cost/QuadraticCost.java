package cost;

import activation.Activation;

/**
 * Created by Nyrmburk on 12/1/2016.
 */
public class QuadraticCost implements Cost {

	public float cost(float actual, float expected) {
		float toSquare = Math.abs(actual - expected);
		return 0.5f * toSquare * toSquare;
	}

	public float costDerivative(float actual, float expected, float value, Activation activation) {
		return (actual - expected) * activation.activationDerivative(value);
	}
}
