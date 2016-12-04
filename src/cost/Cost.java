package cost;

import activation.Activation;

/**
 * Created by Nyrmburk on 12/3/2016.
 */
public interface Cost {

	float cost(float actual, float expected);

	float costDerivative(float actual, float expected, float value, Activation activation);
}
