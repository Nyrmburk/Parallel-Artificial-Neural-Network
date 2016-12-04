package activation;

/**
 * Created by Nyrmburk on 12/3/2016.
 */
public class SigmoidActivation implements Activation {

	public float activation(float value) {
		return (1f / (1f + (float) Math.exp(-value)));
	}

	public float activationDerivative(float activationResult) {
		return activationResult * (1f - activationResult);
	}
}
