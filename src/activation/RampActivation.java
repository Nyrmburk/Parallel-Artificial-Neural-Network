package activation;

/**
 * Created by Nyrmburk on 12/3/2016.
 */
public class RampActivation implements Activation {

	@Override
	public float activation(float value) {
		// max(0, value);
		// this implementation is faster than Math.max() because it assumes the values are safe
		return value >= 0f ? value : 0;
	}

	@Override
	public float activationDerivative(float activationResult) {
		return activationResult >= 0f ? 1 : 0;
	}
}
