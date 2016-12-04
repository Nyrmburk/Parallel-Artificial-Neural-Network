package activation;

/**
 * Created by Nyrmburk on 12/3/2016.
 */
public class LinearActivation implements Activation {

	@Override
	public float activation(float value) {
		return value;
	}

	@Override
	public float activationDerivative(float activationResult) {
		return 1;
	}
}
