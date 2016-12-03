package network;

/**
 * Created by Nyrmburk on 12/3/2016.
 */
public interface Activation {

	// the activation function for each neuron after it has the inputs summed
	float activation(float value);

	float activationDerivative(float activationResult);
}
