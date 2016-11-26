package network;

import kernel.Kernel;
import kernel.KernelExecutor;

import java.util.Iterator;

/**
 * Created by Nyrmburk on 11/21/2016.
 */
public class NetworkExecutor {

	private KernelExecutor executor;
	private Network network;

	public NetworkExecutor(Network network, KernelExecutor executor) {

		this.executor = executor;
		this.network = network;
	}

	public float[] forward(float[] inputs) {

		network.setInputs(inputs);

		Iterator<Kernel> forward = network.forward();
		while (forward.hasNext())
			executor.execute(forward.next());

		return network.getOutputs();
	}

	public float train(float[] inputs, float[] expected) {

		forward(inputs);

		Iterator<Kernel> backward = network.backward(expected);
		while (backward.hasNext())
			executor.execute(backward.next());

		return calculateError(network.getErrors());
	}

	private static float calculateError(float[] errors) {
		float errorSum = 0;
		for (int i = 0; i < errors.length; i++)
			errorSum += errors[i];
		errorSum /= errors.length;
		return errorSum;
	}
}
