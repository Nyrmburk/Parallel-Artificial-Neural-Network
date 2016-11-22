package training;

import kernel.Kernel;
import kernel.KernelExecutor;
import network.Network;

import java.util.Iterator;

/**
 * Created by Nyrmburk on 11/21/2016.
 */
public class BackPropagationNetworkTrainer implements Trainer {

	private Network network;
	private TrainingSet trainingSet;
	private KernelExecutor executor;

	public BackPropagationNetworkTrainer(Network network, TrainingSet trainingSet, KernelExecutor executor) {
		this.network = network;
		this.trainingSet = trainingSet;
		this.executor = executor;
	}

	public void train(float error, int iterations) {

		int currentIteration = 0;
		float currentError = 1;

		float[] outputs = new float[network.getOutputCount()];
		do {
			TrainingData data = trainingSet.next();

			network.setInputs(data.getInputs());
			Iterator<Kernel> forward = network.forward();
			while (forward.hasNext())
				executor.execute(forward.next());
			network.getOutputs(outputs);
			// calculate errors

			Iterator<Kernel> backward = network.backward(data.getExpectedOutputs());
			while (backward.hasNext())
				executor.execute(backward.next());

		} while (currentIteration < iterations && currentError > error);
	}
}
