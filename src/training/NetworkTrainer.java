package training;

import network.NetworkExecutor;

/**
 * Created by Nyrmburk on 11/21/2016.
 */
public class NetworkTrainer implements Trainer {

	private NetworkExecutor network;
	private TrainingSet trainingSet;

	public NetworkTrainer(NetworkExecutor network, TrainingSet trainingSet) {
		this.network = network;
		this.trainingSet = trainingSet;
	}

	public int train(float error, int iterations) {

		int currentIteration = 0;
		float currentError;

		do {
			TrainingData data = trainingSet.next();
			currentError = network.train(data.getInputs(), data.getExpectedOutputs());
			currentIteration++;
		} while (currentIteration < iterations && Math.abs(currentError) > error);

		return currentIteration;
	}
}
