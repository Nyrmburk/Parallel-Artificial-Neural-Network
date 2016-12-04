package network;

import activation.Activation;
import activation.SigmoidActivation;
import cost.Cost;
import cost.CrossEntropyCost;
import kernel.Kernel;

import java.io.*;
import java.util.Iterator;
import java.util.Random;

/**
 * This network type is a fully connected feed-forward network.
 * <p>
 * Created by Nyrmburk on 11/10/2016.
 */
public class FeedForwardNetwork implements Network {

	// network topology
	// each value in the array represents the number of neurons are in that layer
	//
	// [2,3,1]
	//  O O O
	//  O O
	//    O
	private int[] neuronsPerLayer;

	// each value holds the neuron starting index of each layer
	private int[] neuronIndices;

	// each value holds the weight starting index of each layer
	private int[] weightIndices;

	// this array holds the current values for the network
	private float[] values;

	// this array holds the biases for each neuron
	private float[] biases;

	// this array holds the weights for each weight
	private float[] weights;

	// pre-allocated arrays for getting outputs and errors
	private float[] outputs;
	private float[] errors;

	private float learningRate = 0.05f;
	private float momentumWeight = 0;

	private Activation activation = new SigmoidActivation();
	private Cost cost = new CrossEntropyCost();

	// this is the logic behind feed-forward propagation
	private LayerKernel forwardKernel = new LayerKernel() {
		@Override
		public void run(int i) {

			// calculate the indices necessary for execution
			int currentIndex = neuronIndices[getDataLayer()];
			int endIndex = neuronIndices[getDataLayer() + 1];
			int targetIndex = neuronIndices[getTargetLayer()] + i;
			int weightIndex = weightIndices[getDataLayer()] + i * (endIndex - currentIndex);

			// feed the values forward
			values[targetIndex] = 0; // clear the previous value
			while (currentIndex < endIndex)
				values[targetIndex] += values[currentIndex++] * weights[weightIndex++];

			// activate
			values[targetIndex] = activation.activation(values[targetIndex] + biases[targetIndex]);
		}
	};

	// logic behind back-propagation
	private LayerKernel backwardKernel = new LayerKernel() {
		@Override
		public void run(int i) {

			// calculate indices
			int currentIndex = neuronIndices[getDataLayer()];
			int endIndex = currentIndex + neuronsPerLayer[getDataLayer()];
			int targetIndex = neuronIndices[getTargetLayer()] + i;
			int weightIndex = weightIndices[getTargetLayer()] + i;
			final int size = size();

			// update weights
			// accumulate errors
			float error = 0;
			while (currentIndex < endIndex) {

				weights[weightIndex] -= getLearningRate() * values[targetIndex] * values[currentIndex];
				biases[currentIndex] -= getLearningRate() * values[currentIndex];
				error += values[currentIndex] * weights[weightIndex];

				// incrementing the weight index by a non-1 value causes the entire kernel to take 10 times longer.
				// perhaps it is a cache miss?
				// the problem is that the prefetcher will ignore constant stride of large numbers
				// investigate to see if there is a way to traverse the memory in a way that plays better with the prefetcher
				weightIndex += size; // <- this line of code directly affects speed
				currentIndex++;
			}
			values[targetIndex] = error;//activationDerivative(values[targetIndex]) * error;
		}
	};

	// the iterators that return the kernels for the forward and backward operations
	private ForwardLayerIterator forward;
	private BackwardLayerIterator backward;

	// instantiate a network from a seed (for determinism)
	public FeedForwardNetwork(long seed, int... neuronsPerLayer) {
		this(seed, true, neuronsPerLayer);
	}

	// instantiate a network with a random seed (most likely case)
	public FeedForwardNetwork(int... neuronsPerLayer) {
		this(0, false, neuronsPerLayer);
	}

	// the actual constructor
	private FeedForwardNetwork(long seed, boolean useSeed, int... neuronsPerLayer) {
		this.neuronsPerLayer = new int[neuronsPerLayer.length];
		System.arraycopy(neuronsPerLayer, 0, this.neuronsPerLayer, 0, neuronsPerLayer.length);

		neuronIndices = new int[neuronsPerLayer.length];
		weightIndices = new int[neuronsPerLayer.length - 1];

		// calculate neuron indices
		int valuesSize = 0;
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			int size = neuronsPerLayer[i];
			neuronIndices[i] = valuesSize;
			valuesSize += size;
		}

		// calculate weight indices
		int weightsSize = 0;
		int previous = neuronsPerLayer[0];
		for (int i = 0; i < neuronsPerLayer.length - 1; i++) {
			int size = neuronsPerLayer[i + 1];
			weightIndices[i] = weightsSize;
			weightsSize += previous * size;
			previous = size;
		}

		values = new float[valuesSize];
		biases = new float[valuesSize];
		weights = new float[weightsSize];
		outputs = new float[getOutputCount()];
		errors = new float[getOutputCount()];

		// fill the weights with random values
		Random random = useSeed ? new Random(seed) : new Random();
		for (int i = 0; i < weights.length; i++)
			weights[i] = random.nextFloat();

		forward = new ForwardLayerIterator(this.neuronsPerLayer, forwardKernel);
		backward = new BackwardLayerIterator(this.neuronsPerLayer, backwardKernel);
	}

	@Override
	public Iterator<Kernel> forward() {
		forward.reset();
		return forward;
	}

	@Override
	public Iterator<Kernel> backward(float[] expected) {
		backward.reset();

		// perform error calculation of output neurons
		int i = 0;
		int currentIndex = neuronIndices[neuronIndices.length - 1];
		int endIndex = currentIndex + neuronsPerLayer[neuronsPerLayer.length - 1];
		while (currentIndex < endIndex) {
			errors[i] = cost.costDerivative(values[currentIndex], expected[i], values[currentIndex], activation);
			values[currentIndex] = errors[i];
			currentIndex++;
			i++;
		}

		return backward;
	}

	@Override
	public int getInputCount() {
		return neuronsPerLayer[0];
	}

	@Override
	public int getOutputCount() {
		return neuronsPerLayer[neuronsPerLayer.length - 1];
	}

	@Override
	public void setInputs(float[] inputs) {
		System.arraycopy(inputs, 0, values, 0, getInputCount());
	}

	@Override
	public float[] getOutputs() {
		System.arraycopy(values, values.length - getOutputCount(), outputs, 0, getOutputCount());
		return outputs;
	}

	@Override
	public float[] getErrors() {
		return errors;
	}

	public float getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(float learningRate) {
		this.learningRate = learningRate;
	}

	public float getMomentumWeight() {
		return momentumWeight;
	}

	public void setMomentumWeight(float momentumWeight) {
		this.momentumWeight = momentumWeight;
	}

	public Activation getActivation() {
		return activation;
	}

	public void setActivation(Activation activation) {
		this.activation = activation;
	}

	public Cost getCost() {
		return cost;
	}

	public void setCost(Cost cost) {
		this.cost = cost;
	}

	// generate a new kernel for each next layer
	private class ForwardLayerIterator implements Iterator<Kernel> {

		private int[] neuronsPerLayer;
		LayerKernel kernel;
		private int currentLayer = 0;

		public ForwardLayerIterator(int[] neuronsPerLayer, LayerKernel kernel) {
			this.neuronsPerLayer = neuronsPerLayer;
			this.kernel = kernel;
			reset();
		}

		@Override
		public boolean hasNext() {
			return currentLayer < neuronsPerLayer.length - 1;
		}

		@Override
		public Kernel next() {
			kernel.setIndices(currentLayer + 1, currentLayer, neuronsPerLayer[currentLayer + 1]);
			currentLayer++;
			return kernel;
		}

		protected void reset() {
			currentLayer = 0;
		}
	}

	// generate a new kernel for each previous layer
	private class BackwardLayerIterator implements Iterator<Kernel> {

		private int[] neuronsPerLayer;
		LayerKernel kernel;
		private int currentLayer;

		public BackwardLayerIterator(int[] neuronsPerLayer, LayerKernel kernel) {
			this.neuronsPerLayer = neuronsPerLayer;
			this.kernel = kernel;
			reset();
		}

		@Override
		public boolean hasNext() {
			return currentLayer > 0;
		}

		@Override
		public Kernel next() {
			kernel.setIndices(currentLayer - 1, currentLayer, neuronsPerLayer[currentLayer - 1]);
			currentLayer--;
			return kernel;
		}

		protected void reset() {
			currentLayer = neuronsPerLayer.length - 1;
		}
	}

	private abstract class LayerKernel implements Kernel {

		private int size;
		private int targetLayer, dataLayer;

		@Override
		public int size() {
			return size;
		}

		public void setIndices(int targetLayer, int dataLayer, int size) {
			this.targetLayer = targetLayer;
			this.dataLayer = dataLayer;
			this.size = size;
		}

		public int getTargetLayer() {
			return targetLayer;
		}

		public int getDataLayer() {
			return dataLayer;
		}
	}

	public static void save(File file, FeedForwardNetwork network) throws IOException {

		// binary file is as follows:
		// topology length
		// topology
		// learning rate
		// momentum rate
		// weights

		DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
		out.writeInt(network.neuronsPerLayer.length);
		for (int i = 0; i < network.neuronsPerLayer.length; i++)
			out.writeInt(network.neuronsPerLayer[i]);
		out.writeFloat(network.getLearningRate());
		out.writeFloat(network.getMomentumWeight());
		for (int i = 0; i < network.weights.length; i++)
			out.writeFloat(network.weights[i]);
		out.close();
	}

	public static FeedForwardNetwork load(File file) throws IOException {

		DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));

		int[] topology = new int[in.readInt()];
		for (int i = 0; i < topology.length; i++)
			topology[i] = in.readInt();
		float learningRate = in.readFloat();
		float momentumWeight = in.readFloat();

		FeedForwardNetwork network = new FeedForwardNetwork(topology);

		network.setLearningRate(learningRate);
		network.setMomentumWeight(momentumWeight);

		for (int i = 0; i < network.weights.length; i++)
			network.weights[i] = in.readFloat();

		in.close();

		return network;
	}
}
