package network;

import kernel.Kernel;

import java.util.Arrays;
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
	private int[] neuronIndices;
	private int[] weightIndices;

	// this array holds the current values for the network
	private float[] values;
	// this array holds the weights for each synapse
	private float[] weights;

	private LayerKernel forwardKernel = new LayerKernel() {
		@Override
		public void run(int i) {
			int currentIndex = neuronIndices[getDataLayer()];
			int endIndex = neuronIndices[getDataLayer() + 1];
			int targetIndex = neuronIndices[getTargetLayer()] + i;
			int synapseIndex = weightIndices[getDataLayer()] + i * (endIndex - currentIndex);

			values[targetIndex] = 0; // clear the value
			while (currentIndex < endIndex)
				values[targetIndex] += values[currentIndex++] * weights[synapseIndex++];

			// activate
			values[targetIndex] = activation(values[targetIndex]);
		}
	};

	private LayerKernel backwardKernel = new LayerKernel() {
		@Override
		public void run(int i) {
			int currentIndex = neuronIndices[getDataLayer()];
			int endIndex = currentIndex + neuronsPerLayer[getDataLayer()];
			int targetIndex = neuronIndices[getTargetLayer()] + i;
			int synapseIndex = weightIndices[getTargetLayer()] + i;

			// update weights
			// accumulate errors
			float error = 0;
			while (currentIndex < endIndex) {
				weights[synapseIndex] += values[targetIndex] * values[currentIndex];
				error += values[currentIndex] * weights[synapseIndex];
				synapseIndex += size();
				currentIndex++;
			}
			values[targetIndex] = values[targetIndex] * (1 - values[targetIndex]) * error;
		}
	};

	private ForwardLayerIterator forward;
	private BackwardLayerIterator backward;

	public FeedForwardNetwork(long seed, int... neuronsPerLayer) {
		this(seed, true, neuronsPerLayer);
	}

	public FeedForwardNetwork(int... neuronsPerLayer) {
		this(0, false, neuronsPerLayer);
	}

	private FeedForwardNetwork(long seed, boolean useSeed, int... neuronsPerLayer) {
		this.neuronsPerLayer = new int[neuronsPerLayer.length];
		System.arraycopy(neuronsPerLayer, 0, this.neuronsPerLayer, 0, neuronsPerLayer.length);

		neuronIndices = new int[neuronsPerLayer.length];
		weightIndices = new int[neuronsPerLayer.length - 1];

		int valuesSize = 0;
		for (int i = 0; i < neuronsPerLayer.length; i++) {
			int size = neuronsPerLayer[i];
			neuronIndices[i] = valuesSize;
			valuesSize += size;
		}

		int weightsSize = 0;
		int previous = neuronsPerLayer[0];
		for (int i = 0; i < neuronsPerLayer.length - 1; i++) {
			int size = neuronsPerLayer[i + 1];
			weightIndices[i] = weightsSize;
			weightsSize += previous * size;
			previous = size;
		}

		values = new float[valuesSize];
		weights = new float[weightsSize];

		Random random = useSeed ? new Random(seed) : new Random();
		for (int i = 0; i < weights.length; i++)
			weights[i] = random.nextFloat();
		Arrays.fill(weights, 0.5f);

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

		int i = 0;
		int currentIndex = neuronIndices[neuronIndices.length - 1];
		int endIndex = currentIndex + neuronsPerLayer[neuronsPerLayer.length - 1];
		while (currentIndex < endIndex) {
			float value = values[currentIndex];
			values[currentIndex++] = value * (1 - value) * (expected[i++] - value);
		}
		return backward;
	}

	private float activation(float value) {

		return (1f / (1 + (float) Math.exp(-value)));
	}

//	private float errorDerivative(float target, float output) {
//
//
//	}

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
	public void getOutputs(float[] outputs) {
		System.arraycopy(values, values.length - getOutputCount(), outputs, 0, getOutputCount());
	}

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
}
