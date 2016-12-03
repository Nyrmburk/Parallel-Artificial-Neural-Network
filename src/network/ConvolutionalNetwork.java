package network;

import kernel.Kernel;

import java.util.Iterator;

/**
 * Created by Nyrmburk on 12/1/2016.
 */
public class ConvolutionalNetwork implements Network {



	public ConvolutionalNetwork(LayerDefinition... layers) {


	}

	private void convolve() {

	}

	private void pool() {

	}

	@Override
	public Iterator<Kernel> forward() {
		return null;
	}

	@Override
	public Iterator<Kernel> backward(float[] expected) {
		return null;
	}

	@Override
	public int getInputCount() {
		return 0;
	}

	@Override
	public int getOutputCount() {
		return 0;
	}

	@Override
	public void setInputs(float[] inputs) {

	}

	@Override
	public float[] getOutputs() {
		return new float[0];
	}

	@Override
	public float[] getErrors() {
		return new float[0];
	}

	public static class LayerDefinition {

		private int width;
		private int height;
		private int depth;

		public LayerDefinition(int width, int height, int depth) {
			this.width = width;
			this.height = height;
			this.depth = depth;
		}

		public int getWidth() {
			return width;
		}

		public int getHeight() {
			return height;
		}

		public int getDepth() {
			return depth;
		}
	}

	private static class Layer {

		private float[] values;
		private int width;
		private int height;
		private int depth;

		public Layer(int width, int height, int depth) {
			values = new float[width * height * depth];
			this.width = width;
			this.height = height;
			this.depth = depth;
		}

		public int getIndex(int x, int y, int z) {
			int area = width * height;
			return z * area + y * x + x;
		}

		public float[] getValues() {
			return values;
		}

		public int getWidth() {
			return width;
		}

		public int getHeight() {
			return height;
		}

		public int getDepth() {
			return depth;
		}
	}

	private static class ConvolutionalLayer extends Layer {

		private int padding;
		public ConvolutionalLayer(int width, int height, int depth, int padding) {
			super(width + padding + padding,
					height + padding + padding,
					depth + padding + padding);
			this.padding = padding;
		}

		public int getPadding() {
			return padding;
		}
	}

	private static class PoolingLayer extends Layer {

		public PoolingLayer(Layer layer, int factor) {
			super(layer.getWidth() / factor,
					layer.getHeight() / factor,
					layer.getDepth() / factor);
		}
	}
}
