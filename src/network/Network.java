package network;

import kernel.Kernel;

import java.util.Iterator;

/**
 * The network interface is designed to provide a black box that fits most, if not all networks.
 * It is also designed so that each network can be multithreaded using the kernel pattern.
 * To use it, iterate through and execute each kernel in order for the forward and backward methods.
 * <p>
 * Created by Nyrmburk on 11/10/2016.
 */
public interface Network {

	Iterator<Kernel> forward();

	Iterator<Kernel> backward(float[] expected);

	int getInputCount();

	int getOutputCount();

	void setInputs(float[] inputs);

	void getOutputs(float[] outputs);
}
