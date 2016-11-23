import kernel.Kernel;
import kernel.KernelExecutor;
import kernel.ParallelKernelExecutor;
import kernel.SimpleKernelExecutor;
import network.FeedForwardNetwork;
import network.Network;

import java.util.Arrays;
import java.util.Iterator;

/**
 * Created by Nyrmburk on 11/20/2016.
 */
public class Main {

	public static void main(String[] args) {

		long time;
		time = System.currentTimeMillis();
		Network network = new FeedForwardNetwork(0L, 1,10, 2);
		network.setInputs(new float[]{1});
		KernelExecutor executor = new SimpleKernelExecutor();
		System.out.println("initialization took: " + (System.currentTimeMillis() - time));

		for (int i = 0; i < 10; i++) {
			Iterator<Kernel> forward = network.forward();
			while (forward.hasNext())
				executor.execute(forward.next());
		}

		time = System.currentTimeMillis();

		Iterator<Kernel> forward = network.forward();
		while (forward.hasNext())
			executor.execute(forward.next());

		float[] out = {0, 0};
		network.getOutputs(out);
		System.out.println(Arrays.toString(out));
		System.out.println("feed forward took: " + (System.currentTimeMillis() - time));

		for (int i = 0; i < 3; i++) {
			Iterator<Kernel> backward = network.backward(new float[]{1, 2});
			while (backward.hasNext())
				executor.execute(backward.next());
		}

		time = System.currentTimeMillis();
		Iterator<Kernel> backward = network.backward(new float[]{1, 2});
		while (backward.hasNext())
			executor.execute(backward.next());
		System.out.println("backpropagation took: " + (System.currentTimeMillis() - time));

		network.setInputs(new float[]{1});
		forward = network.forward();
		while (forward.hasNext())
			executor.execute(forward.next());
		network.getOutputs(out);
		System.out.println(Arrays.toString(out));

		for (int i = 0; i < 5000; i++) {
			network.setInputs(new float[]{0.5f});
			forward = network.forward();
			while (forward.hasNext())
				executor.execute(forward.next());
			network.getOutputs(out);
			System.out.println(Arrays.toString(out));

			backward = network.backward(new float[]{0.5f, 0.3f});
			while (backward.hasNext())
				executor.execute(backward.next());
		}

//		KernelExecutor executor1 = new ParallelKernelExecutor();
//
//		time = System.currentTimeMillis();
//		forward = network.forward();
//		while (forward.hasNext())
//			executor1.execute(forward.next());
//		network.getOutputs(out);
//		System.out.println(Arrays.toString(out));
//		System.out.println("parallel feed forward took: " + (System.currentTimeMillis() - time));
		System.exit(0);
	}
}
