import kernel.Kernel;
import kernel.KernelExecutor;
import kernel.SimpleKernelExecutor;
import network.*;

import java.util.Arrays;
import java.util.Iterator;

/**
 * Created by Nyrmburk on 11/20/2016.
 */
public class Main {

	public static void main(String[] args) {

		long time;
		time = System.currentTimeMillis();
		Network network = new FeedForwardNetwork(0L, 1, 100, 2);
		network.setInputs(new float[]{1});
		KernelExecutor executor = new SimpleKernelExecutor();
		System.out.println("initialization took: " + (System.currentTimeMillis() - time));

		NetworkExecutor networkExecutor = new NetworkExecutor(network, executor);

		float[] inputs = new float[]{0.5f};
		float[] expected = new float[]{0.5f, 0.3f};
//		for (int i = 0; i < 100; i++)
//			System.out.printf("%.6f\n", networkExecutor.train(inputs, expected));

		for (int i = 0; i < 30; i++) {
			network.setInputs(new float[]{0.5f});
			Iterator<Kernel> forward = network.forward();
			while (forward.hasNext())
				executor.execute(forward.next());
			System.out.print(Arrays.toString(network.getOutputs()) + ", ");

			Iterator<Kernel> backward = network.backward(new float[]{0.5f, 0.3f});
			while (backward.hasNext())
				executor.execute(backward.next());

			System.out.println(Arrays.toString(network.getErrors()));
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
