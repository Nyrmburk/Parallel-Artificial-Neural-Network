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

		Network network = new FeedForwardNetwork(0L, 1, 3, 2);
		network.setInputs(new float[]{1});

		KernelExecutor executor = new SimpleKernelExecutor();
		Iterator<Kernel> forward = network.forward();
		while (forward.hasNext())
			executor.execute(forward.next());

		float[] out = {0, 0};
		network.getOutputs(out);
		System.out.println(Arrays.toString(out));

		Iterator<Kernel> backward = network.backward(new float[]{1, 2});
		while (backward.hasNext())
			executor.execute(backward.next());

		network.setInputs(new float[]{1});
		forward = network.forward();
		while (forward.hasNext())
			executor.execute(forward.next());
		network.getOutputs(out);
		System.out.println(Arrays.toString(out));

		final long time = System.currentTimeMillis();
		Kernel count = new Kernel() {
			@Override
			public void run(int i) {
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				System.out.println(i + ": " + Thread.currentThread().getName() + ", @:" + (System.currentTimeMillis() - time));
			}

			@Override
			public int size() {
				return 7;
			}
		};

		KernelExecutor executor1 = new ParallelKernelExecutor();
//		while (true) {
			executor1.execute(count);
//		}
		System.out.println(System.currentTimeMillis() - time);
		System.out.println("finished");

		forward = network.forward();
		while (forward.hasNext())
			executor1.execute(forward.next());
		network.getOutputs(out);
		System.out.println(out[0]);
		System.exit(0);
	}
}
