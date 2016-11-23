package kernel;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Nyrmburk on 11/20/2016.
 */
public class ParallelKernelExecutor implements KernelExecutor {

	private final Object monitor = new Object();
	private final Object finishedMonitor = new Object();
	private Worker[] workers;
	private AtomicInteger activeThreads = new AtomicInteger();
	private AtomicInteger index = new AtomicInteger();
	private int workDistribution;

	private Kernel kernel;

	public ParallelKernelExecutor() {
		this(Runtime.getRuntime().availableProcessors());
	}

	public ParallelKernelExecutor(int threads) {

		workers = new Worker[threads];
		for (int i = 0; i < threads; i++) {
			workers[i] = new Worker(this);
			new Thread(workers[i], "ParallelKernelExecutor-" + i).start();
		}
	}

	synchronized private void getData(Worker worker) {
		while (kernel == null || index.get() >= kernel.size()) {
			synchronized (monitor) {
				try {
					monitor.wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		worker.from = index.getAndAdd(workDistribution);
		worker.to = worker.from + workDistribution;
		worker.to = worker.to > kernel.size() ? kernel.size() : worker.to;
		worker.kernel = kernel;
	}

	@Override
	public void execute(Kernel kernel) {

		index.set(0);
		workDistribution = kernel.size() / workers.length;
		if (workDistribution == 0)
			workDistribution = 1;

		synchronized (monitor) {
			this.kernel = kernel;
			monitor.notifyAll();
		}

		waitForWorkers();

		this.kernel = null;
	}

	private void waitForWorkers() {
		synchronized (finishedMonitor) {
			while (activeThreads.get() > 0 || kernel.size() > index.get()) {
				try {
					finishedMonitor.wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private static class Worker implements Runnable {

		private int from;
		private int to;
		private Kernel kernel;

		private ParallelKernelExecutor executor;

		public Worker(ParallelKernelExecutor executor) {
			this.executor = executor;
		}

		@Override
		public void run() {

			while (true) {
				executor.getData(this);
				executor.activeThreads.incrementAndGet();

				while (from < to)
					kernel.run(from++);
				kernel = null;

				synchronized (executor.finishedMonitor) {
					executor.activeThreads.decrementAndGet();
					executor.finishedMonitor.notify();
				}
			}
		}
	}
}
