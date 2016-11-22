package kernel;

/**
 * Created by Nyrmburk on 11/20/2016.
 */
public class SimpleKernelExecutor implements KernelExecutor {

	@Override
	public void execute(Kernel kernel) {
		for (int i = 0; i < kernel.size(); i++)
			kernel.run(i);
	}
}
