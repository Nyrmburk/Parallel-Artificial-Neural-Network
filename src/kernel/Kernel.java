package kernel;

/**
 * A kernel is a structure defined to be able to execute in parallel without any collisions.
 * In order to make this true, the kernel may have to implement it's own synchronization.
 *
 * Created by Nyrmburk on 11/10/2016.
 */
public interface Kernel {

	void run(int i);
	int size();
}
