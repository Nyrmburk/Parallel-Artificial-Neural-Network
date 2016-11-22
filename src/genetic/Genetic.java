package genetic;

/**
 * Created by Nyrmburk on 11/21/2016.
 */
public interface Genetic<T> {

	void mutate(float mutationAmount);
	Genetic<T> crossover(Genetic<T> a, Genetic<T> b);
	float getFitness();
}
