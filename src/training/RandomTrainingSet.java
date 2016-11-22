package training;

import java.util.List;
import java.util.Random;

/**
 * Created by Nyrmburk on 11/21/2016.
 */
public class RandomTrainingSet implements TrainingSet {

	private Random random = new Random();
	private List<TrainingData> data;

	public RandomTrainingSet(List<TrainingData> data) {
		this.data = data;
	}

	@Override
	public TrainingData next() {
		return data.get(random.nextInt(data.size()));
	}
}
