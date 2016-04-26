import numpy as np
from VAEB import VAEB
import VAEBImage

size_continuous_latent_space = [2, 10, 20]
model_file = 'reconstruction_res/{0}_{1}.mdl'
log_file = 'reconstruction_res/MSE.res'

def MSE(model, x_test, num_samples) :
    if not model.continuous :
        samples = model.reconstruct(x_test, num_samples)
        return np.mean(np.linalg.norm(samples - x_test, axis=1)**2)
    else :
        total = 0.0
        for i in range(x_test.shape[0]) :
            sample = model.reconstruct(x_test[i], num_samples)
            total += np.power(np.linalg.norm(sample - x_test[i]), 2)
        return total / x_test.shape[0]

def reconstruction_test(x_test, model, file_prefix, continuous) :
    for num_samples in [0, 20] :
        print('num_samples :\n{0}'.format(num_samples))
        #get and print a few examples
        for i in range(8) :
            #if continuous:
            #    print('x_test[i] :\n{0}'.format(x_test[i]))
            #    mu, log_sigma = model.reconstruct(x_test, num_samples)
            #    I = np.eye(mu.shape[1])
            #    cov = (np.exp(log_sigma)**2) * I
            #    sample = np.random.multivariate_normal(mu.reshape(560), cov)  # 560 pixels
            #else :
            sample = model.reconstruct(x_test[i], num_samples)
            VAEBImage.save_image(x_test[i], file_prefix + '_image_{0}_{1}_original.jpg'.format(num_samples, i))
            VAEBImage.save_image(sample, file_prefix + '_image_{0}_{1}_sample.jpg'.format(num_samples, i))

        #get mean squared error of reconstruction
        mse = MSE(model, x_test, num_samples)
        with open(log_file, 'a') as f :
            f.write('{0},{1},{2},{3}\n'.format('continuous' if continuous else 'discrete',
                                               model.n_latent,
                                               'mean' if num_samples == 0 else 'sample',
                                               mse))


def main() :
    with open(log_file, 'w') as f :
        f.write('data_type,latent_size,sample_type,MSE\n')
    for data_type in ['discrete', 'continuous'] :
        for s in size_continuous_latent_space :
            filename = model_file.format(data_type, s)
            model, data = VAEB.load(filename)

            if model.continuous :
                (x_train, x_test) = data
            else :
                (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data

            reconstruction_test(x_test, model, "reconstruction_res/{0}_{1}_".format(data_type, s), model.continuous)

if __name__ == '__main__' :
    main()
