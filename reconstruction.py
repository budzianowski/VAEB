from VAEB1 import VAEB
import VAEBImage

size_continuous_latent_space = [10]
continuous_file = 'reconstruction_res/VAE_continuous_{0}.mdl'

def MSE(y, t) :
    return T.mean(T.pow(y - t, 2))

def reconstruction_test(x_test, model, file_prefix, continuous) :
    for num_samples in [0, 100] :
        print('num_samples :\n{0}'.format(num_samples))
        #get and print a few examples
        for i in range(8) :
            print('i :\n{0}'.format(i))
            if continuous:
                mu, log_sigma = model.reconstruct(x_test[i], num_samples)
                I = np.eye(mu.shape[1])
                cov = (np.exp(log_sigma)**2) * I
                sample = np.random.multivariate_normal(mu.reshape(560), cov)  # 560 pixels
            else :
                sample = model.reconstruct(x_test[i], num_samples)
            VAEBImage.save_image(x_test[i], file_prefix + '_image_{0}_{1}_original.jpg'.format(num_samples, i))
            VAEBImage.save_image(sample.eval(), file_prefix + '_image_{0}_{1}_sample.jpg'.format(num_samples, i))

        #get mean squared error of reconstruction


def main() :
    for s in size_continuous_latent_space :
        filename = continuous_file.format(s)
        model, data = VAEB.load('./test_discrete.mdl')

        if model.continuous :
            (x_train, x_test) = data
        else :
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data

        reconstruction_test(x_test, model, "reconstruction_res/contiuous_{0}_".format(s), model.continuous)

if __name__ == '__main__' :
    main()
