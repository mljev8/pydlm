import numpy as np
import matplotlib.pyplot as plt

selected_color_style = 'jet'

# columns, rows
J,I = 1280,960
J,I =  640,480
J,I =  320,240
J,I =  160,120

R_circ = I//4             # radius
C_rect = (9*I//16, 2*J//5) # center
W_rect = (I//8, J//4)     # width

img_true = np.zeros([I,J], dtype='float32') # but think of it as unit8 [0,...,255]
for i in range(I):
    for j in range(J):
        # egg box
        img_true[i,j] = np.sin(2.*np.pi/20*(i+j)) + np.cos(2.*np.pi/20*j) # [-2., +2.]
        img_true[i,j] *= 15. # [-30.,  +30.]
        img_true[i,j] += 70. # [+40., +100.]
        # circle
        img_true[i,j] += 60.*float((i-I//2)**2 + (j-J//2)**2 <= R_circ**2)
        # rectangle
        img_true[i,j] += 60.*float(abs(i-C_rect[0])<W_rect[0] and abs(j-C_rect[1])<W_rect[1])

noise_realization = 15.*np.random.standard_normal(size=[I,J]) # [-40., +40.] roughly
img_noisy = img_true + noise_realization

print(' True image min/max {:.0f}/{:.0f}'.format( img_true.min(), img_true.max()))
print('Noisy image min/max {:.0f}/{:.0f}'.format(img_noisy.min(),img_noisy.max()))

img_true[:,:]  = np.fmax(np.fmin(img_true,  255.), 0.)
img_noisy[:,:] = np.fmax(np.fmin(img_noisy, 255.), 0.)

img_true  = img_true.astype('uint8')
img_noisy = img_noisy.astype('uint8')

plt.close('all')

fig,ax = plt.subplots()
ax.set_title('Simulated truth ({}x{})'.format(J,I))
ax.imshow(img_true, cmap=selected_color_style)

fig,ax = plt.subplots()
ax.set_title('Simulated truth + additive noise')
ax.imshow(img_noisy, cmap=selected_color_style)



plt.show(block=False)