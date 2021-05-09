import tensorflow as tf
import numpy as np
import pylab
from mayavi import mlab
mlab.init_notebook(backend='png')



## generate data
def f(x, y):
    # Surface...
    return (np.add(np.square(np.add(np.subtract(1.5, x), np.multiply(x, y))),
            np.add(np.square(np.add(np.subtract(2.25, x), np.multiply(x, np.square(y)))),
            np.square(np.add(np.subtract(2.625, x), np.multiply(x, np.power(y, 3)))))))

#Bounds...
axisMin = -4.5
axisMax = 4.5

x = np.linspace(axisMin, axisMax, 100)
y = np.linspace(axisMin, axisMax, 100)

#Starting position...
def init():
    Xopt = tf.Variable(3.0) 
    Yopt = tf.Variable(-3.5) 
    return Xopt, Yopt

X, Y = np.meshgrid(x, y, indexing='ij')
Z = f(X, Y)



def lineAndPoints(opt, epochs):
    Xopt, Yopt = init()

    Xprogress = [0]*epochs
    Yprogress = [0]*epochs

    for epoch in range(epochs):
        Xprogress[epoch] = Xopt.numpy()
        Yprogress[epoch] = Yopt.numpy()
        
        opt.minimize(lambda: (tf.add_n([tf.square(tf.add(tf.subtract(1.5, Xopt), tf.multiply(Xopt,
                        Yopt))),tf.square(tf.add(tf.subtract(2.25, Xopt), tf.multiply(Xopt,
                        tf.square(Yopt)))),tf.square(tf.add(tf.subtract(2.625, Xopt),
                        tf.multiply(Xopt, tf.pow(Yopt, 3))))])), var_list=[Xopt, Yopt])

    startX = Xprogress[0]
    startY = Yprogress[0]
    minX = Xopt.numpy()
    minY = Yopt.numpy()
    print(minX, minY, opt)
    
    return Xprogress, Yprogress, startX, startY, minX, minY



##learning...

### Key variables ###
# learning rate:
Lr = 0.1

#Epochs:
epochs = 100
#####################

###Lines and points setup...
##Adam...
optAdam = tf.keras.optimizers.Adam(learning_rate=Lr)

XprogressAdam, YprogressAdam, startXAdam, startYAdam, \
minXAdam, minYAdam = lineAndPoints(optAdam, epochs)

##RMSprop...
optRMSprop = tf.keras.optimizers.RMSprop(learning_rate=Lr)

XprogressRMSprop, YprogressRMSprop, startXRMSprop, startYRMSprop, \
minXRMSprop, minYRMSprop = lineAndPoints(optRMSprop, epochs)

##Adagrad...
optAdagrad = tf.keras.optimizers.Adagrad(learning_rate=Lr)

XprogressAdagrad, YprogressAdagrad, startXAdagrad, startYAdagrad, \
minXAdagrad, minYAdagrad = lineAndPoints(optAdagrad, epochs)

##Adadelta...
optAdadelta = tf.keras.optimizers.Adadelta(learning_rate=Lr)

XprogressAdadelta, YprogressAdadelta, startXAdadelta, startYAdadelta, \
minXAdadelta, minYAdadelta = lineAndPoints(optAdadelta, epochs)



##Plotting 3D...

## Image size
fig = mlab.figure(size=(2000,2000))

##axis size and scale
ax_ranges = [axisMin, axisMax, axisMin, axisMax, 0, 20000]
zScaler = 1/100000
xyScaler = 1/2
ax_scale = [xyScaler,xyScaler, zScaler]

surf = mlab.surf(X, Y, Z, colormap='winter', vmax = 20, vmin = -10)
surf.actor.actor.scale = ax_scale
surf.actor.property.opacity = 1

#Sizes...
pointRadius = 0.1
lineThickness = pointRadius/10

#Colors...
colorAdam = (1,0,1)
colorRMSprop = (0,1,0)
colorAdagrad = (0,0.5,1)
colorAdadelta = (1,0.5,0)

startPoint = mlab.points3d(startXAdam*xyScaler, startYAdam*xyScaler, f(startXAdam,
                startYAdam)*zScaler, color=(1, 1, 0), scale_factor = pointRadius)

MinPoint   = mlab.points3d(3*xyScaler,0.5*xyScaler,f(3,0.5)*zScaler,
                color=(1, 1, 0), scale_factor = pointRadius)


##Adam...
lineAdam = mlab.plot3d(XprogressAdam, YprogressAdam, f(XprogressAdam,
                YprogressAdam), color=colorAdam, tube_radius = lineThickness)

endPointAdam = mlab.points3d(minXAdam*xyScaler, minYAdam*xyScaler,
                f(minXAdam, minYAdam)*zScaler, color=colorAdam, scale_factor = pointRadius)

lineAdam.actor.actor.scale = ax_scale

##RMSprop...
lineRMSprop = mlab.plot3d(XprogressRMSprop, YprogressRMSprop, f(XprogressRMSprop,
                YprogressRMSprop), color=colorRMSprop,tube_radius = lineThickness)

endPointRMSprop = mlab.points3d(minXRMSprop*xyScaler, minYRMSprop*xyScaler,
                f(minXRMSprop, minYRMSprop)*zScaler, color=colorRMSprop, scale_factor = pointRadius)

lineRMSprop.actor.actor.scale = ax_scale

##Adagrad...
lineAdagrad = mlab.plot3d(XprogressAdagrad, YprogressAdagrad, f(XprogressAdagrad,
                YprogressAdagrad), color=colorAdagrad, tube_radius = lineThickness)

endPointAdagrad = mlab.points3d(minXAdagrad*xyScaler, minYAdagrad*xyScaler, f(minXAdagrad,
                    minYAdagrad)*zScaler, color=colorAdagrad, scale_factor = pointRadius)

lineAdagrad.actor.actor.scale = ax_scale

##Adadelta...
lineAdadelta = mlab.plot3d(XprogressAdadelta, YprogressAdadelta, f(XprogressAdadelta,
                    YprogressAdadelta), color=colorAdadelta, tube_radius = lineThickness)

endPointAdadelta = mlab.points3d(minXAdadelta*xyScaler, minYAdadelta*xyScaler, f(minXAdadelta,
                    minYAdadelta)*zScaler, color=colorAdadelta, scale_factor = pointRadius)

lineAdadelta.actor.actor.scale = ax_scale


mlab.view(azimuth=190, elevation=None, distance=10, focalpoint=None,roll=None, reset_roll=True)

##axis bounds around the surf and has color...
mlab.outline(surf, color=(.7, .7, .7))

##Displays...
name = "3DAll-Epochs="+str(epochs)+"Lr="+str(Lr)+".png"
mlab.savefig(filename=name)



##Plotting 2D...

#2Ding...
def g(xList, yList):
    xt = [i ** 2 for i in xList]
    yt = [j ** 2 for j in yList]
    sum_list = [a + b for a, b in zip(xt, yt)]
    return [k ** 0.5 for k in sum_list]

epoch = np.linspace(0, epochs, len(f(XprogressAdadelta, YprogressAdadelta)))

#Colors...
colorAdam = (1,0,1)
colorRMSprop = (0,1,0)
colorAdagrad = (0,0.5,1)
colorAdadelta = (1,0.5,0)

#Importing pyplot to plot graphs
import matplotlib.pyplot as plt

#Plots...
plt.plot(epoch, f(XprogressAdam, YprogressAdam), color=colorAdam, label='Adam')

plt.plot(epoch, f(XprogressRMSprop, YprogressRMSprop), color=colorRMSprop, label='RMSprop')

plt.plot(epoch, f(XprogressAdagrad, YprogressAdagrad), color=colorAdagrad, label='Adagrad')

plt.plot(epoch, f(XprogressAdadelta, YprogressAdadelta), color=colorAdadelta, label='Adadelta')

#Labels...
Title = 'Optimiser comparison for ' + str(epochs) + ' epochs and a learning rate of ' + str(Lr)
plt.title(Title)
plt.ylabel('Z (rescaled)')
plt.xlabel('Epoch number')
plt.legend(loc="right")

#background color...
fig = plt.gcf()

##Saving image...
name = "2DAll-Epochs="+str(epochs)+"Lr="+str(Lr)+".png"
fig.set_size_inches(18.5/2, 10.5/2)
fig.savefig(name, dpi=250)

#Showing result...
plt.show()