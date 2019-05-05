import matplotlib.pyplot as plt

# classes to help plotting training process

class DynamicPlot():
    #Suppose we know the x range

    def on_launch(self,min_x,max_x):
        plt.ion()
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'r-', label =  'Train')
        self.lines2, = self.ax.plot([],[], 'b-', label = 'Test')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(min_x, max_x)
        self.ax.legend()
        #Other stuff
        self.ax.grid()
        self.xdata = []
        self.ydata = []
        self.zdata = []
        ...

    def on_running(self, xdata, ydata, zdata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        self.lines2.set_xdata(xdata)
        self.lines2.set_ydata(zdata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def on_update(self, xdata, ydata, zdata):
        #Update data (with the new _and_ the old points)
        self.xdata.append(xdata)
        self.ydata.append(ydata)
        self.zdata.append(zdata)
        self.on_running(self.xdata, self.ydata, self.zdata)

    def on_finish(self):
        plt.show()


class DynamicPlotPlot():

    def on_launch(self):
        plt.ion()
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.subplots = []
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.legend()
        #Other stuff
        self.ax.grid()

    def add_subplot(self,subplot):
        
        self.subplots.append(subplot)

    def set_subplot(self, xdata, ydata, idx = 0):
        #Update data (with the new _and_ the old points)
        self.subplots[idx].set_xdata(xdata)
        self.subplots[idx].set_ydata(ydata)

    def on_update(self):
        #Update data (with the new _and_ the old points)
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()