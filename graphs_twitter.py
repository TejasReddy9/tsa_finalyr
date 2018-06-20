import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib import style
import time
style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
	pullData = open("twitter-out.txt","r").read()
	lines = pullData.split('\n')
	
	xar = []
	yar = []
	
	x=0
	y=0
	for line in lines[-200:]:
		x+=1
		if "pos" in line:
			y+=1
		elif "neg" in line:
			y-=1 #people have less words for positive, so tweak it if needed to 0.3 or something
		xar.append(x)
		yar.append(y)
	
	ax1.clear()
	ax1.plot(xar,yar)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()