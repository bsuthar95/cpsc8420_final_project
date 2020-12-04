from keras.models import load_model
import tkinter as tk
import numpy as np

# need to change the MODEL_NAME to actual model
model = load_model('MODEL_NAME.h5')

def predicter(image):
	image = image.resize((28,28))
	image = image.convert('L')
	image = np.array(image)
	image = image.reshape(1,28,28,1)
	image = image/255.0

	result = model.predict([image])[0]
	return np.argmax(result),max(result)

class APP(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)

		self.x = self.y = 0
		self.canvas = tk.Canvas(self, width=500, height=500, bg='white', cursor="cross")
		self.label = tk.Label(self, text="Currently Processing....", font=("Helvetica", 35))
		# self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
		# self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

		self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
		self.label.grid(row=0, column=1, pady=2, padx=2)
		# self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
		# self.button_clear.grid(row=1, column=0, pady=2)

		# self.canvas.bind("<B1-Motion>", self.draw_lines)

	def clear_all(self):
		self.canvas.delete("all")
app=APP()
tk.mainloop()
