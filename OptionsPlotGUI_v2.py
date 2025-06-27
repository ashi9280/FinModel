import tkinter as tk
import DataCollection
import numpy as np
import matplotlib.pyplot as plt

NO_COLUMNS = 10

root = tk.Tk()
root.title("Checkboxes in Multiple Columns")

def search_stock_ticker():
    option_list = []
    print("search_stock_ticker")

    print(f"checkboxes: {checkboxes}")
    print(f"selected_list: {selected_list}")

    # Clear out the checkboxes
    for v, cb, opt in checkboxes[:]:
        cb.destroy()

    ticker = ticker_text_box.get("1.0", "end-1c")
    expiration = expiration_text_box.get("1.0", "end-1c")

    options = DataCollection.get_options(ticker, expiration)
    calls = options.calls
    puts = options.puts

    for _, call in calls.iterrows():
        if not [ticker, call['strike'], call['lastPrice'], "call"] in selected_list:
            option_list.append([ticker, call['strike'], call['lastPrice'], "call"])

    for _, put in puts.iterrows():
        if not [ticker, put['strike'], put['lastPrice'], "put"] in selected_list:
            option_list.append([ticker, put['strike'], put['lastPrice'], "put"])

    for i in range(len(selected_list)):
        var = tk.BooleanVar()
        var.set(True)
        cb = tk.Checkbutton(checkbox_frame, text=selected_list[i], variable=var)
        row = i // NO_COLUMNS
        col = i % NO_COLUMNS
        cb.grid(row=row, column=col, padx=5, pady=5)
        checkboxes.append((var, cb, selected_list[i]))

    for i, opt in enumerate(option_list):
        var = tk.BooleanVar()
        cb = tk.Checkbutton(checkbox_frame, text=opt, variable=var)
        row = (i+len(selected_list)) // NO_COLUMNS
        col = (i+len(selected_list)) % NO_COLUMNS
        cb.grid(row=row, column=col, padx=5, pady=5)
        checkboxes.append((var, cb, opt))

    print(f"option_list: {option_list}")

    print(f"checkboxes: {checkboxes}")
    print(f"selected_list: {selected_list}")
    print()

def remove_unselected():
    print("remove_unselected")
    selected_list.clear()
    for v, cb, opt in checkboxes[:]:
        if v.get() and not opt in selected_list:
            selected_list.append(opt)
        cb.destroy()
    checkboxes.clear()
    for i,s in enumerate(selected_list[:]):
        # add the selected list to the checkboxes
        var = tk.BooleanVar()
        var.set(True)
        cb = tk.Checkbutton(checkbox_frame, text=s, variable=var)
        row = i // NO_COLUMNS
        col = i % NO_COLUMNS
        cb.grid(row=row, column=col, padx=5, pady=5)
        checkboxes.append((var, cb, s))


    print(f"checkboxes: {checkboxes}")
    print(f"selected_list: {selected_list}")
    print()

def plot_options():
    print("plot_options")

    print(f"selected_list: {selected_list}")

    min_price = min(selected_list, key=lambda x: x[1])[1]
    max_price = max(selected_list, key=lambda x: x[1])[1]

    price_range = np.linspace(min_price*0.9, max_price*1.1, 1000)

    profit_list = np.zeros(len(price_range))

    for opt in selected_list:
        if opt[3] == "call":
            profit_list += np.maximum(price_range - opt[1] - opt[2], -opt[2])
        elif opt[3] == "put":
            profit_list += np.maximum(opt[1] - price_range - opt[2], -opt[2])
        else:
            profit_list += price_range - opt[1]

    plt.plot(price_range, profit_list)
    plt.xlabel("Stock Price")
    plt.ylabel("Profit")
    plt.title("Profit vs Stock Price")
    plt.show()
    print()

# Store checkbox variables and widgets
checkboxes = []
selected_list = []

# Create the top label
tk.Label(root, text="Input Stock Ticker, Date, Expiration").pack(pady=(15, 5))

# Text boxes for stock inputs
ticker_text_box = tk.Text(root, height=1, width=10)
ticker_text_box.pack(pady=5)
expiration_text_box = tk.Text(root, height=1, width=10)
expiration_text_box.pack(pady=5)

# Button to search stock ticker
search_button = tk.Button(root, text="Search", command=search_stock_ticker)
search_button.pack(pady=10)

# Button to remove unselected options
remove_button = tk.Button(root, text="Remove Unselected", command=remove_unselected)
remove_button.pack(pady=10)

# Button to plot the options
plot_button = tk.Button(root, text="Plot", command=plot_options)
plot_button.pack(pady=10)

# Create a frame to hold the checkbox grid
checkbox_frame = tk.Frame(root)
checkbox_frame.pack(padx=20, pady=20)


root.mainloop()