from MP5_code import CannyEdgeDetection

joy = CannyEdgeDetection('joy1.bmp', 'joy1_edges.png')
lena = CannyEdgeDetection('lena.bmp', 'lena_edges.png')
pointer = CannyEdgeDetection('pointer1.bmp', 'pointer1_edges.png')
test = CannyEdgeDetection('test1.bmp', 'test1_edges.png')

lena.plot_results()
joy.plot_results()
pointer.plot_results()
test.plot_results()

joy.save_numpy_img(joy.edges, 'joy1_edges.bmp')

joy_7kernel = CannyEdgeDetection('joy1.bmp', 'joy1_7k_edges.png', kernel_size=7)
joy_7kernel.save_numpy_img(joy_7kernel.edges, 'joy1_7k_edges.bmp')

joy_15sig = CannyEdgeDetection('joy1.bmp', 'joy1_15s_edges.png', sigma=0.1)
joy_15sig.save_numpy_img(joy_15sig.edges, 'joy1_15s_edges.bmp')

joy_inc = CannyEdgeDetection('joy1.bmp', 'joy1_inc_edges.png', high_ratio=0.5, low_ratio=0.15)
joy_inc.save_numpy_img(joy_inc.edges, 'joy1_inc_edges.bmp')

joy_dec = CannyEdgeDetection('joy1.bmp', 'joy1_dec_edges.png', high_ratio=0.15, low_ratio=0.5)
joy_dec.save_numpy_img(joy_dec.edges, 'joy1_dec_edges.bmp')


