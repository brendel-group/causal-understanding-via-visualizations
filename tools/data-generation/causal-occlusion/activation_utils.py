from collections import OrderedDict


layer_names_list = [  #'conv2d0',
    #                     'conv2d1',
    #                     'conv2d2',
    "mixed3a",
    "mixed3b",
    "mixed4a",
    "mixed4b",
    "mixed4c",
    "mixed4d",
    "mixed4e",
    "mixed5a",
    "mixed5b",
]


kernel_size_list = ["1x1", "3x3", "5x5", "pool"]


# This is according to the InceptionV1 implementation in lucid 0.3.8. There is a mistake in channel mixed4a...
idx_of_kernel_sizes_in_each_layer_dict = OrderedDict()
# idx_of_kernel_sizes_in_each_layer_dict['conv2d0_7x7'] = 64 # inferred from Szegedy et al. 2014 and confirmed with models.InceptionV1.layers
# idx_of_kernel_sizes_in_each_layer_dict['conv2d1_1x1'] = 64 # unknown. not given in Szegedy et al 2014 and confirmed with models.InceptionV1.layers
# idx_of_kernel_sizes_in_each_layer_dict['conv2d2_3x3'] = 192 # inferred from Szegedy et al. 2014 and confirmed with models.InceptionV1.layers
idx_of_kernel_sizes_in_each_layer_dict["mixed3a_1x1"] = 64
idx_of_kernel_sizes_in_each_layer_dict["mixed3a_3x3"] = 192
idx_of_kernel_sizes_in_each_layer_dict["mixed3a_5x5"] = 224
idx_of_kernel_sizes_in_each_layer_dict["mixed3a_pool"] = 256
idx_of_kernel_sizes_in_each_layer_dict["mixed3b_1x1"] = 128
idx_of_kernel_sizes_in_each_layer_dict["mixed3b_3x3"] = 320
idx_of_kernel_sizes_in_each_layer_dict["mixed3b_5x5"] = 416
idx_of_kernel_sizes_in_each_layer_dict["mixed3b_pool"] = 480
idx_of_kernel_sizes_in_each_layer_dict["mixed4a_1x1"] = 192
idx_of_kernel_sizes_in_each_layer_dict["mixed4a_3x3"] = 396
idx_of_kernel_sizes_in_each_layer_dict["mixed4a_5x5"] = 444
idx_of_kernel_sizes_in_each_layer_dict[
    "mixed4a_pool"
] = 508  # this one should be 512 according to Szegedy et al. 2014
idx_of_kernel_sizes_in_each_layer_dict["mixed4b_1x1"] = 160
idx_of_kernel_sizes_in_each_layer_dict["mixed4b_3x3"] = 384
idx_of_kernel_sizes_in_each_layer_dict["mixed4b_5x5"] = 448
idx_of_kernel_sizes_in_each_layer_dict["mixed4b_pool"] = 512
idx_of_kernel_sizes_in_each_layer_dict["mixed4c_1x1"] = 128
idx_of_kernel_sizes_in_each_layer_dict["mixed4c_3x3"] = 384
idx_of_kernel_sizes_in_each_layer_dict["mixed4c_5x5"] = 448
idx_of_kernel_sizes_in_each_layer_dict["mixed4c_pool"] = 512
idx_of_kernel_sizes_in_each_layer_dict["mixed4d_1x1"] = 112
idx_of_kernel_sizes_in_each_layer_dict["mixed4d_3x3"] = 400
idx_of_kernel_sizes_in_each_layer_dict["mixed4d_5x5"] = 464
idx_of_kernel_sizes_in_each_layer_dict["mixed4d_pool"] = 528
idx_of_kernel_sizes_in_each_layer_dict["mixed4e_1x1"] = 256
idx_of_kernel_sizes_in_each_layer_dict["mixed4e_3x3"] = 576
idx_of_kernel_sizes_in_each_layer_dict["mixed4e_5x5"] = 704
idx_of_kernel_sizes_in_each_layer_dict["mixed4e_pool"] = 832
idx_of_kernel_sizes_in_each_layer_dict["mixed5a_1x1"] = 256
idx_of_kernel_sizes_in_each_layer_dict["mixed5a_3x3"] = 576
idx_of_kernel_sizes_in_each_layer_dict["mixed5a_5x5"] = 704
idx_of_kernel_sizes_in_each_layer_dict["mixed5a_pool"] = 832
idx_of_kernel_sizes_in_each_layer_dict["mixed5b_1x1"] = 384
idx_of_kernel_sizes_in_each_layer_dict["mixed5b_3x3"] = 768
idx_of_kernel_sizes_in_each_layer_dict["mixed5b_5x5"] = 896
idx_of_kernel_sizes_in_each_layer_dict["mixed5b_pool"] = 1024
