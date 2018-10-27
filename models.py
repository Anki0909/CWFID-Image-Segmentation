def segnet(n_classes, input_height=161, input_width=216):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    
    model = Sequential()
    model.add(Layer(input_shape=(input_height, input_width, 3)))
    model.add(ZeroPadding2D(padding=((7,8),(0,0))))
    
    model.add(Conv2D(filter_size,(kernel,kernel),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
    
    model.add(Conv2D(128,(kernel,kernel),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
    
    model.add(Conv2D(256,(kernel,kernel),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
    
    model.add(Conv2D(512,(kernel,kernel),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(512,(kernel,kernel),padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D(size=(pool_size,pool_size)))
    model.add(Conv2D(256,(kernel,kernel),padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D(size=(pool_size,pool_size)))
    model.add(Conv2D(128,(kernel,kernel),padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D(size=(pool_size,pool_size)))
    model.add(Conv2D(filter_size,(kernel,kernel),padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(n_classes,(1,1), padding='same'))
    
    model.add(Cropping2D(cropping=((7,8),(0,0))))
    
    model.outputHeight = model.output_shape[-3]
    model.outputWidth = model.output_shape[-2]
    
    if n_classes == 1:
        model.add(Reshape((model.output_shape[-3], model.output_shape[-2]),
                          input_shape=(model.output_shape[-3], model.output_shape[-2], 3)))
        model.add(Activation('sigmoid'))
    else:
        model.add(Reshape((model.output_shape[-3], model.output_shape[-2], n_classes),
                          input_shape=(model.output_shape[-3], model.output_shape[-2], 3)))
        model.add(Activation('sigmoid'))
        
    #if not optimizer is None:
    #    model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
        
    return model

def unet_res(n_classes = 1, start_neurons = 16, DropoutRatio = 0.5, img_height=161,img_width=216):
    # 101 -> 50
    input_layer = Input((img_height, img_width, 3))
    zero_pad = ZeroPadding2D(padding=((7,8),(4,4)))(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(zero_pad)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(n_classes, (1,1), padding="same", activation=None)(uconv1)
    output_layer = Cropping2D(cropping=((7,8),(4,4)))(output_layer_noActi)
    
    if n_classes == 1:
        output_layer = (Reshape((img_height, img_width),
                          input_shape=(img_height, img_width, 3)))(output_layer)
    else:
        output_layer = (Reshape((img_height, img_width, n_classes),
                          input_shape=(img_height, img_width, 3)))(output_layer)
    
    output_layer =  Activation('sigmoid')(output_layer)
    
    model = Model(input_layer, output_layer)
    
    return model
