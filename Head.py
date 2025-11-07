import keras
from keras import ops

@keras.saving.register_keras_serializable(package="SegFormer")
class MLP(keras.layers.Layer):
    def __init__(self, decode_dim):         # (B, H, W, C_in) 
        super().__init__()
        # creats an fully connected (dense) layer
        self.proj = keras.layers.Dense(decode_dim)

    def call(self, x):
        x = self.proj(x)
        return x    # (B, H, W, decode_dim)


#2D Convolution layer with a 1x1 kernel
@keras.saving.register_keras_serializable(package="SegFormer")
class ConvModule(keras.layers.Layer):
    def __init__(self, decode_dim):
        super().__init__()
        self.conv = keras.layers.Conv2D(
            filters=decode_dim, kernel_size=1, use_bias=False
        )
        self.bn = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.activate = keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x
    


# DECODER
@keras.saving.register_keras_serializable(package="SegFormer")
class SegFormerHead(keras.layers.Layer):
    def __init__(self, num_mlp_layers=4, decode_dim=768, num_classes=19):
        super().__init__()

        self.linear_layers = []
        for _ in range(num_mlp_layers):
            self.linear_layers.append(MLP(decode_dim))

        self.linear_fuse = ConvModule(decode_dim)
        self.dropout = keras.layers.Dropout(0.1)
        self.linear_pred = keras.layers.Conv2D(num_classes, kernel_size=1)

    def call(self, inputs):

        # input is all the 4 feature maps [c1, c2, c3, c4]
        # c1: (B, H/4, W/4, C1)
        # c2: (B, H/8, W/8, C2)
        # c3: (B, H/16, W/16, C3)
        # c4: (B, H/32, W/32, C4)

        H = ops.shape(inputs[0])[1]     #H/4
        W = ops.shape(inputs[0])[2]     #W/4
        outputs = []        #upsampled feature maps

        for x, mlps in zip(inputs, self.linear_layers):

            #1. Unify Channels
            x = mlps(x)     # (B, H_i, W_i, C_i) -> (B, H_i, W_i, decode_dim)

            # 2. Upsample   Resize this feature map to the target size (H, W)
            x = ops.image.resize(x, size=(H, W), interpolation="bilinear")      

            outputs.append(x)   # 3. Add to list.

        # after loop (B, H, W, decode_dim)

        # 4. Concatenate
        x = self.linear_fuse(ops.concatenate(outputs[::-1], axis=3))    # (B, H, W, 4 * decode_dim)

        # 5. Fuse : Pass the big concatenated tensor through the 1x1 Conv-BN-ReLU module.
        # x = self.linear_fuse(x)   #(B, H, W, decode_dim)

        # 6. Apply dropout
        x = self.dropout(x)

        # 7. Final Prediction
        x = self.linear_pred(x)     #(B, H, W, num_classes)

        return x
