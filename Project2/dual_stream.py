import torch
import torch.nn as nn
import torch.nn.functional as F


# Optimized CNN Architecture for Regularization Experiments (~150k parameters)
class TwoStream(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0, use_batchnorm=False, avgLogits = True):
        super(TwoStream, self).__init__()

        # More channels with Global Average Pooling: 3->32->64->128
        self.conv1S = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 128x128 -> 128x128
        self.conv2S = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x64 -> 64x64
        self.conv3S = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv4S = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Batch normalization layers (optional)
        self.bn1S = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.bn2S = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.bn3S = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.bn4S = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()

        self.poolS = nn.MaxPool2d(2, 2)  # Halves dimensions each time


        # But we'll do 4 pools to get to reasonable size: 128->64->32->16->8
        self.flattenS = nn.Flatten()
        # Compact FC head for ~150k total parameters: 8*8*128 -> 64 -> 16 -> 2
        # TODO: WATCH OUT THIS IS HARDCODED BASED ON THE IMAGE SIZE
        self.fc1S = nn.Linear(128 * 8**2, 64)
        self.fc2S = nn.Linear(64, 16)
        self.fc3S = nn.Linear(16, num_classes)

        # Whether to simply average (mean) logits or use a small SVM head over concatenated logits
        self.avgLogits = avgLogits
        # Small SVM head: maps concatenated [logits_s, logits_f] -> final logits
        # Initialized regardless of avgLogits so it can be used later if toggled
        self.svm = nn.Linear(num_classes * 2, num_classes)

        # Configurable dropout
        self.dropout_rateS = dropout_rate
        self.dropoutS = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # --- Flow stream (mirror spatial architecture) ---
        # The flow input channel count can vary; default handled outside by passing appropriate flow tensors.
        # To keep this module flexible we create a second stream expecting a configurable number of input channels.
        # Default flow input channels (e.g. 2 * 9 = 18) should be passed when constructing the model via init_flow_stream.
        self.flow_enabled = False
        # placeholders for flow modules; initialized when init_flow_stream is called
        self.conv1F = None
        self.conv2F = None
        self.conv3F = None
        self.conv4F = None

        self.bn1F = None
        self.bn2F = None
        self.bn3F = None
        self.bn4F = None

        self.poolF = self.poolS
        self.flattenF = nn.Flatten()

        self.fc1F = None
        self.fc2F = None
        self.fc3F = None
        self.dropoutF = None

    def init_flow_stream(self, flow_in_channels, use_batchnorm=False):
        """Initialize the flow stream layers when flow input channel count is known.

        Call this after constructing the TwoStream model if you want to enable the flow branch.
        Example: model.init_flow_stream(flow_in_channels=18)
        """
        # Convolutions mirror the spatial stream but accept flow_in_channels on first conv
        self.conv1F = nn.Conv2d(flow_in_channels, 32, kernel_size=3, padding=1)
        self.conv2F = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3F = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4F = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.bn1F = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.bn2F = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.bn3F = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.bn4F = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()


        # Fully-connected head mirrors the spatial head
        self.fc1F = nn.Linear(128 * 8**2, 64)
        self.fc2F = nn.Linear(64, 16)
        self.fc3F = nn.Linear(16, self.fc3S.out_features)

        self.dropoutF = nn.Dropout(self.dropout_rateS) if self.dropout_rateS > 0 else nn.Identity()
        self.flow_enabled = True / 2.0

    def forward(self, xS, xF=None):
        # ----- Spatial stream -----
        s = self.poolS(F.relu(self.bn1S(self.conv1S(xS))))
        s = self.poolS(F.relu(self.bn2S(self.conv2S(s))))
        s = self.poolS(F.relu(self.bn3S(self.conv3S(s))))
        s = self.poolS(F.relu(self.bn4S(self.conv4S(s))))  # 128→64→32→16→8

        s = self.flattenS(s)  # (N, 128*8*8)
        s = self.dropoutS(F.relu(self.fc1S(s)))
        s = self.dropoutS(F.relu(self.fc2S(s)))
        logits_s = self.fc3S(s)  # (N, num_classes)

        # If flow branch disabled or no flow provided, return spatial logits
        if (xF is None) or (not self.flow_enabled):
            return logits_s

        # ----- Flow stream -----
        f = self.poolF(F.relu(self.bn1F(self.conv1F(xF))))
        f = self.poolF(F.relu(self.bn2F(self.conv2F(f))))
        f = self.poolF(F.relu(self.bn3F(self.conv3F(f))))
        f = self.poolF(F.relu(self.bn4F(self.conv4F(f))))

        f = self.flattenF(f)
        f = self.dropoutF(F.relu(self.fc1F(f)))
        f = self.dropoutF(F.relu(self.fc2F(f)))
        logits_f = self.fc3F(f)

        # Fusion options:
        # - if avgLogits True: apply softmax to each stream, then average the class probabilities
        # - if avgLogits False: feed concatenated logits into small SVM head
        probs_s = F.softmax(logits_s, dim=1)
        probs_f = F.softmax(logits_f, dim=1)
        if self.avgLogits:
            logits = (probs_s + probs_f) / 2.0
        else:
            cat = torch.cat([probs_s, probs_f], dim=1)
            logits = self.svm(cat)

        return logits