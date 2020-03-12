import torch


class AttentionBase(torch.nn.Module):
    def __init__(self, attentionName, attentionScope, featuresNumber, cudaFlag):
        self.attentionName, self.attentionScope, self.cudaFlag = attentionName, attentionScope, cudaFlag
        super(AttentionBase, self).__init__()
        if attentionName == 'StandardAttention':
            self.attentionWeightLayer = torch.nn.Linear(in_features=featuresNumber, out_features=1)
        if attentionName == 'LocalAttention':
            self.attentionWeightLayer = torch.nn.Linear(in_features=featuresNumber * attentionScope, out_features=1)
        if self.attentionName == 'ComponentAttention':
            self.attentionWeightLayer = torch.nn.Conv2d(
                in_channels=1, out_channels=featuresNumber, kernel_size=[attentionScope, featuresNumber], stride=[1, 1],
                padding_mode='VALID')
        if self.attentionName == 'MonotonicAttention':
            self.sumKernel = torch.ones(size=[1, 1, self.attentionScope])
            self.attentionWeightNumeratorLayer = torch.nn.Linear(in_features=featuresNumber, out_features=1)
            self.attentionWeightDenominatorLayer = torch.nn.Linear(in_features=featuresNumber, out_features=1)

    def ApplyAttention(self, dataInput, attentionName, inputSeqLen, hiddenNoduleNumbers):
        if attentionName == 'StandardAttention':
            return self.StandardAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'LocalAttention':
            return self.LocalAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'ComponentAttention':
            return self.ComponentAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'MonotonicAttention':
            return self.MonotonicAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)

    def AttentionMask(self, seqInput):
        returnTensor = torch.cat(
            [torch.cat([torch.ones(v), torch.ones(torch.max(seqInput) - v) * -1]).view([1, -1]) for v in seqInput])
        if self.cudaFlag:
            return returnTensor.cuda() * 9999
        else:
            return returnTensor * 9999

    def StandardAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        attentionOriginWeight = self.attentionWeightLayer(input=dataInput.reshape([-1, hiddenNoduleNumbers]))
        attentionOriginWeight = attentionOriginWeight.view([dataInput.size()[0], dataInput.size()[1]])

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def LocalAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        dataInputPaddingPart = torch.zeros(size=[dataInput.size()[0], self.attentionScope, dataInput.size()[2]])
        if self.cudaFlag:
            dataInputPaddingPart = dataInputPaddingPart.cuda()
        dataInputSupplement = torch.cat([dataInput, dataInputPaddingPart], dim=1)
        dataInputExtension = torch.cat(
            [dataInputSupplement[:, v:dataInput.size()[1] + v, :] for v in range(self.attentionScope)], dim=-1)
        attentionOriginWeight = self.attentionWeightLayer(
            input=dataInputExtension.view([-1, hiddenNoduleNumbers * self.attentionScope])).view(
            [dataInput.size()[0], -1])
        #########################################################

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def ComponentAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        dataInputPaddingPart = torch.zeros(size=[dataInput.size()[0], self.attentionScope - 1, dataInput.size()[2]])
        if self.cudaFlag:
            dataInputPaddingPart = dataInputPaddingPart.cuda()
        dataInputSupplement = torch.cat([dataInput, dataInputPaddingPart], dim=1)
        dataInputSupplement = dataInputSupplement.unsqueeze(1)
        attentionOriginWeight = self.attentionWeightLayer(input=dataInputSupplement).squeeze()
        if len(attentionOriginWeight.size()) == 2: attentionOriginWeight = attentionOriginWeight.unsqueeze(0)
        attentionOriginWeight = attentionOriginWeight.permute(0, 2, 1)

        if seqInput is not None:
            attentionMask = self.AttentionMask(seqInput=seqInput).unsqueeze(-1).repeat([1, 1, hiddenNoduleNumbers])
            attentionMaskWeight = attentionOriginWeight.min(attentionMask)
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=1)
        attentionSeparateResult = torch.mul(dataInput, attentionWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def MonotonicAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        attentionNumeratorWeight = self.attentionWeightNumeratorLayer(input=dataInput).tanh()
        attentionDenominatorRawWeight = self.attentionWeightDenominatorLayer(input=dataInput).exp()
        padDenominatorZero = torch.zeros(size=[attentionDenominatorRawWeight.size()[0], self.attentionScope - 1,
                                               attentionDenominatorRawWeight.size()[2]])
        if self.cudaFlag:
            padDenominatorZero = padDenominatorZero.cuda()
            self.sumKernel = self.sumKernel.float().cuda()

        attentionDenominatorSupplementWeight = torch.cat([padDenominatorZero, attentionDenominatorRawWeight], dim=1)

        attentionDenominatorWeight = torch.conv1d(input=attentionDenominatorSupplementWeight.permute(0, 2, 1),
                                                  weight=self.sumKernel, stride=1)
        attentionOriginWeight = torch.div(attentionNumeratorWeight.squeeze(), attentionDenominatorWeight.squeeze())

        #########################################################

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight
        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight
