import numpy as np
import cv2


def decodeTuples(bufferSearchPrefix, tuplesVector, windowSize, lookAheadSize):
    originalStream = bufferSearchPrefix.tolist()
    lookAheadPtr = windowSize - lookAheadSize
    for currentTuple in tuplesVector:
        if(currentTuple[0] == 0):
            originalStream.append(currentTuple[1])
            lookAheadPtr += 1
        else:
            copyFromIdx = lookAheadPtr - currentTuple[1]
            copyLength = currentTuple[2]
            while(copyLength > 0):
                originalStream.append(originalStream[copyFromIdx])
                copyFromIdx += 1
                copyLength -= 1
            originalStream.append(currentTuple[3])
            lookAheadPtr += currentTuple[2] + 1
    return originalStream


def buildTuplesFromLists(symbols, offsetAndLength, identifiers):
    offsetAndLengthIdx = 0

    tuples = []

    identifiersLength = len(identifiers)
    for identifierIdx in range(identifiersLength):
        if(identifiers[identifierIdx] == 0):
            tuples.append((0, symbols[identifierIdx]))
        else:
            tuples.append((1, offsetAndLength[offsetAndLengthIdx][0],
                           offsetAndLength[offsetAndLengthIdx][1], symbols[identifierIdx]))
            offsetAndLengthIdx += 1

    return tuples


bufferSearchPrefix = np.fromfile(
    './data/bufferSearchPrefix.dat', dtype=np.uint8)

symbols = np.fromfile('./data/symbols.dat', dtype=np.uint8)
offsetAndLength = np.fromfile(
    './data/offsetAndLength.dat', dtype=np.dtype('uint8,uint8'))
identifiers = np.fromfile('./data/identifiers.dat', dtype=np.uint8)

tuples = buildTuplesFromLists(symbols, offsetAndLength, identifiers)
dimensions = np.fromfile('./data/dimensions.dat', dtype=int)  # height x width

imgVector = decodeTuples(bufferSearchPrefix, tuples, 64, 16)
imgMatrix = np.array(imgVector, dtype=np.uint8).reshape(
    dimensions[0], dimensions[1])

cv2.imwrite('output.jpg', imgMatrix)
