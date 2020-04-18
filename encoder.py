import numpy as np
import cv2
import base64
import sys


def getLongestMatch(symbolsVector, windowSize, lookAheadPtr, leftWindowIdx):
    searchPtr = lookAheadPtr - 1
    bestMatchLength = 0
    bestMatchOffset = 0
    bestMatchSymbol = symbolsVector[lookAheadPtr]
    streamLength = len(symbolsVector)
    while(searchPtr > leftWindowIdx):
        matchLength = 0
        originalSearchPtr = searchPtr
        originalLookAheadPtr = lookAheadPtr
        # while searchPtr didn't pass right window and symbols match
        while((searchPtr < leftWindowIdx + windowSize) and (symbolsVector[searchPtr] == symbolsVector[lookAheadPtr])):
            if(lookAheadPtr == streamLength-1):
                break
            matchLength += 1
            searchPtr += 1
            lookAheadPtr += 1

        if(matchLength > bestMatchLength):
            bestMatchLength = matchLength
            bestMatchOffset = originalLookAheadPtr - originalSearchPtr
            bestMatchSymbol = symbolsVector[lookAheadPtr]

        searchPtr = originalSearchPtr - 1
        lookAheadPtr = originalLookAheadPtr
    if(bestMatchLength == 0):
        return (0, bestMatchSymbol)
    else:
        return (1, bestMatchOffset, bestMatchLength, bestMatchSymbol)


def encodeVector(symbolsVector, windowSize, lookAheadSize):
    tuplesVector = []
    leftWindowIdx = 0
    lookAheadPtr = windowSize - lookAheadSize
    streamLength = len(symbolsVector)
    while(lookAheadPtr < streamLength):
        longestMatchTuple = getLongestMatch(
            symbolsVector, windowSize, lookAheadPtr, leftWindowIdx)
        tuplesVector.append(longestMatchTuple)
        if(longestMatchTuple[0] == 0):
            leftWindowIdx += 1
            lookAheadPtr += 1
        else:
            leftWindowIdx += longestMatchTuple[2] + 1
            lookAheadPtr += longestMatchTuple[2] + 1
    return tuplesVector


def generateTuplesOutputs(tuplesVector):
    identifiers = []
    symbols = []
    offsetAndLength = []

    for currentTuple in tuplesVector:
        identifiers.append(currentTuple[0])
        if(currentTuple[0] == 0):
            symbols.append(currentTuple[1])
        else:
            offsetAndLength.append((currentTuple[1], currentTuple[2]))
            symbols.append(currentTuple[3])

    symbolsNPArray = np.array(symbols, dtype=np.uint8)
    tuplesDatatype = np.dtype('uint8,uint8')
    offsetAndLengthNPArray = np.array(offsetAndLength, dtype=tuplesDatatype)
    identifiersNPArray = np.array(identifiers, dtype=np.uint8)

    symbolsNPArray.tofile('./data/symbols.dat')
    offsetAndLengthNPArray.tofile('./data/offsetAndLength.dat')
    identifiersNPArray.tofile('./data/identifiers.dat')


# fileName = input('Enter the file name: ')
# windowSize = int(input('Enter the window size: '))
# lookAheadBufferSize = int(input('Enter the look-ahead buffer size: '))

fileName = 'test.jpg'
windowSize = 64
lookAheadBufferSize = 16

imgMatrix = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

dimensions = np.array(
    [imgMatrix.shape[0], imgMatrix.shape[1]], dtype=int)  # height x width

imgVector = imgMatrix.flatten()
tuplesVector = encodeVector(imgVector, windowSize, lookAheadBufferSize)

# windowSize = 8
# lookAheadBufferSize = 4
# symbolsVector = 'aaaabababaaa'
# tuplesVector = encodeVector(symbolsVector, windowSize, lookAheadBufferSize)

bufferSearchPrefix = imgVector[:(windowSize - lookAheadBufferSize)]
np.array(bufferSearchPrefix, dtype=np.uint8).tofile(
    './data/bufferSearchPrefix.dat')
dimensions.tofile('./data/dimensions.dat')
generateTuplesOutputs(tuplesVector)
