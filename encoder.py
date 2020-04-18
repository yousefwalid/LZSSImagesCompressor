import numpy as np
import cv2
import time
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
    if(windowSize > 255):
        tuplesDatatype = np.dtype('uint16,uint16')
    else:
        tuplesDatatype = np.dtype('uint8,uint8')
    offsetAndLengthNPArray = np.array(offsetAndLength, dtype=tuplesDatatype)
    identifiersNPArray = np.array(identifiers, dtype=np.uint8)

    symbolsNPArray.tofile('./data/symbols.dat')
    offsetAndLengthNPArray.tofile('./data/offsetAndLength.dat')
    identifiersNPArray.tofile('./data/identifiers.dat')


fileName = input('Enter the file name: ')
windowSize = int(input('Enter the window size: '))
lookAheadBufferSize = int(input('Enter the look-ahead buffer size: '))

imgMatrix = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
imgVector = imgMatrix.flatten()

start = time.process_time()
tuplesVector = encodeVector(imgVector, windowSize, lookAheadBufferSize)
print(str(time.process_time() - start) + ' s')

dimensions = np.array(
    [imgMatrix.shape[0], imgMatrix.shape[1]], dtype=int)  # height x width
bufferSearchPrefix = imgVector[:(windowSize - lookAheadBufferSize)]

np.array(bufferSearchPrefix, dtype=np.uint8).tofile(
    './data/bufferSearchPrefix.dat')
dimensions.tofile('./data/dimensions.dat')
np.array([windowSize, lookAheadBufferSize], dtype=np.uint16
         ).tofile('./data/windowSizeLookAheadSize.dat')
generateTuplesOutputs(tuplesVector)
