import gpuMock from 'gpu-mock.js'
import {
  predict,
  compareFilters,
  compareInputs,
  compareBiases,
  getCompareFilterInputIndexStart,
  getCompareFilterInputIndexStop
} from '../../src/layer/convolution'

describe('Convolution Layer', () => {
  describe('.getCompareFilterInputIndexStart', () => {
    test.only('can slide along matrix correctly', () => {
      const stride = 1
      const padding = 10
      expect(getCompareFilterInputIndexStart(0, stride, padding)).toEqual(10)
      expect(getCompareFilterInputIndexStart(1, stride, padding)).toEqual(9)
      expect(getCompareFilterInputIndexStart(2, stride, padding)).toEqual(8)
      expect(getCompareFilterInputIndexStart(3, stride, padding)).toEqual(7)
      expect(getCompareFilterInputIndexStart(4, stride, padding)).toEqual(6)
      expect(getCompareFilterInputIndexStart(5, stride, padding)).toEqual(5)
      expect(getCompareFilterInputIndexStart(6, stride, padding)).toEqual(4)
      expect(getCompareFilterInputIndexStart(7, stride, padding)).toEqual(3)
      expect(getCompareFilterInputIndexStart(8, stride, padding)).toEqual(2)
      expect(getCompareFilterInputIndexStart(9, stride, padding)).toEqual(1)
    })
    test.only('can keep exactly above 0', () => {
      const stride = 1
      const padding = 5
      expect(getCompareFilterInputIndexStart(0, stride, padding)).toEqual(5)
      expect(getCompareFilterInputIndexStart(1, stride, padding)).toEqual(4)
      expect(getCompareFilterInputIndexStart(2, stride, padding)).toEqual(3)
      expect(getCompareFilterInputIndexStart(3, stride, padding)).toEqual(2)
      expect(getCompareFilterInputIndexStart(4, stride, padding)).toEqual(1)
      expect(getCompareFilterInputIndexStart(5, stride, padding)).toEqual(0)
      expect(getCompareFilterInputIndexStart(6, stride, padding)).toEqual(0)
      expect(getCompareFilterInputIndexStart(7, stride, padding)).toEqual(0)
      expect(getCompareFilterInputIndexStart(8, stride, padding)).toEqual(0)
      expect(getCompareFilterInputIndexStart(9, stride, padding)).toEqual(0)
    })
  })
  describe('.getCompareFilterInputIndexStop', () => {
    test.only('can slide along matrix correctly', () => {
      const stride = 1
      const padding = 2
      const deltaSize = 5
      const inputSize = 10
      expect(getCompareFilterInputIndexStop(0, stride, padding, deltaSize, inputSize)).toEqual(7)
      expect(getCompareFilterInputIndexStop(1, stride, padding, deltaSize, inputSize)).toEqual(6)
      expect(getCompareFilterInputIndexStop(2, stride, padding, deltaSize, inputSize)).toEqual(5)
      expect(getCompareFilterInputIndexStop(3, stride, padding, deltaSize, inputSize)).toEqual(4)
    })
    test.only('can stay exactly below inputWidth', () => {
      const stride = 1
      const padding = 2
      const deltaSize = 15
      const inputSize = 10
      expect(getCompareFilterInputIndexStop(0, stride, padding, deltaSize, inputSize)).toEqual(10)
      expect(getCompareFilterInputIndexStop(1, stride, padding, deltaSize, inputSize)).toEqual(10)
      expect(getCompareFilterInputIndexStop(2, stride, padding, deltaSize, inputSize)).toEqual(10)
      expect(getCompareFilterInputIndexStop(3, stride, padding, deltaSize, inputSize)).toEqual(10)
      expect(getCompareFilterInputIndexStop(4, stride, padding, deltaSize, inputSize)).toEqual(10)
      expect(getCompareFilterInputIndexStop(5, stride, padding, deltaSize, inputSize)).toEqual(10)
      expect(getCompareFilterInputIndexStop(6, stride, padding, deltaSize, inputSize)).toEqual(10)
      expect(getCompareFilterInputIndexStop(7, stride, padding, deltaSize, inputSize)).toEqual(10)
      expect(getCompareFilterInputIndexStop(8, stride, padding, deltaSize, inputSize)).toEqual(9)
      expect(getCompareFilterInputIndexStop(9, stride, padding, deltaSize, inputSize)).toEqual(8)
      expect(getCompareFilterInputIndexStop(10, stride, padding, deltaSize, inputSize)).toEqual(7)
      expect(getCompareFilterInputIndexStop(11, stride, padding, deltaSize, inputSize)).toEqual(6)
      expect(getCompareFilterInputIndexStop(12, stride, padding, deltaSize, inputSize)).toEqual(5)
      expect(getCompareFilterInputIndexStop(13, stride, padding, deltaSize, inputSize)).toEqual(4)
      expect(getCompareFilterInputIndexStop(14, stride, padding, deltaSize, inputSize)).toEqual(3)
      expect(getCompareFilterInputIndexStop(15, stride, padding, deltaSize, inputSize)).toEqual(2)
      expect(getCompareFilterInputIndexStop(16, stride, padding, deltaSize, inputSize)).toEqual(1)
      expect(getCompareFilterInputIndexStop(17, stride, padding, deltaSize, inputSize)).toEqual(0)
      expect(getCompareFilterInputIndexStop(18, stride, padding, deltaSize, inputSize)).toEqual(-1)
    })
  })
  describe('.predict (forward propagation)', () => {
    test('can convolution a simple matrix', () => {
      const inputs = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
      const filters = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
      const biases = [1, 2, 3]

      const results = gpuMock(predict, {
        output: [3, 3],
        constants: {
          strideX: 1,
          strideY: 1,
          paddingY: 0,
          paddingX: 0,
          filterHeight: 3,
          filterWidth: 3,
          filterCount: 1,
          inputWidth: 3,
          inputHeight: 3,
          inputDepth: 1,
        },
      })(filters, inputs, biases)

      expect(results).toEqual([[286, 187, 91], [155, 95, 43], [51, 27, 10]])
    })
  })

  describe('.compareFilters (back propagation)', () => {
    test('can convolution a simple matrix', () => {
      const filters = [[[1, 2], [3, 4]]]
      const inputs = [[[1, 2], [3, 4]]]
      const deltas = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
      const results = gpuMock(compareFilters, {
        output: [2, 2],
        constants: {
          strideX: 1,
          strideY: 1,
          paddingY: 0,
          paddingX: 0,
          filterHeight: 2,
          filterWidth: 2,
          filterCount: 1,
          inputWidth: 2,
          inputHeight: 2,
          inputDepth: 1,
          deltasDepth: 1,
          deltasHeight: 3,
          deltasWidth: 3
        },
      })(filters, inputs, deltas)

      expect(results).toEqual([[38, 49], [70, 81]])
    })
    test('can convolution a simple matrix', () => {
      const filters = [[[1, 2], [3, 4]]]
      const inputs = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]
      const deltas = [[
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30, 31, 32],
        [33, 34, 35, 36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45, 46, 47, 48],
        [49, 50, 51, 52, 53, 54, 55, 56],
        [57, 58, 59, 60, 61, 62, 63, 64]
      ]]
      const results = gpuMock(compareFilters, {
        output: [2, 2],
        constants: {
          strideX: 1,
          strideY: 1,
          paddingY: 1,
          paddingX: 1,
          filterHeight: 2,
          filterWidth: 2,
          filterCount: 1,
          inputWidth: 3,
          inputHeight: 3,
          inputDepth: 1,
          deltasDepth: 1,
          deltasHeight: 8,
          deltasWidth: 8
        },
      })(filters, inputs, deltas)

      expect(results).toEqual([[358, 314], [270, 226]])
    })
  })

  describe('.compareInputs (back propagation)', () => {
    test('can convolution a simple matrix', () => {
      const inputs = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
      const deltas = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
      const results = gpuMock(compareInputs, {
        output: [3, 3],
        constants: {
          strideX: 1,
          strideY: 1,
          paddingY: 0,
          paddingX: 0,
          filterHeight: 3,
          filterWidth: 3,
          filterCount: 1,
          inputWidth: 3,
          inputHeight: 3,
          inputDepth: 1,
        },
      })(inputs, deltas)

      expect(results).toEqual([[1, 4, 10], [8, 26, 56], [30, 84, 165]])
    })
  })

  describe('.compareBiases (back propagation)', () => {
    const deltas = [
      [[0, 16], [8, 24]],
      [[1, 17], [9, 25]],
      [[2, 18], [10, 26]],
      [[3, 19], [11, 27]],
      [[4, 20], [12, 28]],
      [[5, 21], [13, 29]],
      [[6, 22], [14, 30]],
      [[7, 23], [15, 31]],
    ]
    test('accumulates values from deltas correctly from 0', () => {
      const biasDeltas = [0, 0, 0, 0, 0, 0, 0, 0]
      const kernel = gpuMock(compareBiases, {
        output: [1, 1, 8],
        constants: {
          x: 2,
          y: 2,
        },
      })
      const result = kernel(biasDeltas, deltas)
      const expectedBiasDeltas = [
        [[48]],
        [[52]],
        [[56]],
        [[60]],
        [[64]],
        [[68]],
        [[72]],
        [[76]],
      ]

      expect(result).toEqual(expectedBiasDeltas)
    })
    test('accumulates values from deltas correctly from greater than 0', () => {
      const biasDeltas = [0, 1, 2, 3, 4, 5, 6, 7]
      const kernel = gpuMock(compareBiases, {
        output: [1, 1, 8],
        constants: {
          x: 2,
          y: 2,
        },
      })
      const result = kernel(biasDeltas, deltas)
      const expectedBiasDeltas = [
        [[48]],
        [[53]],
        [[58]],
        [[63]],
        [[68]],
        [[73]],
        [[78]],
        [[83]],
      ]

      expect(result).toEqual(expectedBiasDeltas)
    })
  })
})
