package m3ql

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKernelDensityEvaluation(t *testing.T) {

	var (
		tests = []struct {
			training []float64
			application []float64
			bandwidth bandwidthFunc
			kernel kernelFunc
			output []float64
		}{
			{
				[]float64{4,5,5,6,12,14,15,15,16,17},
				[]float64{3,15},
				scalar(4),
				kernelFunc(parzen),
				[]float64{.025, .1},
			},
			{
				[]float64{4,5,5,6,12,14,15,15,16,17},
				[]float64{4,17},
				scalar(4),
				kernelFunc(gaussian),
				[]float64{0.04051277823126519, 0.04983784689148686},
			},
			{
				[]float64{4,5,5,6,12,14,15,15,16,17},
				[]float64{4,17},
				scott(10),
				kernelFunc(gaussian),
				[]float64{0.09368746959529126, 0.0753328273078405},
			},
		}
	)

	for _, test := range tests {

		output, err := test.kernel.estimateDensity(test.training, test.application, test.bandwidth)
		require.NoError(t, err)
		require.Equal(t, len(test.application), len(output))

		for step := 0; step < len(output); step++ {
			v := output[step]
			assert.Equal(t, test.output[step], v, "invalid value for %d", step)
		}
	}
}


func TestKernelDensityEvaluationErrors(t *testing.T) {

	var (
		tests = []struct {
			training []float64
			application []float64
			bandwidth float64
			kernel kernelFunc
			output []float64
		}{
			{
				[]float64{4,5,5,6,12,14,15,15,16,17},
				[]float64{3,15},
				0,
				kernelFunc(parzen),
				[]float64{.025, .1},
			},
			{
				[]float64{4,5,5,6,12,14,15,15,16,17},
				[]float64{3,15},
				-2,
				kernelFunc(parzen),
				[]float64{.025, .1},
			},
			{
				[]float64{},
				[]float64{4,17},
				4,
				kernelFunc(gaussian),
				[]float64{0.04051277823126519, 0.04983784689148686},
			},
			{
				[]float64{4,5,5,6,12,14,15,15,16,17},
				[]float64{},
				4,
				kernelFunc(gaussian),
				[]float64{0.04051277823126519, 0.04983784689148686},
			},
		}
	)

	for _, test := range tests {

		_, err := test.kernel.estimateDensity(test.training, test.application, scalar(test.bandwidth))
		require.Error(t, err)
	}
}
