package m3ql

import (
	"math"

	"code.uber.internal/infra/statsdex/x/errors"
)

// how to make this work with multidimensional data
// what if I want to attach other methods to this?

var (
	ErrTraining = errors.NewInvalidParamsError(errors.New("Must have at least three data points to estimate"))
	ErrApplication = errors.NewInvalidParamsError(errors.New("Must have at least one application datapoint"))
	ErrBandwidth = errors.NewInvalidParamsError(errors.New("Bandwidth must be nonzero positive"))
)


type Kernel interface {
	estimateDensity(trainingData []float64, applicationData []float64, bandwidth float64) ([]float64, error)
}

type bandwidthFunc func() float64

type kernelFunc func(float64) float64

func (kf kernelFunc) estimateDensity(trainingData []float64, applicationData []float64, bandwidth bandwidthFunc) ([]float64, error) {

	N := len(trainingData)
	M := len(applicationData)

	if N < 2 {
		return nil, ErrTraining
	}

	if M < 1 {
		return nil, ErrApplication
	}

	bandwidthValue := bandwidth()

	if bandwidthValue <= 0 {
		return nil, ErrBandwidth
	}

	result := make([]float64, M)

	for i, adat := range applicationData{
		for _, tdat := range trainingData {
			u := (adat - tdat)/bandwidthValue
			nh := float64(N)*bandwidthValue
			result[i] = result[i] + (1/nh)*kf(u)
		}
	}

	return result, nil
}

func scalar(scalar float64) bandwidthFunc {
	return func() float64 {
		return scalar
	} 
}

func scott(n float64) bandwidthFunc {
	return func() float64 {
		return math.Pow(n,(-1/(1+4)))
	}
}

func gaussian(value float64) float64 {
	return math.Pow(2*math.Pi, -.5)*math.Exp(-.5*(value*value))
} 

func parzen(value float64) float64 {
	
	if math.Abs(value) < .5 {
		return 1
	} 
	return 0
}