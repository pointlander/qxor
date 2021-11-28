// Copyright 2021 The QXOR Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
)

func main() {
	rand.Seed(7)

	set := tc128.NewSet()
	set.Add("aw1", 4, 4)
	set.Add("inputs", 4, 4)
	set.Add("outputs", 4, 4)

	data := [...][4]int{
		{0, 0, 0, 0},
		{0, 1, 0, 1},
		{1, 0, 1, 1},
		{1, 1, 1, 0},
	}
	inputs := set.Weights[1]
	outputs := set.Weights[2]
	for i := range data {
		input := make([]complex128, 4)
		x := ((1 & data[i][0]) << 1) | (1 & data[i][1])
		input[x] = cmplx.Rect(1, 0)
		inputs.X = append(inputs.X, input...)
		output := make([]complex128, 4)
		y := ((1 & data[i][2]) << 1) | (1 & data[i][3])
		output[y] = cmplx.Rect(1, 0)
		outputs.X = append(outputs.X, output...)
	}

	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rand.Float64()+a, (b-a)*rand.Float64()+a)
	}

	for i := range set.Weights[:1] {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		} else {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		}
	}

	/*deltas := make([][]complex128, 0, 4)
	for _, p := range set.Weights[:4] {
		deltas = append(deltas, make([]complex128, len(p.X)))
	}*/

	l1 := tc128.Mul(set.Get("aw1"), set.Get("inputs"))
	cost := tc128.Avg(tc128.Quadratic(set.Get("outputs"), l1))

	alpha, eta, iterations := complex128(.3), complex128(.3), 128
	points := make(plotter.XYs, 0, iterations)
	_ = alpha
	i := 0
	for i < iterations {
		total := complex128(0)
		start := time.Now()
		set.Zero()

		total += tc128.Gradient(cost).X[0]
		norm := 0.0
		for _, p := range set.Weights[:1] {
			for _, d := range p.D {
				norm += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm = math.Sqrt(norm)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for k, p := range set.Weights[:1] {
			_ = k
			for l, d := range p.D {
				//deltas[k][l] = alpha*deltas[k][l] - eta*d*complex(scaling, 0)
				//p.X[l] += deltas[k][l]
				p.X[l] -= eta * d * complex(scaling, 0)
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: cmplx.Abs(total)})
		fmt.Println(i, cmplx.Abs(total), time.Now().Sub(start))
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	l1(func(a *tc128.V) bool {
		for i := 0; i < 4; i++ {
			max, maxj := 0.0, 0
			for j := 0; j < 4; j++ {
				v := cmplx.Abs(a.X[i*4+j])
				if v > max {
					max, maxj = v, j
				}
			}
			fmt.Printf("%02b\n", maxj)
		}
		return true
	})

	a := mat.NewDense(4, 4, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0, 1,
		0, 0, 1, 0,
	})
	var eig mat.Eigen
	ok := eig.Factorize(a, mat.EigenRight)
	if !ok {
		panic("Eigendecomposition failed")
	}
	fmt.Println("\neigenvalues")
	for i, value := range eig.Values(nil) {
		fmt.Println(i, cmplx.Abs(value))
	}
	vectors := mat.CDense{}
	eig.VectorsTo(&vectors)
	r, c := vectors.Dims()
	fmt.Println("\neigenvectors")
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Printf(" %f", vectors.At(i, j))
		}
		fmt.Printf("\n")
	}
}
