Download Link: https://assignmentchef.com/product/solved-col-774-machine-learning-assignment-1
<br>



<ul>

 <li></li>

</ul>

<ol>

 <li>(20 points) Linear Regression</li>

</ol>

In this problem, we will implement least squares linear regression to predict density of wine based on its acidity. Recall that the error metric for least squares is given by:

<table>

 <tbody>

  <tr>

   <td width="337">~m<u>1 </u>J(θ) = 2mi=1</td>

   <td width="314">(y<sup>(i)</sup> − hθ(x<sup>(i)</sup>))<sup>2</sup></td>

  </tr>

 </tbody>

</table>




where hθ(x) = θ<sup>T</sup>x and all the symbols are as discussed in the class. The files linearX.csv and linearY.csv contain the acidity of the wine (x(i)’s, x<sup>(i)</sup> E 7Z) and its density (y(i)’s, y<sup>(i)</sup> E 7Z), respectively, with one training example per row. We will implement least squares linear regression to learn the relationship between x<sup>(i)</sup>’s and y<sup>(i)</sup>’s.

(a) (8 points) Implement batch gradient descent method for optimizing J(θ). Choose an appropriate learning rate and the stopping criteria (as a function of the change in the value of J(θ)). You can initialize the parameters as θ = 0~ (the vector of all zeros). Do not forget to include the intercept term. Report your learning rate, stopping criteria and the final set of parameters obtained by your algorithm.




<ul>

 <li>(3 points) Plot the data on a two-dimensional graph and plot the hypothesis function learned by your algorithm in the previous part.</li>

 <li>(3 points) Draw a 3-dimensional mesh showing the error function (J(9)) on z-axis and the parameters in the x − y Display the error value using the current set of parameters at each iteration of the gradient descent. Include a time gap of 0.2 seconds in your display for each iteration so that the change in the function value can be observed by the human eye.</li>

 <li>(3 points) Repeat the part above for drawing the contours of the error function at each iteration of the gradient descent. Once again, chose a time gap of 0.2 seconds so that the change be perceived by the human eye.(Note here plot will be 2-D)</li>

 <li>(3 points) Repeat the part above (i.e. draw the contours at each learning iteration) for the step size values of η = {0.001, 0.025, 0.1}. What do you observe? Comment.</li>

</ul>

<ol start="2">

 <li>(20 points) Sampling and Stochastic Gradient Descent</li>

</ol>

In this problem, we will introduce the idea of sampling by adding Gaussian noise to the prediction of ahypothesis and generate synthetic training data. Consider a given hypothesis hθ (i.e. known 90, 91, 92) for

~x0 ~

a data point x = x1 . Note that x0 = 1 is the intercept term.

x2

y = hθ(x) = 90 + <sub>9</sub><sub>1</sub><sub>x</sub><sub>1</sub> + 92×2 Adding Gaussian noise, equation becomes

y = 90 + <sub>9</sub><sub>1</sub><sub>x</sub><sub>1</sub> + <sub>9</sub><sub>2</sub><sub>x</sub><sub>2</sub> + f

where f ∼ N(0, a<sup>2</sup>)

To gain deeper understanding behind Stochastic Gradient Descent (SGD), we will use the SGD algorithm to learn the original hypothesis from the data generated using sampling, for varying batch sizes. We will implement the version where we make a complete pass through the data in a round robin fashion (after initially shuffling the examples). If there are r examples in each batch, then there is a total of m r batches assuming m training examples. For the batch number b (1 ≤ b ≤ m r ), the set of examples is given as: {x<sup>(i1)</sup>, x<sup>(i2)</sup>,··· , <sup>x</sup><sup>(ir)}</sup> where ik = (b − 1)r + k. The Loss function computed over these r examples is given as:

<table>

 <tbody>

  <tr>

   <td width="323">~r<u>1 </u>Jb(9) = 2kk=1</td>

   <td width="328">(y(ik) − hθ(x<sup>(ik)</sup>))<sup>2</sup></td>

  </tr>

 </tbody>

</table>




<table>

 <tbody>

  <tr>

   <td width="429">(a) (4 points) Sample 1 million data points taking values of 9 =</td>

   <td width="25">[90 ]9192</td>

   <td width="18">=</td>

   <td width="19"><sub>[3</sub> ]12</td>

   <td width="42">, x1 ∼</td>

   <td width="117">N(3, 4) and x2 ∼</td>

  </tr>

 </tbody>

</table>

N(−1, 4) independently, and noise variance in y, a<sup>2</sup> = 2.

~90 ~

<ul>

 <li>(6 points) Implement Stochastic gradient descent method for optimizing J(9). Relearn 9 = 91</li>

</ul>

92

using sampled data points of part a) keeping everything same except the batch size. Keep η = 0.001 and initialize ∀j 9j = 0. Report the 9 learned each time separately for values of batch size r = {1, 100, 10000, 1000000}. Carefully decide your convergence criteria in each case. Make sure to watch the online video posted on the course website for deciding the convergence of SGD algorithm.

<ul>

 <li>(6 points) Do different algorithms in the part above (for varying values of r) converge to the same parameter values? How much different are these from the parameters of the original hypothesis from which the data was generated? Comment on the relative speed of convergence and also on number of iterations in each case. Next, for each of learned models above, report the error on a new test data of 10,000 samples provided in the file namedcsv. Note that this test set was generated using the same sampling procedure as described in part (a) above. Also, compute the test error with respect to the prediction of the original hypothesis, and compare with the error obtained using learned hypothesis in each case. Comment.</li>

 <li>(4 points) In the 3 dimensional parameter space(9j on each axis), plot the movement of 9 as the parameters are updated (until convergence) for varying batch sizes. How does the (shape of) movement compare in each case? Does it make intuitive sense? Argue.</li>

</ul>




<ol start="3">

 <li>(15 points) Logistic Regression Consider the log-likelihood function for logistic regression:</li>

</ol>

L(θ) = ~m y(i) log hθ(x<sup>(i)</sup>) + (1 − y(i)) log(1 − hθ(x<sup>(i)</sup>))

i=1

resulting from your fit? (Remember to include the intercept term.)

(b) (5 points) Plot the training data (your axes should be x1 and x2 , corresponding to the two coordinates of the inputs, and you should use a different symbol for each point plotted to indicate whether that example had label 1 or 0). Also plot on the same figure the decision boundary fit by logistic regression. (i.e., this should be a straight line showing the boundary separating the region where h(x) &gt; 0.5 from where h(x) ≤ 0.5.)

<ol start="4">

 <li>(25 points) Gaussian Discrmimant Analysis</li>

</ol>

In this problem, we will implement GDA for separating out salmons from Alaska and Canada. Each salmon is represented by two attributes x1 and x2 depicting growth ring diameters in 1) fresh water, 2) marine water, respectively. File q4x.dat stores the two attribute values with one entry on each row. File q4y.dat contains the target values (y<sup>(i)</sup>’s ∈ {Alaska, Canada}) on respective rows.

<ul>

 <li>(6 points) Implement Gaussian Discriminant Analysis using the closed form equations described in Assume that both the classes have the same co-variance matrix i.e. Σ0 = Σ1 = Σ. Report the values of the means, µ0 and µ1, and the co-variance matrix Σ.</li>

 <li>(2 points) Plot the training data corresponding to the two coordinates of the input features, and you should use a different symbol for each point plotted to indicate whether that example had label Canada or Alaska.</li>

 <li>(3 points) Describe the equation of the boundary separating the two regions in terms of the parameters µ0, µ1 and Σ. Recall that GDA results in a linear separator when the two classes have identical co­variance matrix. Along with the data points plotted in the part above, plot (on the same figure) decision boundary fit by GDA.</li>

 <li>(6 points) In general, GDA allows each of the target classes to have its own covariance matrix. This results (in general) results in a quadratic boundary separating the two class regions. In this case, the maximum-likelihood estimate of the co-variance matrix Σ0 can be derived using the equation:</li>

</ul>

~m 1{y<sup>(i)</sup> = 0}(x<sup>(i)</sup> − µ<sub>y</sub>(i))(x<sup>(i)</sup> − µ<sub>y</sub>(i))<sup>T</sup> i=1

1{y(<sup>i</sup>) = 0}

i=1

And similarly, for Σ1. The expressions for the means remain the same as before. Implement GDA for the above problem in this more general setting. Report the values of the parameter estimates i.e. µ0, µ1, Σ0, Σ1.

<ul>

 <li>(5 points) Describe the equation for the quadratic boundary separating the two regions in terms of the parameters µ0, µ1 and Σ0, Σ1. On the graph plotted earlier displaying the data points and the linear separating boundary, also plot the quadratic boundary obtained in the previous step.</li>

 <li>(3 points) Carefully analyze the linear as well as the quadratic boundaries obtained. Comment on your observations.</li>

</ul>

<strong>‘</strong>Write your own version, and do not call a built-in library function.