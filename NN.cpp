// NN v1.0
// Just for binary classification
// Loss prints -inf sometimes idk why
#include<iostream>
using namespace std;
#include<random>
#include<cmath>
std::mt19937 engine(std::random_device{}());
std::uniform_real_distribution<double> distribution(-1.0, 1.0);
class Matrix {
public:
	int rows;
	int cols;
	double** value;
	Matrix(int r, int c) {
		rows = r;
		cols = c;
		value = new double* [rows];
		for (int i = 0; i < rows; i++) {
			value[i] = new double[cols];
		}
	}
	void shape() {
		cout <<endl<< "(" << rows << " ," << cols << ")"<<endl;
	}
	void random_init() {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				value[i][j] = distribution(engine);
			}
		}
	}
	void print() {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				cout << value[i][j] << " ";
			}
			cout << endl;
		}
	}
	// overload - 
	Matrix* operator-(const Matrix* rhs) {
		Matrix* resultant_matrix = new Matrix(this->rows, this->cols);
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				resultant_matrix->value[i][j] = this->value[i][j] - rhs->value[i][j];
			}
		}
		return resultant_matrix;
	}
	~Matrix() {
		for (int i = 0; i < rows; i++) {
			delete value[i];
		}
		delete value;
	}
};
void matmul(Matrix * a, Matrix * b, Matrix* resultant_matrix) {
	for (int i = 0; i < a->rows; i++) {
		for (int k = 0; k < b->cols; k++) {
			double sum = 0.0;
			for (int j = 0; j < a->cols; j++) {
				sum += a->value[i][j] * b->value[j][k];
			}
			resultant_matrix->value[i][k] = sum;
		}
	}
}
Matrix* transpose(Matrix* a) {
	Matrix* resultant_matrix = new Matrix(a->cols, a->rows);
	for (int i = 0; i < a->rows; i++) {
		for (int j = 0; j < a->cols; j++) {
			resultant_matrix->value[j][i] = a->value[i][j];
		}
	}
	return resultant_matrix;
}

void sigmoid(Matrix* z, Matrix* resultant_matrix) {
	for (int i = 0; i < z->rows; i++) {
		for (int j = 0; j < z->cols; j++) {
			resultant_matrix->value[i][j] = 1.0/(1.0 + exp(-z->value[i][j]));
		}
	}
}


void elementwise_multiply(Matrix * mat1, Matrix * mat2, Matrix* resultant_matrix) {
	if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
		std::cerr << "Error: Matrices must have the same dimensions for element-wise multiplication." << std::endl;
		return;
	}
	for (int i = 0; i < mat1->rows; ++i) {
		for (int j = 0; j < mat1->cols; ++j) {
			resultant_matrix->value[i][j] = mat1->value[i][j] * mat2->value[i][j];
		}
	}
}

void const_sub(double scalar, Matrix* mat, Matrix* resultant_matrix) {
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			resultant_matrix->value[i][j] = scalar - mat->value[i][j];
		}
	}
}

double sum(Matrix* mat) {
	double summ = 0.0;
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			summ += mat->value[i][j];
		}
	}
	return summ;
}
Matrix* broadcast(double scalar, Matrix* shape_mat) {
	Matrix* resultant_matrix = new Matrix(shape_mat->rows, shape_mat->cols);
	for (int i = 0; i < shape_mat->rows; i++) {
		for (int j = 0; j < shape_mat->cols; j++) {
			resultant_matrix->value[i][j] = scalar;
		}
	}
	return resultant_matrix;
}

Matrix* scalar_mul(double scalar, Matrix* mat) {
	Matrix* resultant_matrix = new Matrix(mat->rows, mat->cols);
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			resultant_matrix->value[i][j] = scalar * mat->value[i][j];
		}
	}
	return resultant_matrix;
}


void matadd(Matrix* mat1, Matrix* mat2, Matrix* resultant_matrix) {
	for (int i = 0; i < mat1->rows; ++i) {
		for (int j = 0; j < mat1->cols; ++j) {
			resultant_matrix->value[i][j] = mat1->value[i][j] + mat2->value[0][i];
		}
	}
}

class Dense {
public:
	int in_features;
	int out_features;
    Matrix *weights;
	Matrix* bias;
	Matrix* dz;
	Matrix* dw;
	double db;
	Matrix* a;
	Matrix* z;
	Matrix* intermediate_stuff_back_1;
	Matrix* intermediate_stuff_back_2;
	Dense(int in, int out) {
		in_features = in;
		out_features = out;
		Matrix*temp = new Matrix(in, out);
		Matrix* b = new Matrix(1,out);
		b->random_init();
		temp->random_init();
		weights = temp;
		bias = b;
	}
	~Dense() {
		delete weights;
		delete bias;
		delete dz;
		delete dw;
		delete a;
	}
};
class Model {
public:
	Dense** layers;
	static int num_layers;
	void add(Dense* D) {
		num_layers += 1;
		if (num_layers == 0) {
			layers = new Dense * [1];
			layers[0] = D;
		}
		else {
			Dense** new_array = new Dense * [num_layers];
			for (int i = 0; i < num_layers - 1; i++) {
				new_array[i] = layers[i];
			}
			new_array[num_layers - 1] = D;
			layers = new_array;
		}
	}
	void summary() {
		cout << endl << endl;
		for (int i = 0; i < num_layers; i++) {
			cout<<"Dense Layer "<<i<<" : ("<< layers[i]->weights->rows<<" ,"<< layers[i]->weights->cols<<")"<<endl;
		}
	}
	void forward(Matrix* train_x) {
		for (int i = 0; i < num_layers; i++) {
			if (i == 0) {
				matmul(train_x, layers[i]->weights, layers[i]->z);
				matadd(layers[i]->z,layers[i]->bias, layers[i]->z);
				sigmoid(layers[i]->z, layers[i]->a);
			}
			else {
				matmul(layers[i - 1]->a, layers[i]->weights, layers[i]->z);
				matadd(layers[i]->z, layers[i]->bias, layers[i]->z);
				sigmoid(layers[i]->z, layers[i]->a);
			}
		}
	}
	double loss(Matrix* y_true, Matrix* y_pred) {
		Matrix* y_pred_tr = transpose(y_pred);
		double l = 0.0;
		for (int i = 0; i < y_true->cols; i++) {
			l += -y_true->value[0][i] * log(y_pred_tr->value[0][i]) - (1 - y_true->value[0][i]) * log(1 - y_pred_tr->value[0][i]);
		}
		return l / y_true->cols;
	}
	void backward(Matrix* train_x, Matrix* train_y, double learning_rate) {
		for (int i = num_layers - 1; i >= 0; i--) {
			if (i == num_layers - 1) {
				Matrix* tr = transpose(layers[i]->a);
				layers[i]->dz = *tr - train_y;
				matmul(layers[i]->dz, layers[i - 1]->a, layers[i]->dw);
				layers[i]->db = (1.0 / (train_x->rows)) * sum(layers[i]->dz);
				Matrix* dw_trans = transpose(layers[i]->dw);
				layers[i]->weights = *layers[i]->weights - scalar_mul(learning_rate,dw_trans);
				Matrix* br = broadcast(layers[i]->db, layers[i]->bias);
				layers[i]->bias = *layers[i]->bias - scalar_mul(learning_rate, br);
				delete dw_trans;
				delete br;
				delete tr;
			}
			else if (i == 0) {
				Matrix* tr = transpose(layers[i]->a);
				const_sub(1, tr, layers[i]->intermediate_stuff_back_1);
				elementwise_multiply(tr, layers[i]->intermediate_stuff_back_1, layers[i]->intermediate_stuff_back_1);
				matmul(layers[i + 1]->weights, layers[i + 1]->dz, layers[i]->intermediate_stuff_back_2);
				elementwise_multiply(layers[i]->intermediate_stuff_back_1, layers[i]->intermediate_stuff_back_2, layers[i]->intermediate_stuff_back_2);


				layers[i]->dz = scalar_mul(1.0 / (train_x->rows), layers[i]->intermediate_stuff_back_2);
				
				matmul(layers[i]->dz, train_x, layers[i]->dw);
				layers[i]->db = (1.0 / (train_x->rows)) * sum(layers[i]->dz);
				Matrix* dw_trans = transpose(layers[i]->dw);
				layers[i]->weights = *layers[i]->weights - scalar_mul(learning_rate, dw_trans);
				Matrix* br = broadcast(layers[i]->db, layers[i]->bias);
				layers[i]->bias = *layers[i]->bias - scalar_mul(learning_rate, br);
				delete dw_trans;
				delete br;
				delete tr;
			}
			else {
				Matrix* tr = transpose(layers[i]->a);
				// tr -> (train_examples, num_nodes_in_i)
				// just a scary way of saying 1/m matmul(w[i+1],dz[i+1]) * a[i] * (1 - a[i])
				const_sub(1, tr, layers[i]->intermediate_stuff_back_1);
				elementwise_multiply(tr, layers[i]->intermediate_stuff_back_1, layers[i]->intermediate_stuff_back_1);
				matmul(layers[i + 1]->weights, layers[i + 1]->dz, layers[i]->intermediate_stuff_back_2);
				elementwise_multiply(layers[i]->intermediate_stuff_back_1, layers[i]->intermediate_stuff_back_2, layers[i]->intermediate_stuff_back_2);
				layers[i]->dz = scalar_mul(1.0 / (train_x->rows), layers[i]->intermediate_stuff_back_2);
				matmul(layers[i]->dz, layers[i - 1]->a, layers[i]->dw);
				layers[i]->db = (1.0 / (train_x->rows))* sum(layers[i]->dz);
				Matrix* dw_trans = transpose(layers[i]->dw);
				layers[i]->weights = *layers[i]->weights - scalar_mul(learning_rate, dw_trans);
				Matrix* br = broadcast(layers[i]->db, layers[i]->bias);
				layers[i]->bias = *layers[i]->bias - scalar_mul(learning_rate, br);
				delete dw_trans;
				delete br;
				delete tr;
			}
		}
	}
	void train(Matrix* train_x, Matrix*train_y, double learning_rate, int epochs) {

		for (int i = 0; i < num_layers; i++) {
			layers[i]->intermediate_stuff_back_1 = new Matrix(layers[i]->out_features, train_x->rows);
			layers[i]->intermediate_stuff_back_2 = new Matrix(layers[i]->out_features, train_x->rows);

			layers[i]->dw = new Matrix(layers[i]->out_features, layers[i]->in_features);
			layers[i]->z = new Matrix(train_x->rows, layers[i]->out_features);
			layers[i]->a = new Matrix(train_x->rows, layers[i]->out_features);
		}
		for (int epoch = 0; epoch < epochs; epoch++) {
			forward(train_x);
			if (epoch % 100 == 0) {
				cout << "Epochs : " << epoch << "      ";
				cout << "Loss : " << loss(train_y, layers[num_layers - 1]->a) << endl;
			}
			backward(train_x, train_y, learning_rate);
		}
		cout << "------------------------------------------------------"<<endl;
	}
	~Model() {
		for (int i = 0; i < num_layers; i++) {
			delete layers[i];
		}
	}
};
int Model::num_layers = 0;


int main() {
	Dense * D1 = new Dense(2, 128);
	Dense* D2 = new Dense(128, 64);
	Dense* D3 = new Dense(64, 1);
	Model model;
	model.add(D1);
	model.add(D2);
	model.add(D3);
	cout << model.num_layers;
	// x should be (training_examples, num_features)
	// y should be (1, training_examples)
	double x[4][2] = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
	double y[1][4] = { {0.0,1.0,1.0,0.0} };
	Matrix* X = new Matrix(4, 2);
	Matrix* Y = new Matrix(1, 4);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 2; j++) {
			X->value[i][j] = x[i][j];
		}
	}
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 4; j++) {
			Y->value[i][j] = y[i][j];
		}
	}

	model.summary();
	model.train(X, Y, 0.1, 1000);
	model.layers[model.num_layers - 1]->a->print();
}