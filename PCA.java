package core.matrix;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import ai.grid.common.CoordinateBean;
import ai.grid.common.Tuple;
import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class PCA {

	// ���Ļ���������ֳ�covariance�����������
	public double[][] covariance(double[][] x) {
		int m = x.length; // ��������
		int d = x[0].length; // ����ά��
		double[][] mu = new double[m][d]; // ά��ƽ��ֵ

		// ȡ��ÿһ��ά�ȵ�ƽ��ֵ
		for (int k = 0; k < d; k++) {
			double temp = 0;
			for (int i = 0; i < m; i++) {
				temp += x[i][k];
			}
			temp = temp / m;

			for (int j = 0; j < m; j++) {
				mu[j][k] = temp;
			}
		}

		Matrix xMat = new Matrix(x);
		Matrix muMat = new Matrix(mu);
		Matrix centralMat = xMat.minus(muMat);
		// (X - u)^T(X - u)
		Matrix e2 = centralMat.transpose().times(centralMat);
		e2 = e2.times(1. / (m - 1));
		return e2.getArray();
	}

	// ��ԭʼ���ݱ�׼��
	public double[][] Standardlizer(double[][] x) {
		int n = x.length; // ������к�
		int p = x[0].length; // ������к�
		double[] average = new double[p]; // ÿһ�е�ƽ��ֵ
		double[][] result = new double[n][p]; // ��׼���������
		double[] var = new double[p]; // ����
		// ȡ��ÿһ�е�ƽ��ֵ
		for (int k = 0; k < p; k++) {
			double temp = 0;
			for (int i = 0; i < n; i++) {
				temp += x[i][k];
			}
			average[k] = temp / n;
		}
		// ȡ�÷���
		for (int k = 0; k < p; k++) {
			double temp = 0;
			for (int i = 0; i < n; i++) {
				temp += (x[i][k] - average[k]) * (x[i][k] - average[k]);
			}
			var[k] = temp / (n - 1);
		}
		// ��ñ�׼���ľ���
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < p; j++) {
				result[i][j] = (double) ((x[i][j] - average[j]) / Math
						.sqrt(var[j]));
			}
		}
		return result;
	}

	// �����������ϵ������
	public double[][] CoefficientOfAssociation(double[][] x) {
		int n = x.length; // ������к�
		int p = x[0].length; // ������к�
		double[][] result = new double[p][p];// ���ϵ������
		for (int i = 0; i < p; i++) {
			for (int j = 0; j < p; j++) {
				double temp = 0;
				for (int k = 0; k < n; k++) {
					temp += x[k][i] * x[k][j];
				}
				result[i][j] = temp / (n - 1);
			}
		}
		return result;

	}

	// �������ϵ�����������ֵ
	public double[][] FlagValue(double[][] x) {
		// ����һ������
		Matrix A = new Matrix(x);
		// ������ֵ��ɵĶԽǾ���
		Matrix B = A.eig().getD();
		double[][] result = B.getArray();
		return result;

	}

	// �������ϵ���������������
	public double[][] FlagVector(double[][] x) {
		// ����һ������
		Matrix A = new Matrix(x);
		// ������������ɵĶԽǾ���
		Matrix B = A.eig().getV();
		double[][] result = B.getArray();

		return result;
	}

	// ���м�����������
	public double[][] eigenVectorsU(double[][] x) {
		// ����һ������
		Matrix A = new Matrix(x);

		SingularValueDecomposition svd = A.svd();
		// ����������
		Matrix result = svd.getU();
		return result.transpose().getArray();
	}

	// pca �㷨
	public double[][] principleComponent(double[][] x) {
		// ���Ļ�
		x = covariance(x);
		// ��ʾת��ǰ��covariance, ������
		// System.out.println("ת��ǰ��Э����");
		// new Matrix(x).print(2, 2);
		// ����������
		return eigenVectorsU(x);
	}

	public int[] SelectPrincipalComponent(double[][] x, double threshold) { // ����������ֵ����thresholdѡ�����ɷ�
		int n = x.length; // ������к�,�к�
		double[] a = new double[n];
		int[] result = new int[n];
		int k = 0;
		double temp = 0;
		int m = 0;
		double total = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j) {
					a[k] = x[i][j];
				}
			}
			k++;
		}
		double[] temp1 = new double[a.length];

		System.arraycopy(a, 0, temp1, 0, a.length);
		for (int i = 0; i < n; i++) {
			temp = temp1[i];
			for (int j = i; j < n; j++) {
				if (temp <= temp1[j]) {
					temp = temp1[j];
					temp1[j] = temp1[i];
				}

				temp1[i] = temp;
			}
		}
		for (int i = 0; i < n; i++) {
			temp = a[i];
			for (int j = 0; j < n; j++) {
				if (a[j] >= temp) {
					temp = a[j];

					k = j;

				}
				result[m] = k;

			}
			a[k] = -1000;

			m++;
		}
		for (int i = 0; i < n; i++) {
			total += temp1[i];
		}
		int sum = 1;
		temp = temp1[0];
		for (int i = 0; i < n; i++) {
			if (temp / total <= threshold) {
				temp += temp1[i + 1];
				sum++;
			}
		}
		int[] end = new int[sum];
		System.arraycopy(result, 0, end, 0, sum);

		return end;

	}

	// ȡ�����ɷ�
	public double[][] PrincipalComponent(double[][] x, int[] y) {
		int n = x.length;
		double[][] Result = new double[n][y.length];
		int k = y.length - 1;
		for (int i = 0; i < y.length; i++) {
			for (int j = 0; j < n; j++) {
				Result[j][i] = x[j][y[k]];
			}
			k--;
		}
		return Result;

	}

	// ӳ�䵽pca������ϵ
	public double[][] rotateByPCA(double[][] x) {
		// ������
		double[][] pc = principleComponent(x);
		Matrix u = new Matrix(pc);
		// �����������ͶӰ
		Matrix result = new Matrix(x).times(u.transpose());
		return result.getArray();
	}

	// ��Tuple���ϣ�ӳ�䵽pca������ϵ��ͬ��
	public List<Tuple<Integer>> rotateByPCA(Collection<Tuple<Integer>> coll) {
		// ת��Ϊ����
		List<Tuple<Integer>> lst = new ArrayList<Tuple<Integer>>();

		// ��������
		int m = coll.size();
		double[][] data = new double[m][2];

		// ��ȡ��
		Iterator<Tuple<Integer>> it = coll.iterator();
		for (int i = 0; i < m; i++) {
			Tuple<?> tuple = it.next();
			int x = tuple.getX();
			int y = tuple.getY();
			// ��¼ԭʼ����Ϣ
			Integer type = (Integer) tuple.getType();
			lst.add(new Tuple<Integer>(x, y, type));
			data[i][0] = x;
			data[i][1] = y;
		}

		// ����pca������ת
		double[][] pc = principleComponent(data);
		Matrix u = new Matrix(pc);
		Matrix rotatedMat = new Matrix(data).times(u.transpose());
		double[][] rotated = rotatedMat.getArray();

		// ��ʾת�����covariance��������
		// System.out.println("ת�����Э����");
		// new Matrix(covariance(rotated)).print(2, 2);

		// ������ת���ĵ�
		for (int i = 0; i < m; i++) {
			Tuple<Integer> t = lst.get(i);
			t.setX((int) rotated[i][0]);
			t.setY((int) rotated[i][1]);
		}
		return lst;
	}

	@Deprecated
	public static double[][] matrixRotate(double[][] M) {
		Matrix A = new Matrix(M);
		System.out
				.println("====================ԭ����AΪ:===============================");
		A.print(A.getColumnDimension(), A.getRowDimension());
		Matrix squareMatrix = A.transpose().times(A);
		System.out
				.println("====================A^T*A�ķ���Ϊ:===============================");
		squareMatrix.print(squareMatrix.getColumnDimension(),
				squareMatrix.getRowDimension());
		EigenvalueDecomposition Eig = squareMatrix.eig();
		Matrix D = Eig.getD();
		System.out
				.println("====================�������ֵΪ===============================");
		D.print(D.getColumnDimension(), D.getRowDimension());// ��ӡ����ֵ
		Matrix V = Eig.getV();
		System.out
				.println("====================������������============================");
		V.print(V.getColumnDimension(), V.getRowDimension());// ��ӡ��������
		// ��ʾŷ����¿ռ�������ת
		Matrix M_V = A.times(V);
		System.out
				.println("=====================ŷ����¿ռ�������ת����=========================");
		M_V.print(M_V.getColumnDimension(), M_V.getRowDimension());
		// double[][] MT = M_V.getArray();
		// print(MT);
		return M_V.getArray();
	}

	/**
	 * ����list��set����
	 * 
	 * @param coll
	 * @return
	 */
	@Deprecated
	@SuppressWarnings("unchecked")
	public static <T> T matrixRotate(Collection<CoordinateBean> coll) {
		double[][] data;
		int index = 0;
		try {
			if (coll instanceof List<?>) {
				System.out.println("list");
				List<CoordinateBean> list = (List<CoordinateBean>) coll;
				data = new double[list.size()][2];
				// ��list���ϵ�����ת����double[][]������ʽ
				for (Iterator<CoordinateBean> it = list.iterator(); it
						.hasNext();) {
					CoordinateBean bean = it.next();
					data[index][0] = bean.getX();
					data[index][1] = bean.getY();
					index++;
				}
				//
				data = matrixRotate(data);
				List<CoordinateBean> list1 = null;
				list1 = list.getClass().newInstance();
				for (int i = 0; i < data.length; i++) {
					CoordinateBean bean = new CoordinateBean();
					bean.setX(data[i][0]);
					bean.setY(data[i][1]);
					list1.add(bean);
				}
				return (T) list1;
			} else if (coll instanceof Set<?>) {
				System.out.println("set");
				Set<CoordinateBean> set = (Set<CoordinateBean>) coll;
				data = new double[set.size()][2];
				// ��set���ϵ�����ת����double[][]������ʽ
				for (Iterator<CoordinateBean> it = set.iterator(); it.hasNext();) {
					CoordinateBean bean = it.next();
					data[index][0] = bean.getX();
					data[index][1] = bean.getY();
					index++;
				}
				data = matrixRotate(data);
				Set<CoordinateBean> set1 = null;
				set1 = set.getClass().newInstance();
				for (int i = 0; i < data.length; i++) {
					CoordinateBean bean = new CoordinateBean();
					bean.setX(data[i][0]);
					bean.setY(data[i][1]);
					set1.add(bean);
				}
				return (T) set1;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}

	public static void main(String[] args) {

		double[][] RawData = new double[][] {
				{ 40.4, 24.7, 7.2, 6.1, 8.3, 8.7, 2.442, 20.0, 5.5, 3.1, 2.2 },
				{ 25.0, 12.7, 11.2, 11.0, 12.9, 20.2, 3.542, 9.1, 4.6, 2.3,
						1.03 },
				{ 13.2, 3.3, 3.9, 4.3, 4.4, 5.5, 0.578, 3.6, 3.0, 1.3, 15.6 },
				{ 22.3, 6.7, 5.6, 3.7, 6.0, 7.4, 0.176, 7.3, 22.3, 10.5, 13.5 },
				{ 34.3, 11.8, 7.1, 7.1, 8.0, 8.9, 1.726, 27.5, 8.1, 1.6, 3.21 },
				{ 35.6, 12.5, 16.4, 16.7, 22.8, 29.3, 3.017, 26.6, 33.2, 20.0,
						9.23 },
				{ 22.0, 7.8, 9.9, 10.2, 12.6, 17.6, 0.847, 10.6, 20.0, 14.0,
						2.45 },
				{ 48.4, 13.4, 10.9, 9.9, 10.9, 13.9, 1.772, 1.772, 1.035,
						0.332, 5.36 },
				{ 40.6, 19.1, 19.8, 19.0, 29.7, 39.6, 2.449, 35.8, 12.8, 8.25,
						0.36 },
				{ 24.8, 8.0, 9.8, 8.9, 11.9, 16.2, 0.789, 13.7, 6.32, 0.33,
						5.11 } };

		Matrix mat = new Matrix(RawData);

		PCA pca = new PCA();
		double[][] a = mat.times(mat.transpose()).getArray();
		double[][] u = pca.eigenVectorsU(a);
		new Matrix(u).print(2, 2);
		u = pca.FlagVector(a);
		new Matrix(u).print(2, 2);
		
		
		double[][] result = pca.principleComponent(mat.getArray()); // ����pca����
		//new Matrix(result).print(2, 2);

		/*
		 * Tuple<Integer> t = new Tuple<Integer>(1, 1, 1); Tuple<Integer> t1 =
		 * new Tuple<Integer>(2, 2, 2); Tuple<Integer> t2 = new
		 * Tuple<Integer>(3, 3, 3);
		 * 
		 * Set<Tuple<Integer>> a = new HashSet<Tuple<Integer>>(); a.add(t);
		 * a.add(t1); a.add(t2);
		 * 
		 * List<Set<Tuple<Integer>>> l = new ArrayList<Set<Tuple<Integer>>>();
		 * l.add(a);
		 * 
		 * PCA pca = new PCA(); List<Tuple<Integer>> x = (List<Tuple<Integer>>)
		 * pca.rotateByPCA(a);
		 * 
		 * System.out.print(x);
		 */

		/*
		 * public static void main(String[] args) { PCA test=new PCA(); //ԭʼ����
		 * double[][] RawData=new double[][]
		 * {{40.4,24.7,7.2,6.1,8.3,8.7,2.442,20.0,5.5,3.1,2.2},
		 * {25.0,12.7,11.2,11.0,12.9,20.2,3.542,9.1,4.6,2.3,1.03},
		 * {13.2,3.3,3.9,4.3,4.4,5.5,0.578,3.6,3.0,1.3,15.6},
		 * {22.3,6.7,5.6,3.7,6.0,7.4,0.176,7.3,22.3,10.5,13.5},
		 * {34.3,11.8,7.1,7.1,8.0,8.9,1.726,27.5,8.1,1.6,3.21},
		 * {35.6,12.5,16.4,16.7,22.8,29.3,3.017,26.6,33.2,20.0,9.23},
		 * {22.0,7.8,9.9,10.2,12.6,17.6,0.847,10.6,20.0,14.0,2.45},
		 * {48.4,13.4,10.9,9.9,10.9,13.9,1.772,1.772,1.035,0.332,5.36},
		 * {40.6,19.1,19.8,19.0,29.7,39.6,2.449,35.8,12.8,8.25,0.36},
		 * {24.8,8.0,9.8,8.9,11.9,16.2,0.789,13.7,6.32,0.33,5.11}};
		 * System.out.println("ԭʼ��������Ϊ10x11ά�ľ������£�"); for(int
		 * i=0;i<RawData.length;i++){ for(int j=0;j<RawData[0].length;j++){
		 * System.out.print(RawData[i][j]+" "); } System.out.println(); }
		 * 
		 * System.out.print("��������ֵ��С����ΧΪ0-1����"); Scanner sc = new
		 * Scanner(System.in); double threshold=sc.nextDouble(); //��׼��������
		 * double[][] Standard=test.Standardlizer(RawData); double[][]
		 * Assosiation=test.CoefficientOfAssociation(Standard); int
		 * n=RawData.length; int p=RawData[0].length; for(int i=0;i<p;i++){
		 * for(int j=0;j<p;j++){ System.out.print(Assosiation[i][j]+" "); }
		 * System.out.println(); } System.out.println();
		 * 
		 * double[][] FlagValue=test.FlagValue(Assosiation);
		 * 
		 * for(int i=0;i<p;i++){ for(int j=0;j<p;j++){
		 * System.out.print(FlagValue[i][j]+" "); } System.out.println(); }
		 * System.out.println();
		 * 
		 * double[][] FlagVector=test.FlagVector(Assosiation); for(int
		 * i=0;i<p;i++){ for(int j=0;j<p;j++){
		 * System.out.print(FlagVector[i][j]+" "); } System.out.println(); }
		 * System.out.println();
		 * 
		 * 
		 * int[] xuan=test.SelectPrincipalComponent(FlagValue,threshold);
		 * for(int i=0;i<xuan.length;i++){
		 * System.out.print(".................");
		 * System.out.println(xuan[i]+" ");
		 * 
		 * } System.out.println(); double[][]
		 * result=test.PrincipalComponent(FlagVector, xuan); for(int
		 * i=0;i<p;i++){ for(int j=0;j<xuan.length;j++){
		 * System.out.print(result[i][j]+" "); } System.out.println(); } Matrix
		 * A=new Matrix(RawData); Matrix B=new Matrix(result); Matrix
		 * C=A.times(B); C.print(4, 2); double[][] D=C.getArray(); double[]
		 * E=new double[n]; for(int i=0;i<D.length;i++){ double temp=0; for(int
		 * j=0;j<D[0].length;j++){ temp+=D[i][j]; } E[i]=temp; }
		 * System.out.println(); for(int i=0;i<D.length;i++){
		 * System.out.println(E[i]); }
		 * 
		 * }
		 */
	}
}
