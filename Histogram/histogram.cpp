#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// ===================================================================
// �ο�/����
// ===================================================================
// ֱ��ͼ���� http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
// ֱ��ͼ���⻯ http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html#histogram-equalization
// ��OpenCV������ֱ��ͼ http://blog.csdn.net/xiaowei_cqu/article/details/8833799
// ��ɫͼ��ֱ��ͼ���⻯ --- ����OpenCV��EqualizeHist_Demoʵ�� http://blog.csdn.net/frank_xu_0818/article/details/39232157
// ��OpenCV����ָ�ϡ���ʮƪ ��ɫֱ��ͼ���⻯ http://blog.csdn.net/morewindows/article/details/8364722
// opencvѧϰ֮���壩-ֱ��ͼ����ͻ���ͼ��ֱ��ͼhttp://blog.csdn.net/dujian996099665/article/details/8894556
// How to fill OpenCV image with one solid color http://stackoverflow.com/questions/4337902/how-to-fill-opencv-image-with-one-solid-color
//
// equalizeHist http://docs.opencv.org/2.4.9/modules/imgproc/doc/histograms.html?highlight=equalizehist#void equalizeHist(InputArray src, OutputArray dst)
// split http://docs.opencv.org/2.4.9/modules/core/doc/operations_on_arrays.html?highlight=split#void split(const Mat& src, Mat* mvbegin)
// merge http://docs.opencv.org/2.4.9/modules/core/doc/operations_on_arrays.html?highlight=split#void merge(const Mat* mv, size_t count, OutputArray dst)
// Mat::zeros http://docs.opencv.org/2.4.9/modules/core/doc/basic_structures.html#static MatExpr Mat::zeros(int rows, int cols, int type)
// calcHist http://docs.opencv.org/modules/imgproc/doc/histograms.html?highlight=histogram#calchist
// cvtColor http://docs.opencv.org/2.4.9/modules/imgproc/doc/miscellaneous_transformations.html?highlight=cvtcolor#void cvtColor(InputArray src, OutputArray dst, int code, int dstCn)
// ===================================================================

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << " �÷�: histogram <image_file_path>" << endl;
		return -1;
	}

	// ��������
	void showimg(Mat const & image, char const * title);
	void showRGBHistogram(Mat const & image, char const * title);
	void showHSVHistogram(Mat const & image, char const * title);
	void showGrayHistogram(Mat const & image, char const * title);
	Mat equalizeMultiChannelsImage(Mat const & image, int channels);

	// ����ͼ��
	Mat image = imread(argv[1], IMREAD_COLOR);
	if (!image.data) {
		cout << " �޷���Դͼ��" << std::endl;
		return -1;
	}

	// Ԥ��ͼ��
	showimg(image, "ͼ��Ԥ��");

	// ��ʾ RGB ��ͨ��ֱ��ͼ
	showRGBHistogram(image, "RGB ��ͨ��ֱ��ͼ (���⻯ǰ)");

	// ���⻯ͼ�� (RGB)
	Mat rgb_equalized = equalizeMultiChannelsImage(image, 3);
	// Ԥ�����⻯���ͼ�� (RGB)
	showimg(rgb_equalized, "ͼ��Ԥ�� (���⻯��)");
	// ��ʾ���⻯��ͼ��� RGB ��ͨ��ֱ��ͼ
	showRGBHistogram(rgb_equalized, "RGB ��ͨ��ֱ��ͼ (���⻯��)");

	// �Ҷ�ͼ��
	Mat gray, gray_equalized;
	cvtColor(image, gray, CV_RGB2GRAY);
	showimg(gray, "�Ҷ�ͼ��Ԥ�� (���⻯ǰ)");
	showGrayHistogram(gray, "�Ҷ�ֱ��ͼ (���⻯ǰ)");
	equalizeHist(gray, gray_equalized);
	showimg(gray_equalized, "�Ҷ�ͼ��Ԥ�� (���⻯��)");
	showGrayHistogram(gray_equalized, "�Ҷ�ֱ��ͼ (���⻯��)");

	// HSV
	Mat hsv, hsv_equalized;
	cvtColor(image, hsv, CV_RGB2HSV);
	showHSVHistogram(hsv, "HSV ֱ��ͼ (���⻯ǰ)");
	hsv_equalized = equalizeMultiChannelsImage(hsv, 3);
	showHSVHistogram(hsv_equalized, "HSV ֱ��ͼ (���⻯��)");

	// YCrCb
	Mat ycrcb, ycrcb_equalized;
	cvtColor(image, ycrcb, CV_RGB2YCrCb);
	showHSVHistogram(ycrcb, "YCrCb ֱ��ͼ (���⻯ǰ)");
	ycrcb_equalized = equalizeMultiChannelsImage(ycrcb, 3);
	showHSVHistogram(ycrcb_equalized, "YCrCb ֱ��ͼ (���⻯��)");

	// HLS
	Mat hls, hls_equalized;
	cvtColor(image, hls, CV_RGB2HLS);
	showHSVHistogram(hls, "HLS ֱ��ͼ (���⻯ǰ)");
	hls_equalized = equalizeMultiChannelsImage(hls, 3);
	showHSVHistogram(hls_equalized, "HLS ֱ��ͼ (���⻯��)");

	// Lab
	Mat lab, lab_equalized;
	cvtColor(image, lab, CV_RGB2Lab);
	showHSVHistogram(lab, "Lab ֱ��ͼ (���⻯ǰ)");
	lab_equalized = equalizeMultiChannelsImage(lab, 3);
	showHSVHistogram(lab_equalized, "Lab ֱ��ͼ (���⻯��)");

	// Luv
	Mat luv, luv_equalized;
	cvtColor(image, luv, CV_RGB2Luv);
	showHSVHistogram(luv, "Luv ֱ��ͼ (���⻯ǰ)");
	luv_equalized = equalizeMultiChannelsImage(luv, 3);
	showHSVHistogram(luv_equalized, "Luv ֱ��ͼ (���⻯��)");

	return 0;
}

// ===================================================================
// Ԥ��ͼ��
// ===================================================================
void showimg(Mat const & image, char const * title) {
	namedWindow(title, WINDOW_AUTOSIZE);
	imshow(title, image);
	waitKey(0);
}

// ===================================================================
// ���⻯��ͨ��ͼ��
// ===================================================================
Mat equalizeMultiChannelsImage(Mat const & image, int channels) {
	Mat equalized;
	vector<Mat> planes;
	vector<Mat> equalized_planes;
	split(image, planes);
	for (int i = 0; i < channels; ++i) {
		Mat tmp;
		equalizeHist(planes[i], tmp);
		equalized_planes.push_back(tmp);
	}
	merge(equalized_planes, equalized);
	return equalized;
}

// ===================================================================
// ��ʾͼ�� RGB ֱ��ͼ
// ===================================================================
void showRGBHistogram(Mat const & image, char const * title) {
	void showTripleChannelsHistogram(Mat const & image, char const * title, CvScalar const color1, CvScalar const color2, CvScalar const color3);
	showTripleChannelsHistogram(image, title, CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), CV_RGB(0, 0, 255));
}

// ===================================================================
// ��ʾͼ�� HSV ֱ��ͼ
// ===================================================================
void showHSVHistogram(Mat const & image, char const * title) {
	void showTripleChannelsHistogram(Mat const & image, char const * title, CvScalar const color1, CvScalar const color2, CvScalar const color3);
	showTripleChannelsHistogram(image, title, CV_RGB(50, 50, 50), CV_RGB(80, 65, 0), CV_RGB(128, 6, 132));
}

// ===================================================================
// ��ʾ��ͨ��ֱ��ͼ
// ===================================================================
void showTripleChannelsHistogram(Mat const & image, char const * title, CvScalar const color1, CvScalar const color2, CvScalar const color3) {
	void showimg(Mat const & image, char const * title);

	int bins = 256;
	int histSize[] = { bins };

	float range[] = { 0, 256 };
	const float* ranges[] = { range };

	bool uniform = true;
	bool accumulate = false;

	int scale = 1;
	int hist_height = 256;
	int width_delta = 6;

	MatND hist_r, hist_g, hist_b;
	int channels_r[] = { 0 };
	int channels_g[] = { 1 };
	int channels_b[] = { 2 };

	// ����ֱ��ͼ
	calcHist(&image, 1, channels_r, Mat(), hist_r, 1, histSize, ranges, uniform, accumulate);
	calcHist(&image, 1, channels_g, Mat(), hist_g, 1, histSize, ranges, uniform, accumulate);
	calcHist(&image, 1, channels_b, Mat(), hist_b, 1, histSize, ranges, uniform, accumulate);

	double max_val_r, max_val_g, max_val_b;
	minMaxLoc(hist_r, 0, &max_val_r, 0, 0);
	minMaxLoc(hist_g, 0, &max_val_g, 0, 0);
	minMaxLoc(hist_b, 0, &max_val_b, 0, 0);
	Mat hist_img = Mat::zeros(hist_height, bins * 3 + 2 * width_delta, CV_8UC3);
	hist_img.setTo(cv::Scalar(255, 255, 255)); // ���ñ���

	// ����ֱ��ͼ
	for (int i = 0; i<bins; i++) {
		float bin_val_r = hist_r.at<float>(i);
		float bin_val_g = hist_g.at<float>(i);
		float bin_val_b = hist_b.at<float>(i);

		int intensity_r = cvRound(bin_val_r * hist_height / max_val_r);
		int intensity_g = cvRound(bin_val_g * hist_height / max_val_g);
		int intensity_b = cvRound(bin_val_b * hist_height / max_val_b);

		rectangle(hist_img, Point(i * scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity_r), color1);
		rectangle(hist_img, Point((i + bins) * scale + width_delta, hist_height - 1),
			Point((i + bins + 1) * scale + width_delta - 1, hist_height - intensity_g), color2);
		rectangle(hist_img, Point((i + bins * 2) * scale + 2 * width_delta, hist_height - 1),
			Point((i + bins * 2 + 1) * scale + 2 * width_delta - 1, hist_height - intensity_b), color3);
	}

	// ��ʾֱ��ͼ
	showimg(hist_img, title);
}

// ===================================================================
// ��ʾ�Ҷ�ֱ��ͼ
// ===================================================================
void showGrayHistogram(Mat const & image, char const * title) {
	void showimg(Mat const & image, char const * title);

	Mat hist, histImage;
	int histSize = 256;
	int height = 256;
	int scale = 2;

	calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, 0);

	histImage = Mat::zeros(height, histSize * scale, CV_8UC3);
	histImage.setTo(cv::Scalar(255, 255, 255));

	double max_val;
	minMaxLoc(hist, 0, &max_val, 0, 0);

	for (int i = 0; i < histSize; i++) {
		float bin_val = hist.at<float>(i);
		int intensity = cvRound(bin_val * height / max_val);
		rectangle(histImage, Point(i * scale, height - 1),
			Point((i + 1)*scale - 1, height - intensity),
			CV_RGB(50, 50, 50), scale);
	}

	// ��ʾֱ��ͼ
	showimg(histImage, title);
}
