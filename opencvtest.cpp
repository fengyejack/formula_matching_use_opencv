#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace cv;
using namespace std;

//calibration variables
int Y_MIN  = 42;
int Y_MAX  = 108;
int Cr_MIN = 129;
int Cr_MAX = 173;
int Cb_MIN = 0;
int Cb_MAX = 247;
int c_type=0;
int sub_type=0;

Scalar paintColor;
int cStatus=0;
int clickStatus=0;

Rect rect1(200,2,60,60);
Rect rect2(270,2,60,60);
Rect rect3(340,2,60,60);
Rect rect4(410,2,60,60);
//Rect rect5(384,2,60,60);

vector<Point> drawline;
vector<vector<Point> > reddraw;
vector<vector<Point> > bluedraw;
vector<vector<Point> > greendraw;
VideoWriter outputVideo;


Point2f center;
float radius;
//background substraction
Ptr<BackgroundSubtractor> pMOG2;

//points distance calculator
double calDist(Point a, Point b){
	return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

//points angle calculator
double calAngle(Point a,Point b,Point c){
	double ac=sqrt(pow(c.x-a.x,2)+pow(c.y-a.y,2));
	double bc=sqrt(pow(c.x-b.x,2)+pow(c.y-b.y,2));
	double ab=sqrt(pow(b.x-a.x,2)+pow(b.y-a.y,2));

	return acos((bc*bc+ac*ac-ab*ab)/(2*bc*ac));
}

void drawedge(vector<vector<Point> >,Mat& );
vector<vector<Vec4i> > getDefects(vector<vector<Point> >& );
void drawdefects(vector<vector<Vec4i> > ,vector<vector<Point> > ,Mat& );
Point isClick(vector<vector<Vec4i> > ,vector<vector<Point> > , vector<vector<Vec4i> > );
void judgePoint(Point );
Point getCurrentLocation(vector<vector<Point> > , vector<vector<Vec4i> > );
int main(int argc, char** argv){

	//camera data 
	VideoCapture cap(0);
	Size S=Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	/*
	int ex=static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
	char EXT[] = {ex & 0XFF , (ex & 0XFF00) >> 8,(ex & 0XFF0000) >> 16,(ex & 0XFF000000) >> 24, 0};
	VideoWriter outputVideo;
	outputVideo.open("sample.avi",ex,cap.get(CV_CAP_PROP_FPS),S,true);	
*/
	outputVideo.open("out1.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, S, true);

	if(!cap.isOpened()) return -1;
	pMOG2 = new BackgroundSubtractorMOG2(); 

	vector<vector<Vec4i> > lastDefects;
	vector<vector<Point> > lastcontours;

	//processing
	while(true){
		Mat frame;
		cap>>frame;
		
		//smoothing
		if(c_type==0)	medianBlur(frame,frame,5);
		if(c_type==1)   blur(frame,frame,Size(3,3));
		if(c_type==2)   GaussianBlur( frame,frame, Size(3,3), 0, 0 );
		
		Mat dst;  //result image
		

		//substraction mask
		Mat sub_mask;
		frame.copyTo(sub_mask);
		cvtColor(sub_mask,sub_mask,CV_BGR2GRAY);	
		pMOG2->operator()(sub_mask,sub_mask);
		threshold(sub_mask,sub_mask,100,255,THRESH_BINARY);
		//frame.copyTo(dst,sub_mask);		



		// HSV or YCrCb color mask
		Mat color_mask;
		//frame.copyTo(color_mask,sub_mask);
		frame.copyTo(color_mask);
		if(sub_type==0) cvtColor(color_mask,color_mask,CV_BGR2YCrCb);
		if(sub_type==1) cvtColor(color_mask,color_mask,CV_BGR2HSV);
		
		inRange(color_mask,Scalar(Y_MIN,Cr_MIN,Cb_MIN),Scalar(Y_MAX,Cr_MAX,Cb_MAX),color_mask);

		
		threshold(color_mask,color_mask,100,255,THRESH_BINARY);
		

		Mat mask;
		//bitwise_or(sub_mask,color_mask,mask);
		//sub_mask.copyTo(mask);
		color_mask.copyTo(mask);
		

		
		//erosion &dilate
		Mat dst1; //result image1
		mask.copyTo(dst1);
		Mat element = getStructuringElement(MORPH_RECT, Size(3,3)); 
		erode(dst1,dst1, element);  
        	morphologyEx(dst1,dst1, MORPH_OPEN, element);  
        	dilate(dst1,dst1, element);  
       		morphologyEx(dst1,dst1, MORPH_CLOSE, element);  
		GaussianBlur( dst1,dst1, Size(3,3), 0, 0 );

		


		
		//findcontours
		Mat src_copy = dst1.clone(); //result image 3
   		Mat threshold_output;
   		vector<vector<Point> > contours;
  		vector<Vec4i> hierarchy;
		Canny(src_copy,threshold_output,100,200,3);
		Mat  drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

		
		findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		if(contours.size()>0){
			vector< vector<Point> > filterContours;
			for(int i=0;i<contours.size();i++){
				if(fabs(contourArea(Mat(contours[i])))>5000){
					filterContours.push_back(contours[i]);
				}	
			}
			
			drawedge(filterContours,frame);
			vector<vector<Vec4i> > defects=getDefects(filterContours);
			//cout<<lastDefects.size()<<" "<<defects.size()<<endl;
			if(lastDefects.size()!=0&&defects.size()!=0){
				Point p=isClick(lastDefects,lastcontours,defects);
				//cout<<p<<endl;
				judgePoint(p);
				if(clickStatus==1&&cStatus>=1) {
					Point tpp=getCurrentLocation(filterContours,defects);
					//cout<<tpp<<endl;
					if(tpp.x!=0 && tpp.y!=0){
						drawline.push_back(tpp);
						//cout<<drawline.size()<<endl;
					}
				}
				//cout<<clickStatus<<" "<<cStatus<<endl;
				
			}
			lastDefects=defects;
			lastcontours=filterContours;
			drawdefects(defects,filterContours,frame);
		}
		
		
		

		//gui

		namedWindow("calibration",WINDOW_NORMAL);
		namedWindow("recording",WINDOW_AUTOSIZE);

		createTrackbar("sub_type","calibration",&sub_type,1);
		createTrackbar("Blur_type","calibration",&c_type,3);
		createTrackbar("Y_MIN","calibration",&Y_MIN,255);
		createTrackbar("Y_MAX","calibration",&Y_MAX,255);
		createTrackbar("Cr_MIN","calibration",&Cr_MIN,255);
		createTrackbar("Cr_MAX","calibration",&Cr_MAX,255);
		createTrackbar("Cb_MIN","calibration",&Cb_MIN,255);
		createTrackbar("Cb_MAX","calibration",&Cb_MAX,255);


	//	imshow("calibration");


		resizeWindow("calibration",300,600);
		imshow("recording",frame);
		imshow("calibration",mask);
		outputVideo.write(frame);
		
		if(waitKey(30)>0) break;
		}
	}

void drawedge(vector<vector<Point> > contours,Mat &frame){
	//drawContours(frame,contours,-1,Scalar(0,0,255),2);
	for(int i=0;i<contours.size();i++){
		vector<vector<Point> > hull(1);
		convexHull(Mat(contours[i]),hull[0],true);
		//drawContours(frame,hull,-1,Scalar(0,255,0),2);
		minEnclosingCircle( Mat(contours[i]),center,radius);
		//circle(frame,center,radius,Scalar(144,255,144),1,8,0);
		circle(frame,center,4,Scalar(144,255,144),1,8,0);
	
	}
	rectangle(frame,rect1,Scalar(0,255,0),CV_FILLED);
	rectangle(frame,rect2,Scalar(255,0,0),CV_FILLED);
	rectangle(frame,rect3,Scalar(0,0,255),CV_FILLED);
	rectangle(frame,rect4,Scalar(0,0,255));
//	rectangle(frame,rect5,Scalar(0,0,255));
	if(drawline.size()!=0){
		for(int i=0;i<drawline.size()-1;i++){
			line(frame,drawline[i],drawline[i+1],paintColor,2);
		}
	}

	if(reddraw.size()!=0){
		for(int i=0;i<reddraw.size();i++){
			for(int j=0;j<reddraw[i].size()-1;j++){
			line(frame,reddraw[i][j],reddraw[i][j+1],Scalar(0,0,255),2);
			}
		}
	}

	if(greendraw.size()!=0){
		for(int i=0;i<greendraw.size();i++){
			for(int j=0;j<greendraw[i].size()-1;j++){
			line(frame,greendraw[i][j],greendraw[i][j+1],Scalar(0,255,0),2);
			}
		}
	}
	if(bluedraw.size()!=0){
		for(int i=0;i<bluedraw.size();i++){
			for(int j=0;j<bluedraw[i].size()-1;j++){
			line(frame,bluedraw[i][j],bluedraw[i][j+1],Scalar(255,0,0),2);
			}
		}
	}
	
}

vector<vector<Vec4i> > getDefects(vector<vector<Point> > &contours){
	vector<vector<Vec4i> > defectsSet(contours.size());
	for(int i=0;i<contours.size();i++){
		
		vector<vector<int> > hull(1);
		convexHull(Mat(contours[i]),hull[0],false);
		vector<Vec4i> defects;
		vector<Vec4i> defectsFilter;
		if(hull[0].size()>0){
			convexityDefects(contours[i],hull[0],defects);

			if(defects.size()>=3){
			
				for(int j=0;j<defects.size();j++){
					int startid=defects[j].val[0];
					int endid=defects[i].val[1];
					int farid=defects[i].val[2];
					double depth=defects[j].val[3]/256.0;
				
					Point sp(contours[i][startid]);
					Point ep(contours[i][endid]);
					Point fp(contours[i][farid]);
					double angle=calAngle(sp,ep,fp);


					if((fabs(angle)<(5*3.1415/9)) && (depth>=0.4*radius)){
						defectsFilter.push_back(defects[j]);
					}

					if(defectsFilter.size()>0){
						int sidx=defectsFilter.front().val[0];
						int eidx=defectsFilter.front().val[1];
						int fidx=defectsFilter.front().val[2];
						
						Point sp(contours[i][sidx]);
						Point ep(contours[i][eidx]);
						Point fp(contours[i][fidx]);

	
						int sidx2=defectsFilter.back().val[0];
						int eidx2=defectsFilter.back().val[1];
						int fidx2=defectsFilter.back().val[2];
						
						Point sp2(contours[i][sidx2]);
						Point ep2(contours[i][eidx2]);
						Point fp2(contours[i][fidx2]);

						if(calDist(sp,ep2)<radius/6){
							contours[i][sidx]=ep2;
						}
						if(calDist(ep,sp2)<radius/6){
							contours[i][sidx2]=ep;
						}
						
						for(int z=0;z<defectsFilter.size()-1;z++){
							for(int v=z+1;v<defectsFilter.size();v++){
								
								int sidx=defectsFilter[z].val[0];
								int eidx=defectsFilter[z].val[1];
								int fidx=defectsFilter[z].val[2];

								Point sp(contours[i][sidx]);
								Point ep(contours[i][eidx]);
								Point fp(contours[i][fidx]);


								int sidx2=defectsFilter[v].val[0];
								int eidx2=defectsFilter[v].val[1];
								int fidx2=defectsFilter[v].val[2];

								Point sp2(contours[i][sidx2]);
								Point ep2(contours[i][eidx2]);
								Point fp2(contours[i][fidx2]);
								if(calDist(sp,ep2)<radius/6){
									contours[i][sidx]=ep2;
									break;
								}
								if(calDist(ep,sp2)<radius/6){
									contours[i][sidx2]=ep;
									break;
								}
							}
						}
					}
				
				}
			
			}
				//defectsSet.push_back(defectsFilter);
			defectsSet.at(i)=defectsFilter;
		}
	}
	return defectsSet;
}

void drawdefects(vector<vector<Vec4i> > defects,vector<vector<Point> > contours,Mat &frame){
	for(int i=0;i<defects.size();i++){
		//cout<<defects[i].size()<<endl;
		if(defects[i].size()>0){
			for(int j=0;j<defects[i].size();j++){
				int sidx=defects[i][j].val[0];
				int eidx=defects[i][j].val[1];
				int fidx=defects[i][j].val[2];

				Point sp(contours[i][sidx]);
				Point ep(contours[i][eidx]);
				Point fp(contours[i][fidx]);
				//cout<<ep<<" "<<sp<<endl;
				line(frame,sp,center,Scalar(0,100,0));
				line(frame,ep,center,Scalar(0,100,0));
				circle(frame,ep,3,Scalar(100,0,0),2,8,0);
				circle(frame,sp,3,Scalar(0,0,100),2,8,0);
				circle(frame,fp,3,Scalar(0,100,0),2,8,0);
			}
		}
	}
}


Point isClick(vector<vector<Vec4i> > last,vector<vector<Point> > contours, vector<vector<Vec4i> > current){
	for(int i=0;i<last.size();i++){
		if(last[i].size()==1 && current[i].size()==0){
			Point p(contours[i][last[i][0].val[1]]);
			return p;
		}
	}
	Point p(0,0);
	return p;
}


Point getCurrentLocation(vector<vector<Point> > contours, vector<vector<Vec4i> > current){
	for(int i=0;i<current.size();i++){
		if(current[i].size()!=0){
			for(int j=0;j<current[i].size();j++){
				int eidx=current[i][j].val[1];
				Point p(contours[i][eidx]);
				return p;
			}
		}
	}
	Point p(0,0);
	return p;
}


void judgePoint(Point p){
	if(p.x!=0 && p.y!=0){
		if(p.inside(rect1)){
			cout<<"enter 1"<<endl;
			paintColor=Scalar(0,255,0);
			cStatus=1;
			clickStatus=0;
		}else if(p.inside(rect2)){
			cout<<"enter 2"<<endl;
			paintColor=Scalar(255,0,0);
			cStatus=2;
			clickStatus=0;
		}else if(p.inside(rect3)){
			cout<<"enter 3"<<endl;
			paintColor=Scalar(0,0,255);
			cStatus=3;
			clickStatus=0;
		}else if(p.inside(rect4)){
			cout<<"enter 4"<<endl;
			cStatus=0;
			clickStatus=0;
			drawline.clear();
			reddraw.clear();
			bluedraw.clear();
			greendraw.clear();
		}else if(clickStatus==0&&cStatus>=1){
			cout<<"paiting"<<endl;
			clickStatus=1;
		}else if(clickStatus==1 && cStatus>=1){
			cout<<"stop paiting"<<endl;
			clickStatus=0;
			if(cStatus==1){
				if(drawline.size()!=0) greendraw.push_back(drawline);
			}else if(cStatus==2){
				if(drawline.size()!=0) bluedraw.push_back(drawline);
			}else if(cStatus==3){
				if(drawline.size()!=0) reddraw.push_back(drawline);
			}
			drawline.clear();
			
		}
	}
}
