/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include <Eigen/Core>
#include <string>
#include <GL/glut.h> 
namespace ORB_SLAM3
{


MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }
}

void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();
    mKeyFrameLineWidth = settings->keyFrameLineWidth();
    mGraphLineWidth = settings->graphLineWidth();
    mPointSize = settings->pointSize();
    mCameraSize = settings->cameraSize();
    mCameraLineWidth  = settings->cameraLineWidth();
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void RenderText(const std::string& text, float x, float y, float scale = 1.0f)
{
    // Defina a cor do texto (vermelho, por exemplo)
    glColor3f(0.0f, 0.0f, 0.0f);

    // Posiciona o texto na tela
    glRasterPos2f(x, y);

    // Renderiza cada caractere
    for (const char &c : text) {
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, c);
    }
}
bool glut_initialized = false;
void InitializeGLUT() {
    int argc = 0;
    if (!glut_initialized) {
        glutInit(&argc, NULL);
        glut_initialized = true;
    }
}
Eigen::Vector3f CalculateCentroid(const std::vector<Eigen::Vector3f>& points) {
    Eigen::Vector3f centroid(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < points.size(); i++) {
        centroid += points[i];
    }
    centroid /= points.size();
    //cout << "centroid=" << centroid<<endl;
    return centroid;
}

float FindMaxDistance(const Eigen::Vector3f& centroid, const std::vector<Eigen::Vector3f>& points) {
    float max_distance = 0.0f;
    //cout << "points.size()" << points.size()<<endl;
    for (int i = 0; i < points.size(); i++) {
        float distance = (points[i] - centroid).norm(); 
        if (distance > max_distance) {
            max_distance = distance;
        }
    }
    //cout << "max dist=" << max_distance<<endl;
    return max_distance;
}

std::vector<Eigen::Vector3f> MapDrawer::GetClosestPointsToMapCenter() {
    // Get all map points from the current active map
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if (!pActiveMap)
        return {};

    const vector<MapPoint*>& vpMPs = pActiveMap->GetAllMapPoints();
    if (vpMPs.empty())
        return {};

    // Vector to store valid points
    std::vector<Eigen::Vector3f> validPoints;

    // Collect all valid points
    for (const auto& mp : vpMPs) {
        if (mp && !mp->isBad()) {
            Eigen::Vector3f pos = mp->GetWorldPos();
            if (pos.array().isFinite().all()) { // Ensure the point is finite
                validPoints.push_back(pos);
            }
        }
    }

    // Check if there are enough valid points
    if (validPoints.size() < 5) {
        std::cerr << "Not enough valid points to calculate." << std::endl;
        return {};
    }

    // Calculate the centroid of all valid points (this is the map center)
    Eigen::Vector3f mapCentroid = CalculateCentroid(validPoints);

    // Calculate the distances from each point to the map center
    std::vector<std::pair<float, Eigen::Vector3f>> distances;
    for (const auto& pos : validPoints) {
        float distance = (pos - mapCentroid).norm();
        distances.push_back(std::make_pair(distance, pos));
    }

    // Sort the points by their distance to the map center
    std::sort(distances.begin(), distances.end(),
        [](const std::pair<float, Eigen::Vector3f>& a, const std::pair<float, Eigen::Vector3f>& b) {
            return a.first < b.first;
        });

    // Select the 5 closest points to the map center
    std::vector<Eigen::Vector3f> closestPoints;
    for (int i = 0; i < 5; ++i) {
        closestPoints.push_back(distances[i].second);
    }

    return closestPoints;
}

void MapDrawer::DrawCubeAroundPoints(const std::vector<Eigen::Vector3f>& points, std::string classID) {
    
    if (points.empty()) {
        return;
    }
    Eigen::Vector3f minPoint,maxPoint;
    minPoint = points[0];
    maxPoint = points[0];
    for (const auto& point : points) {
        //cout <<"pos= "<<point.x()<<", "<< point.y() << ", "<< point.z()<< endl;
        minPoint = minPoint.cwiseMin(point);
        maxPoint = maxPoint.cwiseMax(point);
    }
 //   cout << "minPoint= " << minPoint << " maxPoint="<< maxPoint <<endl;
    float padding = 0.0;
    Eigen::Vector3f minPadded = minPoint - Eigen::Vector3f(padding, padding, padding);
    Eigen::Vector3f maxPadded = maxPoint + Eigen::Vector3f(padding, padding, padding);
    Eigen::Vector3f center = (minPadded + maxPadded) / 2.0f;
    Eigen::Vector3f scale = (maxPadded - minPadded) / 2.0f;
    // // define each vertice
    Eigen::Vector3f vertices[8];
    vertices[0] = minPadded;
    vertices[1] = Eigen::Vector3f(maxPadded.x(), minPadded.y(), minPadded.z());
    vertices[2] = Eigen::Vector3f(maxPadded.x(), maxPadded.y(), minPadded.z());
    vertices[3] = Eigen::Vector3f(minPadded.x(), maxPadded.y(), minPadded.z());
    vertices[4] = Eigen::Vector3f(minPadded.x(), minPadded.y(), maxPadded.z());
    vertices[5] = Eigen::Vector3f(maxPadded.x(), minPadded.y(), maxPadded.z());
    vertices[6] = maxPadded;
    vertices[7] = Eigen::Vector3f(minPadded.x(), maxPadded.y(), maxPadded.z());
    //draw each line 
    glPushMatrix();
    glBegin(GL_LINES);
    //botton
    glVertex3f(vertices[0].x(), vertices[0].y(), vertices[0].z());
    glVertex3f(vertices[1].x(), vertices[1].y(), vertices[1].z());

    glVertex3f(vertices[1].x(), vertices[1].y(), vertices[1].z());
    glVertex3f(vertices[2].x(), vertices[2].y(), vertices[2].z());

    glVertex3f(vertices[2].x(), vertices[2].y(), vertices[2].z());
    glVertex3f(vertices[3].x(), vertices[3].y(), vertices[3].z());

    glVertex3f(vertices[3].x(), vertices[3].y(), vertices[3].z());
    glVertex3f(vertices[0].x(), vertices[0].y(), vertices[0].z());
    //top 
    glVertex3f(vertices[4].x(), vertices[4].y(), vertices[4].z());
    glVertex3f(vertices[5].x(), vertices[5].y(), vertices[5].z());

    glVertex3f(vertices[5].x(), vertices[5].y(), vertices[5].z());
    glVertex3f(vertices[6].x(), vertices[6].y(), vertices[6].z());

    glVertex3f(vertices[6].x(), vertices[6].y(), vertices[6].z());
    glVertex3f(vertices[7].x(), vertices[7].y(), vertices[7].z());

    glVertex3f(vertices[7].x(), vertices[7].y(), vertices[7].z());
    glVertex3f(vertices[4].x(), vertices[4].y(), vertices[4].z());
    //top to down 
    glVertex3f(vertices[0].x(), vertices[0].y(), vertices[0].z());
    glVertex3f(vertices[4].x(), vertices[4].y(), vertices[4].z());

    glVertex3f(vertices[1].x(), vertices[1].y(), vertices[1].z());
    glVertex3f(vertices[5].x(), vertices[5].y(), vertices[5].z());

    glVertex3f(vertices[2].x(), vertices[2].y(), vertices[2].z());
    glVertex3f(vertices[6].x(), vertices[6].y(), vertices[6].z());

    glVertex3f(vertices[3].x(), vertices[3].y(), vertices[3].z());
    glVertex3f(vertices[7].x(), vertices[7].y(), vertices[7].z());
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    //glTranslatef(center[0], center[1], center[2]);
    //glScalef(scale[0], scale[1], scale[2]);
    //pangolin::glDrawColouredCube();
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnd();
    glPopMatrix();
    InitializeGLUT();
    std::string text = "ClassID: " + classID;
    RenderText(text, vertices[0].x(), vertices[0].y());
}

void MapDrawer::DrawRegion() {
    // Get the 5 points closest to the map center
    std::vector<Eigen::Vector3f> closestPoints = GetClosestPointsToMapCenter();    
    // Draw the cube around those points
    DrawCubeAroundPoints(closestPoints, "teeste");
}

void MapDrawer::DrawObject(const YoloDetect::Object& object) {
    // Vector to store valid points
    std::vector<Eigen::Vector3f> validPoints;
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    // Collect all valid points
    for (const auto& mp : object.mapPoints)
    {
        // Ensure the map point is valid and not bad
        if (mp && !mp->isBad())
        {
            Eigen::Vector3f pos = mp->GetWorldPos();
            
            // Ensure the position is finite
            if (pos.array().isFinite().all())
            {
                validPoints.push_back(pos);
                glVertex3f(pos(0),pos(1),pos(2));
                //cout<<"Draw"<<endl;
            }
        }
    }
    // here we have the object area, the camera pose, the max and min deph
//     float depthMin = object.depthMinMax.first;
//     float depthMax = object.depthMinMax.second;
//     float depthAvg = (depthMin + depthMax) / 2.0f;
//     float fx=477.22575539032283;
//     float fy=477.22575539032283;
//     // float fx=958.2243041992188;
//     // float fy=958.2243041992188;
//     // float cx=640.9063110351562;
//     // float cy=350.24981689453125;
//     float cx=270.5;
//     float cy=270.5;
//     float scaleFactorX = depthAvg / fx;
//     float scaleFactorY = depthAvg / fy;
//     float cubeWidth = object.area.width * scaleFactorX;
//     float cubeHeight = object.area.height * scaleFactorY;
//     float cubeDepth = depthMax - depthMin;
//     float centerX = object.area.x + object.area.width / 2.0f;
//     float centerY = object.area.y + object.area.height / 2.0f;
//     Eigen::Vector3f centerCam(centerX * scaleFactorX, centerY * scaleFactorY, depthAvg);
//     //get the cube coordinates related to the world. 
//     // Calculate the 3D position of the cube center
//     float X = (centerX - cx) * depthAvg / fx;
//     float Y = (centerY - cy) * depthAvg / fy;
//     Eigen::Vector3f cubeCenter(X, Y, depthAvg);
//     Eigen::Vector4f centerCamHomog(centerCam(0), centerCam(1), centerCam(2), 1.0f);
//     Eigen::Vector4f centerWorldHomog = mCameraPose * centerCamHomog;
//     //Eigen::Vector3f centerWorld = centerWorldHomog.head<3>();
//     Eigen::Vector3f centerWorld = cubeCenter;
//     std::vector<Eigen::Vector3f> cubePoints = {
//         Eigen::Vector3f(centerWorld(0) - cubeWidth / 2, centerWorld(1) - cubeHeight / 2, centerWorld(2) - cubeDepth / 2), // Point 0
//         Eigen::Vector3f(centerWorld(0) + cubeWidth / 2, centerWorld(1) - cubeHeight / 2, centerWorld(2) - cubeDepth / 2), // Point 1
//         Eigen::Vector3f(centerWorld(0) + cubeWidth / 2, centerWorld(1) + cubeHeight / 2, centerWorld(2) - cubeDepth / 2), // Point 2
//         Eigen::Vector3f(centerWorld(0) - cubeWidth / 2, centerWorld(1) + cubeHeight / 2, centerWorld(2) - cubeDepth / 2), // Point 3
//         Eigen::Vector3f(centerWorld(0) - cubeWidth / 2, centerWorld(1) - cubeHeight / 2, centerWorld(2) + cubeDepth / 2), // Point 4
//         Eigen::Vector3f(centerWorld(0) + cubeWidth / 2, centerWorld(1) - cubeHeight / 2, centerWorld(2) + cubeDepth / 2), // Point 5
//         Eigen::Vector3f(centerWorld(0) + cubeWidth / 2, centerWorld(1) + cubeHeight / 2, centerWorld(2) + cubeDepth / 2), // Point 6
//         Eigen::Vector3f(centerWorld(0) - cubeWidth / 2, centerWorld(1) + cubeHeight / 2, centerWorld(2) + cubeDepth / 2)  // Point 7
//     };

// //    glEnd();
     std::string classID = object.classID;
//     //cout <<"Draw"<<endl;
//     // Draw a cube around the valid points
    DrawCubeAroundPoints(validPoints,classID);
}

void MapDrawer::DrawMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);
    Eigen::Matrix<float,3,1> pos;
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = (*sit)->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs;

    if(!pActiveMap)
        return;

    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();

            glMultMatrixf((GLfloat*)Twc.data());

            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);
                glColor3f(1.0f,0.0f,0.0f);
                glBegin(GL_LINES);
            }
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);
                if (bDrawOptLba) {
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs
                    }
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color
                    }
                }
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color
                }
                glBegin(GL_LINES);
            }

            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();

            glEnd();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }

    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    if(bDrawKF)
    {
        for(Map* pMap : vpMaps)
        {
            if(pMap == pActiveMap)
                continue;

            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}

void MapDrawer::DrawObjectMapPoints(const YoloDetect::Object& object)
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;
    std::vector<Eigen::Vector3f> validPoints;
    // Define points
    const vector<MapPoint*> &vpObjectMPs = pActiveMap->GetAllObjectMapPoints();
    glPointSize(5);
    glBegin(GL_POINTS);
    glColor3f(0.0, 1.0, 0.0);
   // All map points
    for (std::vector<MapPoint *>::const_iterator i = vpObjectMPs.begin(); i != vpObjectMPs.end(); i++)
    {
        if ((*i)->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = (*i)->GetWorldPos();
        glVertex3f(pos(0), pos(1), pos(2));
        validPoints.push_back(pos);
    }
    glEnd();
    std::string classID = object.classID;
    // Draw a cube around the valid points
    DrawCubeAroundPoints(validPoints,classID);

}

void MapDrawer::DrawObjectMapPoints(int index, std::string classID)
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;
    std::vector<Eigen::Vector3f> validPoints;
    // Define points
    const vector<MapPoint*> &vpObjectMPs = pActiveMap->GetObjectMapPoints(index);
    glPointSize(5);
    glBegin(GL_POINTS);
    glColor3f(0.0, 1.0, 0.0);
    // All map points
    for (std::vector<MapPoint *>::const_iterator i = vpObjectMPs.begin(); i != vpObjectMPs.end(); i++)
    {
        if ((*i)->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = (*i)->GetWorldPos();
        glVertex3f(pos(0), pos(1), pos(2));
        validPoints.push_back(pos);
    }
    glEnd();
    // Draw a cube around the valid points
    DrawCubeAroundPoints(validPoints,classID);

}

} //namespace ORB_SLAM
