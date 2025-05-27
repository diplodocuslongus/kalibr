#include <vector>
#include <opencv2/core/core.hpp>
#include <aslam/cameras/GridCalibrationTargetObservation.hpp>

namespace aslam {
namespace cameras {

GridCalibrationTargetObservation::GridCalibrationTargetObservation(
    GridCalibrationTargetBase::Ptr target, cv::Mat image)
    : _imRows(0),
      _imCols(0),
      _T_t_c_isSet(false)  {
  setImage(image);
  setTarget(target);
}

GridCalibrationTargetObservation::GridCalibrationTargetObservation(
    GridCalibrationTargetBase::Ptr target)
    : _imRows(0),
      _imCols(0),
      _T_t_c_isSet(false)  {
  setTarget(target);
}

GridCalibrationTargetObservation::~GridCalibrationTargetObservation() {
}

void GridCalibrationTargetObservation::setTarget(
    GridCalibrationTargetBase::Ptr target) {
  SM_ASSERT_TRUE(Exception, target.get() != NULL,
                 "Refusing to set a null target");
  _target = target;
  _points = Eigen::MatrixXd::Zero(target->size(), 2);
  _success.clear();
  _success.resize(target->size(), false);
}

/// \brief get all (observed) corners in image coordinates
unsigned int GridCalibrationTargetObservation::getCornersTargetFrame(
    std::vector<cv::Point3f> &outCornerList) const {
  SM_ASSERT_TRUE(Exception, _target.get() != NULL, "The target is not set");

  //max. number of corner in the grid
  unsigned int numCorners = _target->size();

  // output the points
  unsigned int cntCorners = 0;
  outCornerList.clear();
  for (unsigned int i = 0; i < numCorners; i++) {
    // add only if the gridpoint which were observed in the image
    if (_success[i]) {
      //convert to cv:Point2f and store
      cv::Point3f corner(_target->point(i)[0], _target->point(i)[1], 0.0);
      outCornerList.push_back(corner);

      //count the observed corners
      cntCorners += 1;
    }
  }
  return cntCorners;
}

unsigned int GridCalibrationTargetObservation::getAllCornersTargetFrame(
    Eigen::Matrix<double, -1, 3> &outCornerList) const {
  SM_ASSERT_TRUE(Exception, _target.get() != NULL, "The target is not set");
  unsigned int numCorners = _target->size();
  outCornerList.resize(numCorners, 3);
  for (unsigned int i = 0; i < numCorners; i++) {
    outCornerList(i, 0) = _target->point(i)[0];
    outCornerList(i, 1) = _target->point(i)[1];
    outCornerList(i, 2) = 0.0;
  }
  return numCorners;
}

/// \brief get all (observed) corners in target frame coordinates
///        returns the number of observed corners
unsigned int GridCalibrationTargetObservation::getCornersImageFrame(
    std::vector<cv::Point2f> &outCornerList) const {
  SM_ASSERT_TRUE(Exception, _target.get() != NULL, "The target is not set");

  //max. number of corner in the grid
  unsigned int numCorners = _target->size();

  // output the points
  unsigned int cntCorners = 0;
  outCornerList.clear();
  for (unsigned int i = 0; i < numCorners; i++) {
    // add only if the gridpoint was observed in the image
    if (_success[i]) {
      //get points
      Eigen::Vector2d cornerEigen;
      imagePoint(i, cornerEigen);

      //convert to cv:Point2f and store
      cv::Point2f corner(cornerEigen(0), cornerEigen(1));
      outCornerList.push_back(corner);

      //count the observed corners
      cntCorners += 1;
    }
  }

  return cntCorners;
}

/// \brief get the the reprojected corners of all observed corners
///        returns the number of observed corners
unsigned int GridCalibrationTargetObservation::getCornerReprojection(const boost::shared_ptr<CameraGeometryBase> cameraGeometry,
                                                                     std::vector<cv::Point2f> &outPointReproj) const
{
  //check if transformation has been calculated
  SM_ASSERT_TRUE(Exception, _T_t_c_isSet, "Extrinsics not set. Reprojection can only be calculated if the extrinsics have been calculated. Use findTarget()...");

  std::vector<cv::Point3f> targetPoints;
  unsigned int numCorners = getCornersTargetFrame(targetPoints);
  Eigen::Vector2d cornersReproj;

  unsigned int cntCorners = 0;
  outPointReproj.clear();

  for (unsigned int i = 0; i < numCorners; i++) {
    //reproject
    Eigen::Vector3d p(targetPoints[i].x, targetPoints[i].y, targetPoints[i].z);
    Eigen::VectorXd pReproj = T_t_c().inverse() * p;
    Eigen::VectorXd pReprojImg;

    //handle camera geometry
    cameraGeometry->vsEuclideanToKeypoint(pReproj, pReprojImg);

    //store points
    cv::Point2f cornerReproj(pReprojImg(0), pReprojImg(1));
    outPointReproj.push_back(cornerReproj);

    //count the corners
    cntCorners += 1;
  }

  return cntCorners;
}

bool KannalaBrandtProject(const Eigen::Vector3d &p3d, const Eigen::MatrixXd &allp, 
                          Eigen::VectorXd &proj) {
  // https://github.com/berndpfrommer/basalt-headers/blob/robustness_to_bad_data/include/basalt/camera/kannala_brandt_camera4.hpp
  typedef double Scalar;
  const Scalar& fx = allp(0, 0);
  const Scalar& fy = allp(1, 0);
  const Scalar& cx = allp(2, 0);
  const Scalar& cy = allp(3, 0);
  const Scalar& k1 = allp(4, 0);
  const Scalar& k2 = allp(5, 0);
  const Scalar& k3 = allp(6, 0);
  const Scalar& k4 = allp(7, 0);

  const Scalar& x = p3d[0];
  const Scalar& y = p3d[1];
  const Scalar& z = p3d[2];

  const Scalar r2 = x * x + y * y;
  const Scalar r = sqrt(r2);
  const Scalar epsSqrt = 1e-5;
  proj.resize(2, 1);
  if (r > epsSqrt) {
    const Scalar theta = atan2(r, z);
    const Scalar theta2 = theta * theta;

    Scalar r_theta = k4 * theta2;
    r_theta += k3;
    r_theta *= theta2;
    r_theta += k2;
    r_theta *= theta2;
    r_theta += k1;
    r_theta *= theta2;
    r_theta += 1;
    r_theta *= theta;

    const Scalar mx = x * r_theta / r;
    const Scalar my = y * r_theta / r;

    proj[0] = fx * mx + cx;
    proj[1] = fy * my + cy;
  } else {
    // Check that the point is not cloze to zero norm
    if (z < epsSqrt) return false;

    proj[0] = fx * x / z + cx;
    proj[1] = fy * y / z + cy;
  }
  return true;  
}

bool GridCalibrationTargetObservation::projectATargetPoint(const boost::shared_ptr<CameraGeometryBase> cameraGeometry,
                                                                   const sm::kinematics::Transformation & T_t_c,
                                                                   const size_t i, cv::Point2f &outPointReproj, bool kb) const
{    
  cv::Point3f targetPoint(_target->point(i)[0], _target->point(i)[1], 0.0);
  
  Eigen::Vector2d cornersReproj;

  //reproject
  Eigen::Vector3d p(targetPoint.x, targetPoint.y, targetPoint.z);
  Eigen::Vector3d pC = T_t_c.inverse() * p;
  Eigen::VectorXd pReprojImg;

  //handle camera geometry
  bool isValid = false;
  if (kb) {
    Eigen::MatrixXd allp;
    cameraGeometry->getParameters(allp, true, true, true);
    isValid  = KannalaBrandtProject(pC, allp, pReprojImg);
  } else {
    isValid = cameraGeometry->vsEuclideanToKeypoint(pC, pReprojImg);
  }

  //store points
  outPointReproj.x = pReprojImg(0);
  outPointReproj.y = pReprojImg(1);    
  return isValid;
}

unsigned int GridCalibrationTargetObservation::getTotalTargetPoint() const
{
  //max. number of corner in the grid
  return _target->size();
}

/// \brief get the point index of all (observed) corners (order corresponds to the output of getCornersImageFrame and getCornersTargetFrame)
///        returns the number of observed corners
unsigned int GridCalibrationTargetObservation::getCornersIdx(
    std::vector<unsigned int> &outCornerIdx) const {
  SM_ASSERT_TRUE(Exception, _target.get() != NULL, "The target is not set");

  //max. number of corner in the grid
  unsigned int numCorners = _target->size();

  // output the idx's
  outCornerIdx.clear();
  for (unsigned int i = 0; i < numCorners; i++)
    if (_success[i])
      outCornerIdx.push_back(i);

  return outCornerIdx.size();
}

/// \brief update an image observation
void GridCalibrationTargetObservation::updateImagePoint(
    size_t i, const Eigen::Vector2d & point) {
  SM_ASSERT_TRUE(Exception, _target.get() != NULL, "The target is not set");
  SM_ASSERT_LT(
      Exception, i, _success.size(),
      "Index out of bounds. The list has " << _success.size() << " points");
  //SM_DEBUG_STREAM("Updating point " << i << ". Point matrix: (" << _points.rows() << ", " << _points.cols() << ")");
  _points.row(i) = point;
  _success[i] = true;
}

/// \brief remove an image observation
void GridCalibrationTargetObservation::removeImagePoint(size_t i) {
  SM_ASSERT_TRUE(Exception, _target.get() != NULL, "The target is not set");
  SM_ASSERT_LT(
      Exception, i, _success.size(),
      "Index out of bounds. The list has " << _success.size() << " points");
  _success[i] = false;
}

bool GridCalibrationTargetObservation::imagePoint(
    size_t i, Eigen::Vector2d & outPoint) const {
  SM_ASSERT_TRUE(Exception, _target.get() != NULL, "The target is not set");
  SM_ASSERT_LT(
      Exception, i, _success.size(),
      "Index out of bounds. The list has " << _success.size() << " points");
  outPoint = _points.row(i);
  return _success[i];
}

/// \brief get a point from the target expressed in the target frame
/// \return true if the grid point was seen in this image.
bool GridCalibrationTargetObservation::imageGridPoint(
    size_t r, size_t c, Eigen::Vector2d & outPoint) const {
  SM_ASSERT_TRUE(Exception, _target.get() != NULL, "The target is not set");
  return imagePoint(_target->gridCoordinatesToPoint(r, c), outPoint);
}

void GridCalibrationTargetObservation::setImage(cv::Mat image) {
  _image = image;
  _imRows = image.rows;
  _imCols = image.cols;
}
void GridCalibrationTargetObservation::clearImage() {
  _image = cv::Mat();
}

/// \brief return true if the class has at least one successful observation
bool GridCalibrationTargetObservation::hasSuccessfulObservation() const {
  for (unsigned int i=0; i < _success.size(); i++) {
    if (_success[i]) {
      return true;
    }
  }
  return false;
}

}  // namespace cameras
}  // namespace aslam

//export explicit instantions for all included archives
#include <sm/boost/serialization.hpp>
#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT_IMPLEMENT(aslam::cameras::GridCalibrationTargetObservation);

