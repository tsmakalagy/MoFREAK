#ifndef BAGOFWORDSREPRESENTATION_H
#define BAGOFWORDSREPRESENTATION_H

#include <opencv2\core\core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <list>
#include <unordered_map>
#include <sstream>
#include <limits>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost::filesystem;

class BagOfWordsRepresentation
{
public:
	BagOfWordsRepresentation(std::vector<std::string> &file_list, int num_clust, int ftr_dim, int num_groups,
		bool appearance_is_bin, bool motion_is_bin, int dset, std::string svm_path);
	BagOfWordsRepresentation(int num_clust, int ftr_dim, std::string svm_path, int num_groups, int dset);

	~BagOfWordsRepresentation();
	void intializeBOWMemory(std::string SVM_PATH);

	void computeBagOfWords(std::string SVM_PATH, std::string MOFREAK_PATH, std::string METADATA_PATH);
	void computeSlidingBagOfWords(std::string &input_file, int alpha, int label, std::ofstream &out);
	void convertFileToBOWFeature(std::string file);
	void writeBOWFeaturesToFiles();

	void setMotionDescriptor(unsigned int size, bool binary = false);
	void setAppearanceDescriptor(unsigned int size, bool binary = false);

private:
	void computeHMDB51BagOfWords(std::string SVM_PATH, std::string MOFREAK_PATH, std::string METADATA_PATH);
	void extractMetadata(std::string filename, int &action, int &group, int &clip_number);
	void loadClusters();

	cv::Mat buildHistogram(std::string &file, bool &success);
	int actionStringToActionInt(std::string act);

	unsigned int hammingDistance(unsigned char a, unsigned char b);
	unsigned int hammingDistance(cv::Mat &a, cv::Mat &b);
	int bruteForceMatch(cv::Mat &feature);

	std::vector<std::string> split(const std::string &s, char delim);
	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

	std::vector<std::string> files;
	cv::BFMatcher *bf_matcher;
	const unsigned int NUMBER_OF_CLUSTERS;
	const unsigned int FEATURE_DIMENSIONALITY;
	const unsigned int NUMBER_OF_GROUPS;
	const std::string SVM_PATH;

	cv::Mat *clusters;
	std::vector<cv::Mat> clusters_for_matching;

	//enum WEIZMANN_action {bend, jack, jump, pjump, run, side, skip, walk, wave1, wave2};
	enum HMDB_action {BRUSH_HAIR, CARTWHEEL, CATCH, CHEW, CLAP, CLIMB, CLIMB_STAIRS,
		DIVE, DRAW_SWORD, DRIBBLE, DRINK, EAT, FALL_FLOOR, FENCING, FLIC_FLAC, GOLF,
		HANDSTAND, HIT, HUG, JUMP, KICK, KICK_BALL, KISS, LAUGH, PICK, POUR, PULLUP,
		PUNCH, PUSH, PUSHUP, RIDE_BIKE, RIDE_HORSE, RUN, SHAKE_HANDS, SHOOT_BALL,
		SHOOT_BOW, SHOOT_GUN, SIT, SITUP, SMILE, SMOKE, SOMERSAULT, STAND,
		SWING_BASEBALL, SWORD, SWORD_EXERCISE, TALK, THROW, TURN, WALK, WAVE};

	unsigned int motion_descriptor_size;
	unsigned int appearance_descriptor_size;
	bool motion_is_binary;
	bool appearance_is_binary;
	std::unordered_map<std::string, int> actions;

	// BOW features for SVM input
	std::vector<std::vector<std::string> > bow_training_crossvalidation_sets;
	std::vector<std::vector<std::string> > bow_testing_crossvalidation_sets;

	// files to print SVM features to
	std::vector<std::ofstream *> training_files;
	std::vector<std::ofstream *> testing_files;

	int dataset;
	enum datasets {KTH, TRECVID, HOLLYWOOD, UTI1, UTI2, HMDB51, UCF101};
};
#endif