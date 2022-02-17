class DayNightClassifier(object):
    def __init__(self, 
                 day_value = 190.0,
                 day_sd = 190.0,
                 night_value = 190.0,
                 night_sd = 190.0):
        """
        Initialize classifier to a set of initial avg values in HSV space
        The parameters/features gets updated as the classifier sees more images
        
        """
        self.day_value = day_value
        self.day_sd = day_sd
        self.night_value = night_value
        self.night_sd = night_sd
    def _read_image(self,str_):
        """
        
        Reads comma separated string as a numpy array
        Input:
        str_ - comma separated string 
        
        Returns:
        input_image - RGB image in HWC format
        
        """
        input_image = np.fromstring(str_,dtype = int, sep= ',').reshape(-1,3)
        dim_1 = input_image.shape[0]
        l = b = int((dim_1)**(1/2))
        input_image = input_image.reshape(l, b, 3)
        return input_image
    
    def _rgb_to_hsv(self, input_array):
        """
        
        Converts input numpy array from RGB to HSV space
        Input:
        input_array - Numpy array in HWC format
        
        Returns:
        Image in HSV format
        
        """
        return cv2.cvtColor(input_array, cv2.COLOR_RGB2HSV)
    
    def _no_conv_features(self, input_array):
        """
        
        Extracts necessary HSC features from input array
        Input:
        input_array - Numpy array in HWC format
        
        Returns:
        List with average value and standard deviation of Value in HSV space
        
        """
        hsv_array = self._rgb_to_hsv(input_array)
        h,s,v = cv2.split(hsv_array)
        avg_hue = np.sum(h)/(input_array.shape[0]*input_array.shape[1])
        SD_hue = np.std(h)
        avg_value = np.sum(v)/(input_array.shape[0]*input_array.shape[1])
        SD_value = np.std(v)
        return [avg_value,SD_value]
    
    def classifier(self , str_):
        """
        Classifies an input array as DAY or NIGHT
        Input:
        str_ - comma separated string 
        
        Returns:
        List with ['DAY'] or ['NIGHT'] as classified output
        
        
        """
        input_array =self._read_image(str_)        
        cur_avg , cur_sd = self._no_conv_features(input_array)
        
        if (self.day_value-self.day_sd) <cur_avg <= (self.day_value+ self.day_sd):
            self.day_value = (self.day_value+ cur_avg)/2
            self.day_sd = (self.day_sd+ cur_sd)/2
            return ['DAY']
        elif (self.night_value-self.night_sd) <cur_avg <= (self.night_value+ self.night_sd):
            self.night_value = (self.night_value+cur_avg)/2
            self.night_sd = (self.night_sd+cur_sd)/2            
            return ['NIGHT']
        else:
            # Make a random choice for difficult ones
            return [np.random.choice(['DAY','NIGHT'])]
            
            
            
        
