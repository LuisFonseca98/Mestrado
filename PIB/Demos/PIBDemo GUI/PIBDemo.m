function varargout = PIBDemo(varargin)
% PIBDEMO MATLAB code for PIBDemo.fig
%      PIBDEMO,edit by itself, creates a new PIBDEMO or raises the existing
%      singleton*.
%
%      H = PIBDEMO returns the handle to a new PIBDEMO or the handle to
%      the existing singleton*.
%
%      PIBDEMO('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PIBDEMO.M with the given input arguments.
%
%      PIBDEMO('Property','Value',...) creates a new PIBDEMO or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before PIBDemo_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to PIBDemo_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help PIBDemo

% Last Modified by GUIDE v2.5 12-Mar-2018 19:13:45

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @PIBDemo_OpeningFcn, ...
                   'gui_OutputFcn',  @PIBDemo_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before PIBDemo is made visible.
function PIBDemo_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to PIBDemo (see VARARGIN)

% Choose default command line output for PIBDemo
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes PIBDemo wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = PIBDemo_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('bird.gif');
axes(handles.axes1);
imshow(I); 
m = mean(mean(I));
title( sprintf('Average Value =%.1f', m) );


axes(handles.axes2);
imhist(I); title('Histogram');


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I   = imread('bird.gif');
lut = uint8(255 : -1 : 0);
I1  = intlut(I, lut);
axes(handles.axes3);
imshow(I1);
m = mean(mean(I1));
title( sprintf('Average Value =%.1f', m) );

axes(handles.axes4);
imhist(I1); title('Histogram');
