# DermaVision AI - Frontend Features

## 🎨 Brand Identity
**Name:** DermaVision AI  
**Tagline:** Advanced AI-powered skin disease analysis and diagnosis platform

## ✨ Key Features

### 🔐 Authentication System
- **Login Page**: Secure authentication with email/password
- **Registration**: New user signup with validation
- **Demo Account**: Quick access with pre-filled credentials
- **Protected Routes**: Automatic redirection for unauthenticated users
- **Session Management**: Persistent login using Zustand + localStorage

### 📊 Interactive Dashboard
- **Real-time Statistics**: Total analyses, today's count, average confidence, active models
- **Analytics Charts**: 
  - Line chart showing analysis trends over time
  - Pie chart displaying top diagnosed conditions
- **Recent Analyses**: Quick access to latest diagnosis results
- **Quick Actions**: One-click navigation to analyze new images

### 🔬 Advanced Analysis Page
- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **Batch Processing**: Upload and analyze multiple images simultaneously
- **Real-time Preview**: See uploaded images before analysis
- **Progress Indicators**: Loading states and animated transitions
- **Detailed Results**: 
  - Primary diagnosis with confidence score
  - Alternative predictions with percentages
  - Medical disclaimer
- **Export to PDF**: Generate downloadable reports
- **Share Results**: (Coming soon)

### 🎯 User Experience
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Dark Mode Support**: Toggle between light and dark themes
- **Smooth Animations**: Framer Motion powered transitions
- **Toast Notifications**: Real-time feedback for all actions
- **Modern UI**: Tailwind CSS with custom color palette
- **Icon System**: Lucide React icons throughout

### 🏗️ Architecture
- **State Management**: Zustand for global state (auth, theme, analyses)
- **API Integration**: Axios with interceptors for token management
- **Routing**: React Router v6 with nested routes
- **Component Structure**: 
  - `pages/`: Login, Register, Dashboard, Analyze, History, Profile
  - `components/`: Layout with sidebar navigation
  - `services/`: API service layer
  - `store/`: Zustand stores

### 📦 Technology Stack
- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS with custom theme
- **Charts**: Recharts for data visualization
- **File Upload**: React Dropzone
- **Animations**: Framer Motion
- **PDF Export**: jsPDF + html2canvas
- **Icons**: Lucide React
- **Notifications**: React Hot Toast
- **Date Handling**: date-fns

## 🚀 Getting Started

### Install Dependencies
```bash
npm install
```

### Run Development Server
```bash
npm run dev
```

The app will be available at http://localhost:5173

### Build for Production
```bash
npm run build
```

## 🎨 Color Palette
- **Primary**: Purple gradient (#667eea to #764ba2)
- **Secondary**: Purple (#a855f7)
- **Success**: Green (#10b981)
- **Error**: Red (#ef4444)
- **Background**: Light gray (#f9fafb)

## 📱 Pages Overview

### `/login`
- Email/password authentication
- Demo account button
- Link to registration

### `/register`
- Full name, email, password fields
- Password confirmation
- Input validation

### `/dashboard`
- Statistics cards
- Analytics charts
- Recent analyses list
- Quick action button

### `/analyze`
- Drag & drop upload zone
- Multi-image support
- Batch analysis
- Results display with export

### `/history` (Coming Soon)
- Complete analysis history
- Search and filter
- Detailed view

### `/profile` (Coming Soon)
- User information
- Settings
- Preferences

## 🔒 Authentication Flow
1. User visits app → Redirected to `/login`
2. Login successful → Token stored → Navigate to `/dashboard`
3. Subsequent visits → Auto-login from stored token
4. Logout → Clear token → Redirect to `/login`

## 🎯 Demo Credentials
```
Email: demo@dermavision.ai
Password: demo123
```

## 📝 Notes
- The backend API endpoints are defined but may need implementation
- Some features (History, Profile) have placeholder pages
- The app uses mock data for charts when API is unavailable
- All routes except login/register require authentication
