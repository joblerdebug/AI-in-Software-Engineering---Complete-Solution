"""
Task 3: Predictive Analytics for Resource Allocation
Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json


class PredictiveModel:
    """Predictive model for resource allocation priority"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.performance_metrics = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess the breast cancer dataset"""
        print("📊 Loading and preprocessing data...")
        
        # Load dataset
        cancer_data = load_breast_cancer()
        df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
        df['target'] = cancer_data.target
        
        # Create priority levels based on tumor characteristics
        def create_priority_level(row):
            worst_radius = row['worst radius']
            worst_concave = row['worst concave points']
            
            if worst_radius > 18 and worst_concave > 0.08:
                return 'high'
            elif worst_radius > 15 or worst_concave > 0.05:
                return 'medium'
            else:
                return 'low'
        
        df['priority'] = df.apply(create_priority_level, axis=1)
        
        # Select features for modeling
        self.feature_names = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst concave points', 'worst symmetry'
        ]
        
        X = df[self.feature_names]
        y = df['priority']
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Priority distribution:\n{df['priority'].value_counts()}")
        print(f"Features used: {len(self.feature_names)}")
        
        return X, y_encoded, df
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        print("🤖 Training Random Forest model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Evaluate model
        self.performance_metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
        
        return X_test_scaled, y_test, y_pred, X_train.shape[0], X_test.shape[0]
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """Evaluate model performance"""
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Cross-validation scores
        X_full = self.scaler.transform(self.model.feature_names_in_)
        cv_scores = cross_val_score(self.model, X_full, self.label_encoder.transform(
            pd.Series(self.model.classes_[y_true])), cv=5
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'cross_val_mean': cv_scores.mean(),
            'cross_val_std': cv_scores.std(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def visualize_results(self, df, y_test, y_pred):
        """Create comprehensive visualizations"""
        print("📈 Generating visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_, 
                   yticklabels=self.label_encoder.classes_, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Priority')
        ax1.set_ylabel('Actual Priority')
        
        # 2. Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        ax2.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
        ax2.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Importance Score')
        
        # 3. Priority Distribution
        df['priority'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax3, colors=['#ff9999', '#66b3ff', '#99ff99'])
        ax3.set_title('Priority Level Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('')
        
        # 4. Performance Metrics
        metrics_names = ['Accuracy', 'F1-Score', 'CV Score']
        metrics_values = [
            self.performance_metrics['accuracy'],
            self.performance_metrics['f1_score'],
            self.performance_metrics['cross_val_mean']
        ]
        
        bars = ax4.bar(metrics_names, metrics_values, color=['#2ecc71', '#3498db', '#9b59b6'])
        ax4.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('reports/model_performance/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='models/priority_predictor.joblib'):
        """Save trained model and preprocessing objects"""
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_artifacts, filename)
        print(f"💾 Model saved as {filename}")
    
    def run_complete_analysis(self):
        """Run complete predictive analytics pipeline"""
        print("🚀 Starting Predictive Analytics Pipeline...")
        print("=" * 50)
        
        # Load and preprocess data
        X, y, df = self.load_and_preprocess_data()
        
        # Train model
        X_test, y_test, y_pred, train_size, test_size = self.train_model(X, y)
        
        # Display results
        print(f"\n📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 30)
        print(f"Accuracy: {self.performance_metrics['accuracy']:.4f}")
        print(f"F1-Score: {self.performance_metrics['f1_score']:.4f}")
        print(f"Cross-validation Score: {self.performance_metrics['cross_val_mean']:.4f} ± {self.performance_metrics['cross_val_std']:.4f}")
        print(f"Training samples: {train_size}")
        print(f"Testing samples: {test_size}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Generate visualizations
        self.visualize_results(df, y_test, y_pred)
        
        # Save model
        self.save_model()
        
        return self.performance_metrics


def main():
    """Main execution function"""
    predictor = PredictiveModel()
    metrics = predictor.run_complete_analysis()
    return metrics


if __name__ == "__main__":
    main()
