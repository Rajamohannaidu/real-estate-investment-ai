# src/explainability_enhanced.py
# Enhanced version with better visualizations and counterfactuals

import numpy as np
import pandas as pd
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelExplainability:
    """
    Enhanced model interpretability with:
    - SHAP (waterfall, force, dependence plots)
    - LIME
    - Counterfactual explanations
    - Feature interactions
    """
    
    def __init__(self, model, X_train, feature_names, model_type='tree'):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.model_type = model_type
        self.shap_explainer = None
        self.lime_explainer = None
        
    def initialize_shap(self):
        """Initialize SHAP explainer based on model type"""
        try:
            if self.model_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(self.model)
                print("✓ Using SHAP TreeExplainer")
            else:
                background = shap.sample(self.X_train, 100)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict, background
                )
                print("✓ Using SHAP KernelExplainer")
        except Exception as e:
            print(f"⚠️ Could not initialize SHAP: {e}")
                
    def initialize_lime(self):
        """Initialize LIME explainer"""
        try:
            X_array = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                X_array,
                feature_names=self.feature_names,
                mode='regression',
                random_state=42
            )
            print("✓ LIME explainer initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize LIME: {e}")
    
    def explain_prediction(self, instance, show_top_n=10):
        """
        Comprehensive explanation of a single prediction
        Returns: dict with SHAP values, LIME values, and visualizations
        """
        if self.shap_explainer is None:
            self.initialize_shap()
        
        # Get prediction
        prediction = self.model.predict(instance)[0]
        
        # SHAP explanation
        shap_values = self.shap_explainer.shap_values(instance)
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create feature importance dict
        feature_impacts = {}
        for i, feature in enumerate(self.feature_names):
            feature_value = instance.iloc[0, i] if isinstance(instance, pd.DataFrame) else instance[0, i]
            feature_impacts[feature] = {
                'value': float(feature_value),
                'shap_value': float(shap_values[i]),
                'impact': 'increases' if shap_values[i] > 0 else 'decreases',
                'magnitude': abs(float(shap_values[i]))
            }
        
        # Sort by magnitude
        sorted_features = dict(sorted(
            feature_impacts.items(),
            key=lambda x: x[1]['magnitude'],
            reverse=True
        ))
        
        return {
            'prediction': prediction,
            'feature_impacts': sorted_features,
            'shap_values': shap_values,
            'top_features': dict(list(sorted_features.items())[:show_top_n])
        }
    
    def create_waterfall_plot(self, instance, prediction_value):
        """
        Create interactive waterfall plot showing feature contributions
        Better than bar chart for understanding cumulative effects
        """
        explanation = self.explain_prediction(instance, show_top_n=10)
        
        features = []
        impacts = []
        cumulative = []
        
        # Base value (average prediction)
        base_value = self.model.predict(self.X_train).mean()
        current = base_value
        
        for feature, data in explanation['top_features'].items():
            features.append(feature)
            impact = data['shap_value']
            impacts.append(impact)
            current += impact
            cumulative.append(current)
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Add bars
        colors = ['green' if i > 0 else 'red' for i in impacts]
        
        fig.add_trace(go.Waterfall(
            name="Feature Impact",
            orientation="v",
            x=features,
            y=impacts,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#fc8181"}},
            increasing={"marker": {"color": "#48bb78"}},
            totals={"marker": {"color": "#667eea"}}
        ))
        
        fig.update_layout(
            title=f"Waterfall Plot: Feature Contributions to Prediction",
            xaxis_title="Features",
            yaxis_title="Impact on Price (₹)",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_force_plot_data(self, instance):
        """
        Prepare data for force plot visualization
        Shows how features push prediction higher or lower
        """
        explanation = self.explain_prediction(instance)
        
        # Separate positive and negative impacts
        positive_features = []
        positive_impacts = []
        negative_features = []
        negative_impacts = []
        
        for feature, data in explanation['top_features'].items():
            if data['shap_value'] > 0:
                positive_features.append(f"{feature}<br>({data['value']:.1f})")
                positive_impacts.append(data['shap_value'])
            else:
                negative_features.append(f"{feature}<br>({data['value']:.1f})")
                negative_impacts.append(abs(data['shap_value']))
        
        return {
            'positive_features': positive_features,
            'positive_impacts': positive_impacts,
            'negative_features': negative_features,
            'negative_impacts': negative_impacts,
            'prediction': explanation['prediction']
        }
    
    def create_feature_importance_plot(self, X_sample=None, top_n=15):
        """
        Create interactive global feature importance plot
        """
        if X_sample is None:
            X_sample = self.X_train.iloc[:100] if isinstance(self.X_train, pd.DataFrame) else self.X_train[:100]
        
        if self.shap_explainer is None:
            self.initialize_shap()
        
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create horizontal bar chart
        fig = px.bar(
            importance_df,
            y='feature',
            x='importance',
            orientation='h',
            title=f'Top {top_n} Most Important Features (Global)',
            labels={'importance': 'Mean |SHAP value|', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def generate_counterfactual(self, instance, feature_to_change, target_change):
        """
        Generate counterfactual explanation:
        "If [feature] was [value], the price would be..."
        
        Args:
            instance: Current property
            feature_to_change: Which feature to modify
            target_change: How much to change it (e.g., +1000 for area)
        """
        # Get original prediction
        original_pred = self.model.predict(instance)[0]
        
        # Create counterfactual instance
        counterfactual = instance.copy()
        feature_idx = self.feature_names.index(feature_to_change)
        
        if isinstance(counterfactual, pd.DataFrame):
            original_value = counterfactual.iloc[0, feature_idx]
            counterfactual.iloc[0, feature_idx] += target_change
            new_value = counterfactual.iloc[0, feature_idx]
        else:
            original_value = counterfactual[0, feature_idx]
            counterfactual[0, feature_idx] += target_change
            new_value = counterfactual[0, feature_idx]
        
        # Get new prediction
        new_pred = self.model.predict(counterfactual)[0]
        
        # Calculate impact
        price_change = new_pred - original_pred
        percent_change = (price_change / original_pred) * 100
        
        return {
            'feature': feature_to_change,
            'original_value': original_value,
            'new_value': new_value,
            'change_amount': target_change,
            'original_price': original_pred,
            'new_price': new_pred,
            'price_change': price_change,
            'percent_change': percent_change,
            'explanation': f"If {feature_to_change} increased from {original_value:.0f} to {new_value:.0f}, "
                          f"the predicted price would change by ₹{abs(price_change):,.0f} "
                          f"({percent_change:+.1f}%)"
        }
    
    def create_what_if_analysis(self, instance):
        """
        Create multiple counterfactuals for key features
        Shows investors how changes affect price
        """
        key_features = ['area', 'bedrooms', 'bathrooms', 'parking']
        changes = {
            'area': 500,      # +500 sq ft
            'bedrooms': 1,    # +1 bedroom
            'bathrooms': 1,   # +1 bathroom
            'parking': 1      # +1 parking
        }
        
        results = []
        for feature in key_features:
            if feature in self.feature_names:
                cf = self.generate_counterfactual(instance, feature, changes[feature])
                results.append(cf)
        
        # Create visualization
        if results:
            df = pd.DataFrame([{
                'Feature': r['feature'],
                'Change': f"+{r['change_amount']:.0f}",
                'Price Impact': r['price_change']
            } for r in results])
            
            fig = px.bar(
                df,
                x='Feature',
                y='Price Impact',
                title='What-If Analysis: Impact of Feature Changes on Price',
                labels={'Price Impact': 'Price Change (₹)'},
                color='Price Impact',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=400)
            
            return {
                'counterfactuals': results,
                'plot': fig
            }
        
        return None
    
    def create_feature_interaction_plot(self, feature1, feature2, X_sample=None):
        """
        Show how two features interact to affect price
        E.g., how area and bedrooms together influence price
        """
        if X_sample is None:
            X_sample = self.X_train.iloc[:100] if isinstance(self.X_train, pd.DataFrame) else self.X_train[:100]
        
        if self.shap_explainer is None:
            self.initialize_shap()
        
        try:
            # Get SHAP interaction values
            shap_interaction = self.shap_explainer.shap_interaction_values(X_sample)
            
            idx1 = self.feature_names.index(feature1)
            idx2 = self.feature_names.index(feature2)
            
            # Extract interaction values
            interaction_values = shap_interaction[:, idx1, idx2]
            
            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=X_sample.iloc[:, idx1] if isinstance(X_sample, pd.DataFrame) else X_sample[:, idx1],
                y=X_sample.iloc[:, idx2] if isinstance(X_sample, pd.DataFrame) else X_sample[:, idx2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=interaction_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Interaction<br>Strength")
                ),
                text=[f"Interaction: {v:.0f}" for v in interaction_values],
                hovertemplate=f"{feature1}: %{{x}}<br>{feature2}: %{{y}}<br>%{{text}}"
            ))
            
            fig.update_layout(
                title=f'Feature Interaction: {feature1} × {feature2}',
                xaxis_title=feature1,
                yaxis_title=feature2,
                height=500
            )
            
            return fig
        
        except Exception as e:
            print(f"⚠️ Could not create interaction plot: {e}")
            return None
    
    def create_comprehensive_report(self, instance):
        """
        Generate complete explainability report for a prediction
        """
        # Basic explanation
        explanation = self.explain_prediction(instance)
        
        # Text summary
        summary = self._generate_text_summary(explanation)
        
        # Visualizations
        waterfall_plot = self.create_waterfall_plot(instance, explanation['prediction'])
        importance_plot = self.create_feature_importance_plot()
        what_if = self.create_what_if_analysis(instance)
        
        return {
            'prediction': explanation['prediction'],
            'summary': summary,
            'feature_impacts': explanation['feature_impacts'],
            'visualizations': {
                'waterfall': waterfall_plot,
                'importance': importance_plot,
                'what_if': what_if
            }
        }
    
    def _generate_text_summary(self, explanation):
        """Generate human-readable summary"""
        lines = [
            f"🏠 Predicted Property Price: ₹{explanation['prediction']:,.0f}\n",
            "📊 Top Factors Influencing This Prediction:\n"
        ]
        
        for i, (feature, data) in enumerate(list(explanation['top_features'].items())[:5], 1):
            direction = "increases" if data['shap_value'] > 0 else "decreases"
            lines.append(
                f"{i}. {feature} (value: {data['value']:.1f}) {direction} "
                f"price by ₹{abs(data['shap_value']):,.0f}"
            )
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("Enhanced Explainability Module")
    print("Supports:")
    print("  ✓ Waterfall plots")
    print("  ✓ Force plots")
    print("  ✓ Counterfactual explanations")
    print("  ✓ What-if analysis")
    print("  ✓ Feature interactions")
    print("  ✓ Global feature importance")