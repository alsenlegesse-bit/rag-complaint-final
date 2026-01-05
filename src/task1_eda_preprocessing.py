#!/usr/bin/env python3
"""
Task 1: Exploratory Data Analysis and Data Preprocessing
CrediTrust Financial - Complaint Analysis System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sys
from pathlib import Path
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

class CrediTrustEDA:
    """Main class for Task 1 EDA and Preprocessing"""
    
    def __init__(self):
        self.df = None
        self.df_filtered = None
        self.narrative_col = None
        self.product_col = None
        self.output_path = "data/processed/filtered_complaints.csv"
        
    def setup_project(self):
        """Create necessary directories"""
        print("🔧 Setting up project structure...")
        
        directories = [
            'data/raw',
            'data/processed',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Created: {directory}")
        
        print("✅ Project structure ready!\n")
        return True
    
    def show_data_instructions(self):
        """Display data download instructions"""
        print("=" * 80)
        print("📥 DATA DOWNLOAD REQUIRED")
        print("=" * 80)
        print("\nTo complete Task 1, you need to download the CFPB complaint dataset:")
        print("\n🔗 Download link:")
        print("   https://drive.google.com/file/d/1MMmioXFFOVMIc7GTrXNefgXM6UiHuCZ8/view?usp=sharing")
        print("\n💾 Save the file as:")
        print("   data/raw/complaints.csv")
        print("\n📁 Expected file structure:")
        print("   rag-complaint-chatbot/")
        print("   └── data/")
        print("       └── raw/")
        print("           └── complaints.csv")
        print("\n⏳ Once downloaded, run this script again.")
        print("=" * 80)
        return False
    
    def load_data(self):
        """Load the CFPB complaint dataset"""
        data_path = "data/raw/complaints.csv"
        
        if not os.path.exists(data_path):
            print(f"❌ Dataset not found at: {data_path}")
            return self.show_data_instructions()
        
        print(f"📂 Loading dataset from: {data_path}")
        
        try:
            # Try different encodings if needed
            try:
                self.df = pd.read_csv(data_path, low_memory=False)
            except UnicodeDecodeError:
                self.df = pd.read_csv(data_path, encoding='latin-1', low_memory=False)
                print("   ⚠️  Used latin-1 encoding")
            
            print(f"✅ Successfully loaded!")
            print(f"   📊 Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def identify_key_columns(self):
        """Identify important columns in the dataset"""
        print("\n🔍 Identifying key columns...")
        
        # Find narrative column
        narrative_candidates = ['Consumer complaint narrative', 'Complaint narrative', 
                               'Consumer_complaint_narrative', 'complaint_narrative']
        
        for col in narrative_candidates:
            if col in self.df.columns:
                self.narrative_col = col
                print(f"   ✓ Narrative column: '{self.narrative_col}'")
                break
        
        if not self.narrative_col:
            # Try to find by pattern
            for col in self.df.columns:
                if 'narrative' in str(col).lower() or 'complaint' in str(col).lower():
                    self.narrative_col = col
                    print(f"   ⚠️  Found narrative-like column: '{self.narrative_col}'")
                    break
        
        # Find product column
        product_candidates = ['Product', 'product', 'Product category', 'product_category']
        for col in product_candidates:
            if col in self.df.columns:
                self.product_col = col
                print(f"   ✓ Product column: '{self.product_col}'")
                break
        
        if not self.product_col:
            for col in self.df.columns:
                if 'product' in str(col).lower():
                    self.product_col = col
                    print(f"   ⚠️  Found product-like column: '{self.product_col}'")
                    break
        
        return self.narrative_col is not None and self.product_col is not None
    
    def basic_eda(self):
        """Perform basic exploratory data analysis"""
        print("\n" + "=" * 80)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # 1. Basic info
        print(f"\n1️⃣  DATASET OVERVIEW")
        print(f"   • Total records: {self.df.shape[0]:,}")
        print(f"   • Total columns: {self.df.shape[1]}")
        
        # 2. Data types
        print(f"\n2️⃣  DATA TYPES")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   • {dtype}: {count} columns")
        
        # 3. Missing values
        print(f"\n3️⃣  MISSING VALUES")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        # Top 10 columns with most missing values
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_pct
        }).sort_values('Missing_Percent', ascending=False)
        
        print("\n   Top 10 columns with missing values:")
        for idx, row in missing_df.head(10).iterrows():
            print(f"   • {idx}: {row['Missing_Count']:,} ({row['Missing_Percent']:.1f}%)")
        
        # 4. Narrative analysis
        if self.narrative_col:
            print(f"\n4️⃣  NARRATIVE ANALYSIS")
            
            # Calculate narrative lengths
            self.df['narrative_length'] = self.df[self.narrative_col].astype(str).apply(
                lambda x: len(str(x).split())
            )
            
            # Count narratives
            has_narrative = self.df[self.narrative_col].notna() & (self.df['narrative_length'] > 5)
            narrative_count = has_narrative.sum()
            
            print(f"   • With narratives: {narrative_count:,} ({narrative_count/len(self.df)*100:.1f}%)")
            print(f"   • Without narratives: {len(self.df) - narrative_count:,}")
            print(f"   • Average length: {self.df[has_narrative]['narrative_length'].mean():.1f} words")
            print(f"   • Median length: {self.df[has_narrative]['narrative_length'].median():.1f} words")
            
            # Save narrative length plot
            self.plot_narrative_lengths()
    
    def plot_narrative_lengths(self):
        """Plot distribution of narrative lengths"""
        plt.figure(figsize=(10, 6))
        
        # Get narratives
        has_narrative = self.df[self.narrative_col].notna() & (self.df['narrative_length'] > 5)
        lengths = self.df[has_narrative]['narrative_length']
        
        # Create histogram
        plt.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean: {lengths.mean():.1f}')
        plt.axvline(lengths.median(), color='green', linestyle='--', label=f'Median: {lengths.median():.1f}')
        
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.title('Distribution of Complaint Narrative Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('reports/narrative_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   • Saved plot: reports/narrative_length_distribution.png")
    
    def analyze_products(self):
        """Analyze product distribution"""
        if not self.product_col:
            print("❌ Product column not found!")
            return
        
        print(f"\n5️⃣  PRODUCT DISTRIBUTION")
        
        product_counts = self.df[self.product_col].value_counts()
        
        print(f"\n   Total unique products: {len(product_counts)}")
        print(f"\n   Top 10 products by complaint count:")
        for product, count in product_counts.head(10).items():
            percentage = (count / len(self.df)) * 100
            print(f"   • {product}: {count:,} ({percentage:.1f}%)")
        
        # Plot product distribution
        self.plot_product_distribution(product_counts)
    
    def plot_product_distribution(self, product_counts):
        """Plot product distribution"""
        plt.figure(figsize=(12, 8))
        
        # Top 15 products
        top_15 = product_counts.head(15)
        
        bars = plt.barh(range(len(top_15)), top_15.values)
        plt.yticks(range(len(top_15)), top_15.index)
        plt.xlabel('Number of Complaints')
        plt.title('Top 15 Products by Number of Complaints')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + max(top_15.values)*0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{int(bar.get_width()):,}', 
                    ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('reports/product_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   • Saved plot: reports/product_distribution.png")
    
    def filter_required_products(self):
        """Filter for the 5 required product categories"""
        print("\n" + "=" * 80)
        print("🎯 FILTERING FOR REQUIRED PRODUCTS")
        print("=" * 80)
        
        # Define required products (case-insensitive matching)
        required_products = [
            'credit card', 
            'personal loan', 
            'buy now, pay later', 
            'savings account', 
            'money transfer'
        ]
        
        print(f"\nLooking for these 5 products:")
        for product in required_products:
            print(f"   • {product.title()}")
        
        # Convert product column to lowercase for matching
        self.df['product_lower'] = self.df[self.product_col].astype(str).str.lower()
        
        # Find matches
        matched_mask = self.df['product_lower'].str.contains('|'.join(required_products), na=False)
        
        self.df_filtered = self.df[matched_mask].copy()
        
        print(f"\n📈 Filtering Results:")
        print(f"   • Original dataset: {len(self.df):,} complaints")
        print(f"   • After filtering: {len(self.df_filtered):,} complaints")
        print(f"   • Percentage kept: {len(self.df_filtered)/len(self.df)*100:.1f}%")
        
        # Show distribution in filtered data
        filtered_counts = self.df_filtered[self.product_col].value_counts()
        print(f"\n📊 Distribution in filtered data:")
        for product, count in filtered_counts.items():
            percentage = (count / len(self.df_filtered)) * 100
            print(f"   • {product}: {count:,} ({percentage:.1f}%)")
        
        # Remove temporary column
        self.df.drop('product_lower', axis=1, inplace=True, errors='ignore')
        if 'product_lower' in self.df_filtered.columns:
            self.df_filtered.drop('product_lower', axis=1, inplace=True)
    
    def remove_empty_narratives(self):
        """Remove records with empty narratives"""
        print("\n" + "=" * 80)
        print("🧹 REMOVING EMPTY NARRATIVES")
        print("=" * 80)
        
        if not self.narrative_col:
            print("❌ Narrative column not found!")
            return
        
        before_count = len(self.df_filtered)
        
        # Remove empty or very short narratives
        self.df_filtered = self.df_filtered[
            self.df_filtered[self.narrative_col].notna() & 
            (self.df_filtered['narrative_length'] > 5)
        ].copy()
        
        after_count = len(self.df_filtered)
        
        print(f"\n📊 Results:")
        print(f"   • Before: {before_count:,} complaints")
        print(f"   • After: {after_count:,} complaints")
        print(f"   • Removed: {before_count - after_count:,} complaints")
        print(f"   • Percentage kept: {after_count/before_count*100:.1f}%")
    
    def clean_text_narratives(self):
        """Clean and preprocess text narratives"""
        print("\n" + "=" * 80)
        print("✨ CLEANING TEXT NARRATIVES")
        print("=" * 80)
        
        def clean_text(text):
            """Clean individual text narrative"""
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove common boilerplate phrases
            boilerplate = [
                r'i am writing to file a complaint',
                r'i would like to file a complaint',
                r'this is a complaint regarding',
                r'to whom it may concern',
                r'dear sir or madam',
                r'cc:\s*\w+',
                r'xx/xx/xxxx',
                r'xxxxxxxx',
            ]
            
            for phrase in boilerplate:
                text = re.sub(phrase, '', text, flags=re.IGNORECASE)
            
            # Remove special characters (keep basic punctuation)
            text = re.sub(r'[^\w\s.,!?\-]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            return text
        
        print("\n🧼 Cleaning narratives...")
        
        # Apply cleaning
        self.df_filtered['cleaned_narrative'] = self.df_filtered[self.narrative_col].apply(clean_text)
        
        # Calculate cleaned lengths
        self.df_filtered['cleaned_length'] = self.df_filtered['cleaned_narrative'].apply(
            lambda x: len(str(x).split())
        )
        
        # Remove narratives that became empty after cleaning
        before_clean = len(self.df_filtered)
        self.df_filtered = self.df_filtered[self.df_filtered['cleaned_length'] > 3].copy()
        after_clean = len(self.df_filtered)
        
        print(f"\n📊 Cleaning Results:")
        print(f"   • Original average length: {self.df_filtered['narrative_length'].mean():.1f} words")
        print(f"   • Cleaned average length: {self.df_filtered['cleaned_length'].mean():.1f} words")
        print(f"   • Words removed: {self.df_filtered['narrative_length'].sum() - self.df_filtered['cleaned_length'].sum():,}")
        print(f"   • Removed empty narratives: {before_clean - after_clean}")
        print(f"   • Final count: {after_clean:,} complaints")
    
    def save_processed_data(self):
        """Save the processed dataset"""
        print("\n" + "=" * 80)
        print("💾 SAVING PROCESSED DATA")
        print("=" * 80)
        
        # Select relevant columns
        columns_to_keep = [
            self.product_col,
            self.narrative_col,
            'cleaned_narrative',
            'narrative_length',
            'cleaned_length'
        ]
        
        # Add any other columns that might be useful
        for col in ['Issue', 'Sub-issue', 'Company', 'State', 'Date received']:
            if col in self.df_filtered.columns:
                columns_to_keep.append(col)
        
        # Create final dataframe
        final_df = self.df_filtered[columns_to_keep].copy()
        
        # Save to CSV
        final_df.to_csv(self.output_path, index=False)
        
        print(f"\n✅ Data saved successfully!")
        print(f"   • File: {self.output_path}")
        print(f"   • Shape: {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
        print(f"   • Size: {os.path.getsize(self.output_path) / 1024**2:.2f} MB")
        
        return final_df
    
    def generate_summary_report(self):
        """Generate a summary report of the preprocessing"""
        print("\n" + "=" * 80)
        print("📋 TASK 1 SUMMARY REPORT")
        print("=" * 80)
        
        summary = f"""
        TASK 1 COMPLETION SUMMARY
        {'=' * 60}
        
        1. DATA LOADING:
           • Source: CFPB Complaint Dataset
           • Original size: {self.df.shape[0]:,} complaints × {self.df.shape[1]} columns
        
        2. DATA FILTERING:
           • Products kept: 5 specified financial products
           • After filtering: {len(self.df_filtered):,} complaints
           • Percentage retained: {len(self.df_filtered)/len(self.df)*100:.1f}%
        
        3. TEXT PREPROCESSING:
           • Original narratives cleaned and normalized
           • Removed boilerplate text and special characters
           • Final average length: {self.df_filtered['cleaned_length'].mean():.1f} words
        
        4. OUTPUT:
           • Saved to: {self.output_path}
           • Final size: {self.df_filtered.shape[0]:,} complaints
           • Ready for Task 2: Text chunking and embedding
        
        {'=' * 60}
        ✅ TASK 1 COMPLETED SUCCESSFULLY!
        """
        
        print(summary)
        
        # Save report to file
        with open('reports/task1_summary.txt', 'w') as f:
            f.write(summary)
        
        print(f"\n📄 Report saved: reports/task1_summary.txt")
    
    def run(self):
        """Main execution method"""
        print("\n" + "=" * 80)
        print("🚀 CREDITRUST FINANCIAL - TASK 1: EDA & PREPROCESSING")
        print("=" * 80)
        
        # Step 1: Setup
        self.setup_project()
        
        # Step 2: Load data
        if not self.load_data():
            return
        
        # Step 3: Identify columns
        if not self.identify_key_columns():
            print("❌ Could not identify key columns. Check your dataset.")
            return
        
        # Step 4: Basic EDA
        self.basic_eda()
        
        # Step 5: Analyze products
        self.analyze_products()
        
        # Step 6: Filter for required products
        self.filter_required_products()
        
        # Step 7: Remove empty narratives
        self.remove_empty_narratives()
        
        # Step 8: Clean text narratives
        self.clean_text_narratives()
        
        # Step 9: Save processed data
        self.save_processed_data()
        
        # Step 10: Generate summary
        self.generate_summary_report()


def main():
    """Main function"""
    try:
        # Create instance and run
        eda = CrediTrustEDA()
        eda.run()
        
        print("\n🎉 All tasks completed successfully!")
        print("📁 Next steps:")
        print("   1. Check the 'reports/' folder for visualizations")
        print("   2. Check 'data/processed/filtered_complaints.csv' for processed data")
        print("   3. Proceed to Task 2: Text chunking and vector store creation")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
