import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score
from typing import Sequence, Optional, Union, Dict, List, Optional


def weighted_cohens_kappa(
    ratings1: Sequence[int],
    ratings2: Sequence[int],
    weights: Optional[Union[str, Sequence[float]]] = None
) -> float:
    """
    Calculate the weighted Cohen's kappa statistic for two sets of ratings.

    Parameters:
    ratings1 (Sequence[int]): First set of ratings.
    ratings2 (Sequence[int]): Second set of ratings.
    weights (Optional[Union[str, Sequence[float]]]): Optional weights for the categories. 
        If None, unweighted kappa is calculated. Can be 'linear', 'quadratic', or a sequence of weights.

    Returns:
    float: The weighted Cohen's kappa statistic.
    """
    if weights is None:
        return cohen_kappa_score(ratings1, ratings2)
    return cohen_kappa_score(ratings1, ratings2, weights=weights)


def analyze_interrater_reliability(data: pd.DataFrame, 
                                 subject_col: str = 'subject',
                                 rater_col: str = 'rater', 
                                 score_col: str = 'score',
                                 rater_groups: Optional[Dict[str, List[str]]] = None) -> Dict:
    """
    Analyze inter-rater reliability using ICC, with optional group comparisons.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Long format data with columns for subject, rater, and score
    subject_col : str
        Column name for subjects (e.g., 'report_id')
    rater_col : str  
        Column name for raters (e.g., 'rater_name')
    score_col : str
        Column name for scores
    rater_groups : dict, optional
        Dictionary mapping group names to lists of rater names
        e.g., {'LLM': ['LLM1', 'LLM2'], 'Human': ['Human1', 'Human2']}
    
    Returns:
    --------
    dict : Results containing ICC values and interpretations
    """
    
    def interpret_icc(icc_value):
        """Interpret ICC values"""
        if icc_value < 0.5:
            return "Poor"
        elif icc_value < 0.75:
            return "Moderate" 
        elif icc_value < 0.9:
            return "Good"
        else:
            return "Excellent"
    
    results = {}
    
    # Overall ICC
    overall_icc = pg.intraclass_corr(data=data,
                                   targets=subject_col,
                                   raters=rater_col, 
                                   ratings=score_col)
    
    # Extract ICC(2,1)
    icc_2_1 = overall_icc[overall_icc['Type'] == 'ICC2']['ICC'].iloc[0]
    ci_lower = overall_icc[overall_icc['Type'] == 'ICC2']['CI95%'].iloc[0][0]
    ci_upper = overall_icc[overall_icc['Type'] == 'ICC2']['CI95%'].iloc[0][1]
    
    results['overall'] = {
        'ICC': round(icc_2_1, 4),
        'CI_lower': round(ci_lower, 4),
        'CI_upper': round(ci_upper, 4),
        'interpretation': interpret_icc(icc_2_1),
        'n_subjects': data[subject_col].nunique(),
        'n_raters': data[rater_col].nunique()
    }
    
    # Group-wise analysis if groups provided
    if rater_groups:
        results['groups'] = {}
        
        for group_name, rater_list in rater_groups.items():
            # Filter data for this group
            group_data = data[data[rater_col].isin(rater_list)].copy()
            
            if len(group_data[rater_col].unique()) < 2:
                results['groups'][group_name] = {'error': 'Need at least 2 raters'}
                continue
                
            # Calculate ICC for this group
            group_icc = pg.intraclass_corr(data=group_data,
                                         targets=subject_col,
                                         raters=rater_col,
                                         ratings=score_col)
            
            group_icc_2_1 = group_icc[group_icc['Type'] == 'ICC2']['ICC'].iloc[0]
            group_ci_lower = group_icc[group_icc['Type'] == 'ICC2']['CI95%'].iloc[0][0]
            group_ci_upper = group_icc[group_icc['Type'] == 'ICC2']['CI95%'].iloc[0][1]
            
            results['groups'][group_name] = {
                'ICC': round(group_icc_2_1, 4),
                'CI_lower': round(group_ci_lower, 4), 
                'CI_upper': round(group_ci_upper, 4),
                'interpretation': interpret_icc(group_icc_2_1),
                'n_raters': len(rater_list)
            }
    
    return results


def print_icc_summary(results: Dict):
    """Pretty print ICC analysis results"""
    
    print("=" * 60)
    print("INTER-RATER RELIABILITY ANALYSIS")
    print("=" * 60)
    
    # Overall results
    overall = results['overall']
    print(f"\nOVERALL RELIABILITY:")
    print(f"  ICC(2,1): {overall['ICC']:.4f} [{overall['CI_lower']:.4f}, {overall['CI_upper']:.4f}]")
    print(f"  Interpretation: {overall['interpretation']}")
    print(f"  Subjects: {overall['n_subjects']}, Raters: {overall['n_raters']}")
    
    # Group results
    if 'groups' in results:
        print(f"\nGROUP-WISE RELIABILITY:")
        for group_name, group_data in results['groups'].items():
            if 'error' in group_data:
                print(f"  {group_name}: {group_data['error']}")
            else:
                print(f"  {group_name}: ICC = {group_data['ICC']:.4f} [{group_data['CI_lower']:.4f}, {group_data['CI_upper']:.4f}] ({group_data['interpretation']})")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example 1
    ratings1 = [2, 2, 1, 3]
    ratings2 = [3, 3, 2, 4]
    weights = "linear"

    kappa = weighted_cohens_kappa(ratings1, ratings2, weights)
    print(f"Weighted Cohen's Kappa: {kappa:.4f}")

    # Example 2
    sample_data = pd.DataFrame({
        'report': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 
                  'D', 'D', 'D', 'D', 'E', 'E', 'E', 'E'],
        'rater': ['LLM1', 'LLM2', 'Human1', 'Human2'] * 5,
        'score': [3, 2, 3, 4, 1, 2, 2, 1, 4, 4, 3, 4, 2, 3, 2, 3, 4, 3, 4, 3]
    })
    
    groups = {
        'LLM': ['LLM1', 'LLM2'],
        'Human': ['Human1', 'Human2']
    }
    
    results = analyze_interrater_reliability(
        data=sample_data,
        subject_col='report',
        rater_col='rater', 
        score_col='score',
        rater_groups=groups
    )
    
    print_icc_summary(results)
    print(f"\nKey findings:")
    print(f"Overall consistency: {results['overall']['ICC']:.3f}")
    if 'groups' in results:
        llm_icc = results['groups']['LLM']['ICC']
        human_icc = results['groups']['Human']['ICC'] 
        print(f"LLM consistency: {llm_icc:.3f}")
        print(f"Human consistency: {human_icc:.3f}")