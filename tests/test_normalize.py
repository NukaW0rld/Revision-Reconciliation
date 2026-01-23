import pytest
from delta_preservation.reconcile.normalize import parse_requirement, MatchFingerprint


class TestNormText:
    """Test that norm_text is uppercased and whitespace-collapsed."""
    
    def test_uppercase_conversion(self):
        result = parse_requirement("diameter 8mm")
        assert result.norm_text == "DIAMETER 8MM"
    
    def test_whitespace_collapse(self):
        result = parse_requirement("  length   12.5   mm  ")
        assert result.norm_text == "LENGTH 12.5 MM"
    
    def test_mixed_case_and_whitespace(self):
        result = parse_requirement("  Thread  M4-6H   ")
        assert result.norm_text == "THREAD M4-6H"


class TestCountTokens:
    """Test that count_tokens correctly capture multiplicity like 2 X, 2x, 6 x."""
    
    def test_count_with_uppercase_x(self):
        result = parse_requirement("2 X Ø8 mm holes")
        assert "2X" in result.count_tokens
    
    def test_count_with_lowercase_x(self):
        result = parse_requirement("6x M4 threads")
        assert "6X" in result.count_tokens
    
    def test_count_with_space(self):
        result = parse_requirement("4 x holes")
        assert "4X" in result.count_tokens
    
    def test_count_no_space(self):
        result = parse_requirement("12x countersink")
        assert "12X" in result.count_tokens
    
    def test_multiple_counts(self):
        result = parse_requirement("2X holes and 4 x threads")
        assert "2X" in result.count_tokens
        assert "4X" in result.count_tokens


class TestTypeTokens:
    """Test that type_tokens include expected semantic categories."""
    
    def test_length_type(self):
        result = parse_requirement("Length 100 mm")
        assert "LENGTH" in result.type_tokens
    
    def test_diameter_type(self):
        result = parse_requirement("Diameter Ø8 mm")
        assert "DIAMETER" in result.type_tokens
    
    def test_depth_type(self):
        result = parse_requirement("Depth 5 mm")
        assert "DEPTH" in result.type_tokens
    
    def test_thread_type(self):
        result = parse_requirement("Thread M4-6H")
        assert "THREAD" in result.type_tokens
    
    def test_countersink_type(self):
        result = parse_requirement("Countersink Ø12 mm")
        assert "COUNTERSINK" in result.type_tokens
    
    def test_edge_radius_type(self):
        result = parse_requirement("Edge radius R2 mm")
        assert "EDGE RADIUS" in result.type_tokens
    
    def test_drawing_notes_type(self):
        result = parse_requirement("Drawing notes: all dimensions in mm")
        assert "DRAWING NOTES" in result.type_tokens
    
    def test_multiple_types(self):
        result = parse_requirement("Diameter and depth 8 mm")
        assert "DIAMETER" in result.type_tokens
        assert "DEPTH" in result.type_tokens


class TestSymbolTokens:
    """Test that symbol_tokens correctly include Ø and R where present."""
    
    def test_diameter_symbol(self):
        result = parse_requirement("Ø8 mm")
        assert "Ø" in result.symbol_tokens
    
    def test_radius_symbol(self):
        result = parse_requirement("R2.5 mm")
        assert "R" in result.symbol_tokens
    
    def test_both_symbols(self):
        result = parse_requirement("Ø8 mm with R1 fillet")
        assert "Ø" in result.symbol_tokens
        assert "R" in result.symbol_tokens
    
    def test_no_symbols(self):
        result = parse_requirement("Length 100 mm")
        assert "Ø" not in result.symbol_tokens
        assert "R" not in result.symbol_tokens
    
    def test_radius_with_decimal(self):
        result = parse_requirement("R.5 mm")
        assert "R" in result.symbol_tokens


class TestNumericTokens:
    """Test that numeric_tokens include all numeric values as floats in deterministic order."""
    
    def test_simple_numeric(self):
        result = parse_requirement("Ø8 mm")
        values = [val for val, _ in result.numeric_tokens]
        assert 8.0 in values
    
    def test_decimal_numeric(self):
        result = parse_requirement("Ø8.5 mm")
        values = [val for val, _ in result.numeric_tokens]
        assert 8.5 in values
    
    def test_tolerance_with_plusminus(self):
        result = parse_requirement("Ø8 +/- 0.03 mm")
        values = [val for val, _ in result.numeric_tokens]
        assert 8.0 in values
        assert 0.03 in values
    
    def test_asymmetric_tolerance(self):
        result = parse_requirement("+0.3 / +0.1")
        values = [val for val, _ in result.numeric_tokens]
        assert 0.3 in values
        assert 0.1 in values
    
    def test_multiple_numerics_deterministic_order(self):
        result = parse_requirement("Ø8 +/- 0.03 mm")
        # Should extract in order they appear
        values = [val for val, _ in result.numeric_tokens]
        original_strings = [orig for _, orig in result.numeric_tokens]
        assert len(values) >= 2
        assert "8" in original_strings or "8.0" in original_strings
        assert "0.03" in original_strings


class TestUnits:
    """Test that units resolve correctly to MM, IN, or DEG."""
    
    def test_millimeters(self):
        result = parse_requirement("Ø8 mm")
        assert result.units == "MM"
    
    def test_inches(self):
        result = parse_requirement("Length 2.5 in")
        assert result.units == "IN"
    
    def test_degrees(self):
        result = parse_requirement("Angle 45 deg")
        assert result.units == "DEG"
    
    def test_no_units(self):
        result = parse_requirement("Thread M4-6H")
        assert result.units is None
    
    def test_case_insensitive_mm(self):
        result = parse_requirement("Diameter 8 MM")
        assert result.units == "MM"


class TestPatternClass:
    """Test that pattern_class lands in correct coarse bucket for real examples."""
    
    def test_fillet_radius(self):
        result = parse_requirement("Edge radius R2 mm")
        assert result.pattern_class == "fillet"
    
    def test_hole_diameter(self):
        result = parse_requirement("Ø8 mm hole")
        assert result.pattern_class == "hole"
    
    def test_thread_callout(self):
        result = parse_requirement("Thread M4-6H")
        assert result.pattern_class == "hole"
    
    def test_drawing_note(self):
        result = parse_requirement("Drawing notes: all dimensions in mm")
        assert result.pattern_class == "note"
    
    def test_dimension_length(self):
        result = parse_requirement("Length 100 mm")
        assert result.pattern_class == "dimension"
    
    def test_dimension_depth(self):
        result = parse_requirement("Depth 5 mm")
        assert result.pattern_class == "dimension"
    
    def test_dimension_angle(self):
        result = parse_requirement("Angle 45 deg")
        assert result.pattern_class == "dimension"
    
    def test_countersink_is_hole(self):
        result = parse_requirement("Countersink Ø12 mm")
        assert result.pattern_class == "hole"
    
    def test_radius_prefix(self):
        result = parse_requirement("R 2.5 mm")
        assert result.pattern_class == "fillet"


class TestNumericValueChange:
    """Test that only numeric values change between similar requirements."""
    
    def test_diameter_value_change(self):
        req1 = parse_requirement("Ø8 +/- 0.03 mm")
        req2 = parse_requirement("Ø10 +/- 0.03 mm")
        
        # All non-numeric tokens should match
        assert req1.norm_text != req2.norm_text  # Different because of numeric change
        assert req1.count_tokens == req2.count_tokens
        assert req1.type_tokens == req2.type_tokens
        assert req1.symbol_tokens == req2.symbol_tokens
        assert req1.units == req2.units
        assert req1.pattern_class == req2.pattern_class
        
        # Numeric tokens should differ
        values1 = [val for val, _ in req1.numeric_tokens]
        values2 = [val for val, _ in req2.numeric_tokens]
        assert values1 != values2
        
        # Specifically, the main diameter changed
        assert 8.0 in values1
        assert 10.0 in values2
        
        # But tolerance should be the same
        assert 0.03 in values1
        assert 0.03 in values2
    
    def test_thread_size_change(self):
        req1 = parse_requirement("2X Thread M4-6H")
        req2 = parse_requirement("2X Thread M6-6H")
        
        # All non-numeric tokens should match
        assert req1.count_tokens == req2.count_tokens
        assert req1.type_tokens == req2.type_tokens
        assert req1.symbol_tokens == req2.symbol_tokens
        assert req1.units == req2.units
        assert req1.pattern_class == req2.pattern_class
        
        # Numeric tokens should differ
        values1 = [val for val, _ in req1.numeric_tokens]
        values2 = [val for val, _ in req2.numeric_tokens]
        assert values1 != values2
        assert 4.0 in values1
        assert 6.0 in values2
