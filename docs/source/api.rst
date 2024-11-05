api documentation
====================

.. currentmodule:: tubular

base module
------------------

.. autosummary::
    :toctree: api/

    base.BaseTransformer
    base.DataFrameMethodTransformer

capping module
------------------

.. autosummary::
    :toctree: api/
    capping.BaseCappingTransformer
    capping.CappingTransformer
    capping.OutOfRangeNullTransformer
    
comparison module
------------------

.. autosummary::
    :toctree: api/

    comparison.EqualityChecker

dates module
------------------

.. autosummary::
    :toctree: api/

    dates.BaseDateTransformer
    dates.BaseDateTwoColumnTransformer
    dates.BetweenDatesTransformer
    dates.DateDifferenceTransformer
    dates.DateDiffLeapYearTransformer
    dates.SeriesDtMethodTransformer    
    dates.ToDatetimeTransformer
    dates.DatetimeInfoExtractor
    dates.DatetimeSinusoidCalculator

imputers module
------------------

.. autosummary::
    :toctree: api/

    imputers.ArbitraryImputer
    imputers.BaseImputer
    imputers.MeanImputer        
    imputers.MedianImputer
    imputers.ModeImputer
    imputers.NearestMeanResponseImputer    
    imputers.NullIndicator
    
mapping module
------------------

.. autosummary::
    :toctree: api/

    mapping.BaseMappingTransformer
    mapping.BaseMappingTransformMixin
    mapping.MappingTransformer
    mapping.BaseCrossColumnMappingTransformer
    mapping.CrossColumnMappingTransformer
    mappping.BaseCrossColumnNumericTransformer
    mapping.CrossColumnMultiplyTransformer
    mapping.CrossColumnAddTransformer

misc module
------------------

.. autosummary::
    :toctree: api/

    misc.SetValueTransformer
    misc.ColumnDtypeSetter

mixin module
------------------

.. autosummary::
    :toctree: api/

    mixins.DropOriginalMixin
    mixins.NewColumnNameMixin
    mixins.WeightColumnMixin


nominal module
------------------

.. autosummary::
    :toctree: api/

    nominal.BaseNominalTransformer
    nominal.GroupRareLevelsTransformer   
    nominal.MeanResponseTransformer      
    nominal.NominalToIntegerTransformer
    nominal.OrdinalEncoderTransformer
    nominal.OneHotEncodingTransformer 
    
numeric module
------------------

.. autosummary::
    :toctree: api/

    numeric.BaseNumericTransformer
    numeric.LogTransformer
    numeric.CutTransformer  
    numeric.TwoColumnOperatorTransformer
    numeric.ScalingTransformer
    numeric.InteractionTransformer
    numeric.PCATransformer
 
strings module
------------------

.. autosummary::
    :toctree: api/

    strings.SeriesStrMethodTransformer
    strings.StringConcatenator
