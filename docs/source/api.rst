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

    capping.CappingTransformer
    capping.OutOfRangeNullTransformer

dates module
------------------

.. autosummary::
    :toctree: api/

    dates.BetweenDatesTransformer
    dates.DateDifferenceTransformer
    dates.DateDiffLeapYearTransformer
    dates.SeriesDtMethodTransformer    
    dates.ToDatetimeTransformer

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
    mapping.CrossColumnMappingTransformer    
    mapping.CrossColumnMultiplyTransformer
    mapping.CrossColumnAddTransformer

misc module
------------------

.. autosummary::
    :toctree: api/

    misc.SetValueTransformer

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

    numeric.LogTransformer
    numeric.CutTransformer   
    numeric.ScalingTransformer   
 
strings module
------------------

.. autosummary::
    :toctree: api/

    strings.SeriesStrMethodTransformer
