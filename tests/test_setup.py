"""
Setup wizard tests: SETUP-01 through SETUP-04.
Stubs marked xfail until shop/ package (Plan 02+) exists.
"""
import pytest


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_wizard_intercepts_all_routes(client, shop_config):
    """SETUP-01: Setup wizard middleware redirects all routes to /setup/step1
    when setup_complete=False.

    Steps:
    - Ensure shop_config has setup_complete=False.
    - GET / (home/dashboard).
    - Assert response redirects to /setup/step1.
    """
    raise NotImplementedError("shop/ not implemented yet")


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_form3_upload_autodetect(client, admin_user, shop_config):
    """SETUP-02: POST /setup/step4/upload with a valid xlsx returns column mapping UI.

    Steps:
    - Construct a minimal AS9102 Form 3 xlsx in memory (BytesIO).
    - POST it to /setup/step4/upload as a multipart file upload.
    - Assert 200 response body contains HTML with dropdown elements for column mapping.
    """
    raise NotImplementedError("shop/ not implemented yet")


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_empty_file_error(client, admin_user, shop_config):
    """SETUP-03: POST /setup/step4/upload with empty bytes returns error before mapping UI.

    Steps:
    - POST an empty bytes object to /setup/step4/upload.
    - Assert an error response is returned (not the column mapping HTML).
    """
    raise NotImplementedError("shop/ not implemented yet")


@pytest.mark.xfail(strict=False, reason="shop/ not implemented yet — Wave 1+")
def test_noncontiguous_char_no(client, admin_user, shop_config):
    """SETUP-04: Non-contiguous char_no values are accepted without normalization.

    Steps:
    - Construct an xlsx with char_no values [1, 5, 99, 200] (non-contiguous).
    - POST it to /setup/step4/upload.
    - Assert all four char_no values appear in the response (accepted as-is).
    """
    raise NotImplementedError("shop/ not implemented yet")
