from fastapi import (
    APIRouter,
    Depends,
)

from app.modules.auth.dependencies import (
    get_current_user,
)

from app.modules.auth.models import (
    UserModel,
)

from app.modules.sql.dependencies import (
    get_sql_service,
)

from app.modules.sql.schemas import (
    SQLExecuteRequest,
    SQLExecuteResponse,
    SchemaResponse,
)

from app.modules.sql.service import (
    SQLService,
)

from app.shared.responses.base import (
    APIResponse,
)

router = APIRouter(

    prefix="/sql",

    tags=["SQL Lab"],

)

##########################################################
# Execute SQL
##########################################################

@router.post(
    "/execute",
    response_model=APIResponse[
        SQLExecuteResponse
    ],
    summary="Execute SQL Query",
)
async def execute_query(

    request: SQLExecuteRequest,

    current_user: UserModel = Depends(
        get_current_user,
    ),

    service: SQLService = Depends(
        get_sql_service,
    ),

):

    result = service.execute(

        current_user=current_user,

        query=request.query,

    )

    return APIResponse(

        success=True,

        message="Query executed successfully.",

        data=SQLExecuteResponse(
            **result
        ),

    )


##########################################################
# Database Schema
##########################################################

@router.get(
    "/schema",
    response_model=APIResponse[
        SchemaResponse
    ],
    summary="Get Database Schema",
)
async def schema(

    current_user: UserModel = Depends(
        get_current_user,
    ),

    service: SQLService = Depends(
        get_sql_service,
    ),

):

    result = service.schema(

        current_user=current_user,

    )

    return APIResponse(

        success=True,

        message="Schema loaded.",

        data=SchemaResponse(
            **result,
        ),

    )


##########################################################
# Database Statistics
##########################################################

@router.get(
    "/statistics",
    response_model=APIResponse[dict],
    summary="Database Statistics",
)
async def statistics(

    current_user: UserModel = Depends(
        get_current_user,
    ),

    service: SQLService = Depends(
        get_sql_service,
    ),

):

    result = service.statistics(

        current_user=current_user,

    )

    return APIResponse(

        success=True,

        message="Statistics loaded successfully.",

        data=result,

    )


##########################################################
# Reset Database
##########################################################

@router.post(
    "/reset",
    response_model=APIResponse[None],
    summary="Reset User Database",
)
async def reset_database(

    current_user: UserModel = Depends(
        get_current_user,
    ),

    service: SQLService = Depends(
        get_sql_service,
    ),

):

    service.reset_database(

        current_user=current_user,

    )

    return APIResponse(

        success=True,

        message="Database reset successfully.",

        data=None,

    )