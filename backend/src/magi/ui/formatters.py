from magi.services.status import ServiceState, service_status
from magi.ui.config import SERVICE_UIS


def format_status_markdown() -> str:
    """Format the current status as markdown with clickable links."""
    status = service_status.get_all()
    md = ""

    for service, info in status.items():
        # Format service name and status
        service_name = service.replace("_", " ").title()
        line = f"{info.state.to_emoji()} **{service_name}**: {info.message}"

        # Add UI link if available
        if ui_info := SERVICE_UIS.get(service):
            line += f" ([{ui_info.url}]({ui_info.url}))"

        md += line + "\n\n"

    return md


def all_services_ok() -> bool:
    """Check if all services are in OK state."""
    return all(
        info.state == ServiceState.OK for info in service_status.get_all().values()
    )
