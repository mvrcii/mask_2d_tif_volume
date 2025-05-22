#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()


def smart_target_inference(source_url, data_dir):
    """Intelligently infer target directory from source URL"""
    # Clean up URL - remove trailing slash for consistent processing
    clean_url = source_url.rstrip('/')

    try:
        # Extract path from URL
        parsed = urlparse(clean_url)
        path_parts = parsed.path.strip('/').split('/')

        # Look for pattern like "full-scrolls/Scroll4/PHerc1667.volpkg/volumes/20231117161658"
        if 'full-scrolls' in path_parts:
            full_scrolls_idx = path_parts.index('full-scrolls')
            remaining_parts = path_parts[full_scrolls_idx + 1:]

            if len(remaining_parts) >= 3:
                scroll_name = remaining_parts[0]  # e.g., "Scroll4"
                pherc_part = remaining_parts[1]  # e.g., "PHerc1667.volpkg"
                volumes_part = remaining_parts[2]  # e.g., "volumes"

                # Build target path components
                target_parts = []

                # Convert scroll name: "Scroll4" -> "scroll4.volpkg"
                if scroll_name.lower().startswith('scroll'):
                    target_parts.append(scroll_name.lower() + '.volpkg')
                else:
                    target_parts.append(scroll_name.lower())

                # Add "volumes" if it exists
                if volumes_part == 'volumes':
                    target_parts.append('volumes')

                # Add any remaining parts (like timestamp directories)
                if len(remaining_parts) > 3:
                    target_parts.extend(remaining_parts[3:])

                target_path = Path(data_dir) / '/'.join(target_parts)
                return target_path

    except Exception:
        pass

    # Fallback: use last few parts of the URL
    try:
        parsed = urlparse(clean_url)
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]
        if len(path_parts) >= 2:
            # Take last 2-3 meaningful parts
            relevant_parts = path_parts[-3:] if len(path_parts) >= 3 else path_parts
            target_path = Path(data_dir) / '/'.join(relevant_parts)
            return target_path
    except Exception:
        pass

    # Final fallback
    return Path(data_dir) / "tif_files"


def get_tif_files_from_directory(url):
    """Extract all TIF file URLs from a directory listing"""
    try:
        with console.status("[bold green]Scanning directory..."):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

        tif_files = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith(('.tif', '.tiff')):
                full_url = urljoin(url, href)
                tif_files.append((href, full_url))

        # Sort files numerically for better progress tracking
        tif_files.sort(key=lambda x: x[0])
        return tif_files

    except requests.RequestException as e:
        console.print(f"[red]Error accessing directory: {e}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return []


def download_file(url, target_path, progress, task_id):
    """Download a single file with progress tracking"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        progress.update(task_id, total=total_size)

        with open(target_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress.update(task_id, advance=len(chunk))

        return True

    except requests.RequestException as e:
        console.print(f"[red]Network error downloading {os.path.basename(target_path)}: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Error downloading {os.path.basename(target_path)}: {e}[/red]")
        return False


def analyze_existing_files(target_dir, tif_files):
    """Analyze what files already exist and calculate size estimates"""
    existing_files = []
    missing_files = []
    file_sizes = []

    for filename, file_url in tif_files:
        target_path = Path(target_dir) / filename
        if target_path.exists() and target_path.stat().st_size > 0:
            file_size = target_path.stat().st_size
            existing_files.append((filename, file_url))
            file_sizes.append(file_size)
        else:
            missing_files.append((filename, file_url))

    # Calculate size estimates if we have existing files
    avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
    total_existing_size = sum(file_sizes)
    estimated_total_size = avg_file_size * len(tif_files) if avg_file_size > 0 else 0
    estimated_remaining_size = avg_file_size * len(missing_files) if avg_file_size > 0 else 0

    return existing_files, missing_files, {
        'total_existing_size': total_existing_size,
        'avg_file_size': avg_file_size,
        'estimated_total_size': estimated_total_size,
        'estimated_remaining_size': estimated_remaining_size
    }


def format_size(size_bytes):
    """Format size in human readable format"""
    if size_bytes == 0:
        return "Unknown"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def interactive_setup(data_dir):
    """Interactive setup to get source URL and confirm target directory"""
    console.print(Panel("[bold blue]TIF File Downloader - Interactive Setup[/bold blue]", title="Welcome"))

    # Get source URL
    while True:
        source_url = Prompt.ask("[green]Enter the source URL containing TIF files[/green]")

        # Clean up URL
        source_url = source_url.strip()
        if not source_url.startswith(('http://', 'https://')):
            console.print("[red]Please enter a valid HTTP/HTTPS URL[/red]")
            continue

        # Ensure trailing slash for consistency
        if not source_url.endswith('/'):
            source_url += '/'

        # Test if URL is accessible
        try:
            with console.status("[bold yellow]Testing URL accessibility..."):
                response = requests.head(source_url, timeout=10)
                response.raise_for_status()
            break
        except requests.RequestException as e:
            console.print(f"[red]Cannot access URL: {e}[/red]")
            if not Confirm.ask("Try a different URL?"):
                sys.exit(1)

    # Infer target directory
    suggested_target = smart_target_inference(source_url, data_dir)

    console.print(f"\n[yellow]Source URL:[/yellow] {source_url}")
    console.print(f"[yellow]Suggested target:[/yellow] {suggested_target}")

    # Confirm or modify target
    if Confirm.ask(f"\nUse suggested target directory?", default=True):
        target_dir = suggested_target
    else:
        custom_target = Prompt.ask("Enter custom target directory")
        target_dir = Path(custom_target).expanduser().resolve()

    return source_url, target_dir


def download_tifs(source_url, target_dir, check_only=False):
    """Download all TIF files from a remote directory with resume capability"""

    # Properly expand ~ and ensure target directory exists
    target_path = Path(target_dir).expanduser().resolve()
    if not check_only:
        target_path.mkdir(parents=True, exist_ok=True)

    # Ensure source URL ends with /
    if not source_url.endswith('/'):
        source_url += '/'

    mode_text = "[bold yellow]SCAN MODE[/bold yellow]" if check_only else "[bold blue]DOWNLOAD MODE[/bold blue]"
    console.print(Panel(
        f"{mode_text} - TIF File Downloader\n[green]Source:[/green] {source_url}\n[green]Target:[/green] {target_path}"))

    # Get list of TIF files
    tif_files = get_tif_files_from_directory(source_url)

    if not tif_files:
        console.print("[red]No TIF files found in the directory[/red]")
        return

    # Analyze existing files
    existing_files, missing_files, size_info = analyze_existing_files(target_path, tif_files)

    # Display status table
    table = Table(title="Download Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Size", justify="right", style="yellow")

    table.add_row("Total files found", str(len(tif_files)),
                  format_size(size_info['estimated_total_size']) if size_info[
                                                                        'estimated_total_size'] > 0 else "Unknown")
    table.add_row("Already downloaded", str(len(existing_files)),
                  format_size(size_info['total_existing_size']))
    table.add_row("To download", str(len(missing_files)),
                  format_size(size_info['estimated_remaining_size']) if size_info[
                                                                            'estimated_remaining_size'] > 0 else "Unknown")

    if size_info['avg_file_size'] > 0:
        table.add_row("Average file size", "", format_size(size_info['avg_file_size']))
        completion_pct = (len(existing_files) / len(tif_files)) * 100 if tif_files else 0
        table.add_row("Progress", f"{completion_pct:.1f}%", "")

    console.print(table)

    if check_only:
        console.print(Panel(
            f"[bold yellow]SCAN COMPLETE[/bold yellow]\n"
            f"Files found: {len(tif_files)}\n"
            f"Already downloaded: {len(existing_files)}\n"
            f"Ready to download: {len(missing_files)}\n"
            f"Storage used: {format_size(size_info['total_existing_size'])}\n"
            f"Estimated remaining: {format_size(size_info['estimated_remaining_size']) if size_info['estimated_remaining_size'] > 0 else 'Unknown'}\n"
            f"Estimated total: {format_size(size_info['estimated_total_size']) if size_info['estimated_total_size'] > 0 else 'Unknown'}\n\n"
            f"[dim]Run without --check to start downloading[/dim]",
            title="Scan Results"
        ))
        return

    if not missing_files:
        console.print("[green]All files already downloaded![/green]")
        return

    # Show which file we're resuming from
    if existing_files:
        last_downloaded = max(existing_files, key=lambda x: x[0])
        console.print(f"[yellow]Resuming after: {last_downloaded[0]}[/yellow]")

    # Download missing files
    console.print(f"\n[bold green]Starting download of {len(missing_files)} files...[/bold green]")

    successful_downloads = 0

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
    ) as progress:

        for i, (filename, file_url) in enumerate(missing_files, 1):
            target_file_path = target_path / filename

            task_id = progress.add_task(f"[cyan]{filename}[/cyan] ({i}/{len(missing_files)})", total=0)

            if download_file(file_url, target_file_path, progress, task_id):
                successful_downloads += 1
                console.print(f"[green]✓[/green] {filename}")
            else:
                # Clean up failed download
                if target_file_path.exists():
                    target_file_path.unlink()
                console.print(f"[red]✗[/red] {filename}")

            progress.remove_task(task_id)

    # Final summary
    final_existing = len(existing_files) + successful_downloads
    final_completion = (final_existing / len(tif_files)) * 100 if tif_files else 0

    console.print(Panel(
        f"[bold green]Download Complete![/bold green]\n"
        f"Successfully downloaded: {successful_downloads}/{len(missing_files)} files\n"
        f"Total files in directory: {final_existing}/{len(tif_files)} ({final_completion:.1f}%)\n"
        f"Files saved to: {target_path}",
        title="Summary"
    ))


def main():
    parser = argparse.ArgumentParser(
        description='Interactive TIF file downloader with smart path inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Mode (Recommended):
  python download_tifs.py
  python download_tifs.py --data-dir ~/my_data
  python download_tifs.py --check

Non-Interactive Mode:
  python download_tifs.py --source https://example.com/tifs/ --target ./local/path/
  python download_tifs.py --check --source https://example.com/tifs/ --target ./local/path/

The script will intelligently suggest target directories based on the source URL.
        """
    )

    parser.add_argument(
        '--data-dir',
        default='/home/marcel/scrollprize/data',
        help='Base data directory for storing downloads (default: /home/marcel/scrollprize/data)'
    )

    parser.add_argument(
        '--source',
        help='Source URL containing TIF files (skips interactive mode)'
    )

    parser.add_argument(
        '--target',
        help='Target directory for downloads (skips interactive mode, requires --source)'
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Scan and analyze files without downloading'
    )

    args = parser.parse_args()

    try:
        # Non-interactive mode
        if args.source and args.target:
            if not args.source.startswith(('http://', 'https://')):
                console.print("[red]Error: source URL must be a valid HTTP/HTTPS URL[/red]")
                sys.exit(1)
            download_tifs(args.source, args.target, check_only=args.check)

        # Interactive mode
        else:
            if args.source or args.target:
                console.print("[red]Error: Both --source and --target are required for non-interactive mode[/red]")
                sys.exit(1)

            source_url, target_dir = interactive_setup(args.data_dir)
            download_tifs(source_url, target_dir, check_only=args.check)

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
