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
    clean_url = source_url.rstrip('/')

    try:
        parsed = urlparse(clean_url)
        path_parts = parsed.path.strip('/').split('/')

        if 'full-scrolls' in path_parts:
            full_scrolls_idx = path_parts.index('full-scrolls')
            remaining_parts = path_parts[full_scrolls_idx + 1:]

            if len(remaining_parts) >= 3:
                scroll_name = remaining_parts[0]
                pherc_part = remaining_parts[1]
                volumes_part = remaining_parts[2]

                target_parts = []

                if scroll_name.lower().startswith('scroll'):
                    target_parts.append(scroll_name.lower() + '.volpkg')
                else:
                    target_parts.append(scroll_name.lower())

                if volumes_part == 'volumes':
                    target_parts.append('volumes')

                if len(remaining_parts) > 3:
                    target_parts.extend(remaining_parts[3:])

                target_path = Path(data_dir) / '/'.join(target_parts)
                return target_path

    except Exception:
        pass

    try:
        parsed = urlparse(clean_url)
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]
        if len(path_parts) >= 2:
            relevant_parts = path_parts[-3:] if len(path_parts) >= 3 else path_parts
            target_path = Path(data_dir) / '/'.join(relevant_parts)
            return target_path
    except Exception:
        pass

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

        tif_files.sort(key=lambda x: x[0])
        return tif_files

    except requests.RequestException as e:
        console.print(f"[red]Error accessing directory: {e}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return []


def get_expected_file_size(existing_file_sizes):
    """Determine expected file size from existing files (most common size)"""
    if not existing_file_sizes:
        return None

    # Count occurrences of each file size
    size_counts = {}
    for size in existing_file_sizes:
        size_counts[size] = size_counts.get(size, 0) + 1

    # Return the most common size
    return max(size_counts.items(), key=lambda x: x[1])[0]


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
    """Analyze what files already exist, detect corruption by size comparison, and calculate size estimates"""
    existing_files = []
    missing_files = []
    corrupted_files = []
    all_file_sizes = []

    # First pass: collect all existing files and their sizes
    for filename, file_url in tif_files:
        target_path = Path(target_dir) / filename

        if target_path.exists() and target_path.stat().st_size > 0:
            actual_size = target_path.stat().st_size
            existing_files.append((filename, file_url, actual_size))
            all_file_sizes.append(actual_size)
        else:
            missing_files.append((filename, file_url))

    # Determine expected file size from existing files
    expected_size = get_expected_file_size(all_file_sizes)

    # Second pass: identify corrupted files
    valid_files = []
    valid_file_sizes = []

    for filename, file_url, actual_size in existing_files:
        if expected_size is not None and actual_size != expected_size:
            corrupted_files.append((filename, file_url, actual_size, expected_size))
        else:
            valid_files.append((filename, file_url))
            valid_file_sizes.append(actual_size)

    # Calculate size estimates using only valid files
    avg_file_size = sum(valid_file_sizes) / len(valid_file_sizes) if valid_file_sizes else 0
    total_existing_size = sum(valid_file_sizes)
    estimated_total_size = avg_file_size * len(tif_files) if avg_file_size > 0 else 0
    estimated_remaining_size = avg_file_size * (len(missing_files) + len(corrupted_files)) if avg_file_size > 0 else 0

    return valid_files, missing_files, corrupted_files, {
        'total_existing_size': total_existing_size,
        'avg_file_size': avg_file_size,
        'estimated_total_size': estimated_total_size,
        'estimated_remaining_size': estimated_remaining_size,
        'expected_file_size': expected_size
    }


def handle_corrupted_files(corrupted_files, target_dir):
    """Display corrupted files and prompt user for deletion"""
    if not corrupted_files:
        return []

    console.print("\n[bold red]Corrupted files detected (size mismatch):[/bold red]")

    corruption_table = Table(title="Corrupted Files")
    corruption_table.add_column("Filename", style="cyan")
    corruption_table.add_column("Actual Size", justify="right", style="red")
    corruption_table.add_column("Expected Size", justify="right", style="green")
    corruption_table.add_column("Difference", justify="right", style="yellow")

    for filename, file_url, actual_size, expected_size in corrupted_files:
        difference = actual_size - expected_size
        diff_str = f"{'+' if difference > 0 else ''}{format_size(difference)}"
        corruption_table.add_row(
            filename,
            format_size(actual_size),
            format_size(expected_size),
            diff_str
        )

    console.print(corruption_table)

    if Confirm.ask(f"\n[bold yellow]Delete {len(corrupted_files)} corrupted files and re-download them?[/bold yellow]",
                   default=True):
        deleted_files = []
        for filename, file_url, actual_size, expected_size in corrupted_files:
            target_path = Path(target_dir) / filename
            try:
                target_path.unlink()
                deleted_files.append((filename, file_url))
                console.print(f"[red]Deleted:[/red] {filename}")
            except Exception as e:
                console.print(f"[red]Failed to delete {filename}: {e}[/red]")

        return deleted_files
    else:
        console.print("[yellow]Keeping corrupted files. They will not be re-downloaded.[/yellow]")
        return []


def format_size(size_bytes):
    """Format size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    if size_bytes < 0:
        return f"-{format_size(-size_bytes)}"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def interactive_setup(data_dir):
    """Interactive setup to get source URL and confirm target directory"""
    console.print(Panel("[bold blue]TIF File Downloader - Interactive Setup[/bold blue]", title="Welcome", expand=False))

    while True:
        source_url = Prompt.ask("[green]Enter the source URL containing TIF files[/green]")

        source_url = source_url.strip()
        if not source_url.startswith(('http://', 'https://')):
            console.print("[red]Please enter a valid HTTP/HTTPS URL[/red]")
            continue

        if not source_url.endswith('/'):
            source_url += '/'

        try:
            with console.status("[bold yellow]Testing URL accessibility..."):
                response = requests.head(source_url, timeout=10)
                response.raise_for_status()
            break
        except requests.RequestException as e:
            console.print(f"[red]Cannot access URL: {e}[/red]")
            if not Confirm.ask("Try a different URL?"):
                sys.exit(1)

    suggested_target = smart_target_inference(source_url, data_dir)

    console.print(f"\n[yellow]Source URL:[/yellow] {source_url}")
    console.print(f"[yellow]Suggested target:[/yellow] {suggested_target}")

    if Confirm.ask(f"\nUse suggested target directory?", default=True):
        target_dir = suggested_target
    else:
        custom_target = Prompt.ask("Enter custom target directory")
        target_dir = Path(custom_target).expanduser().resolve()

    return source_url, target_dir


def download_tifs(source_url, target_dir, check_only=False):
    """Download all TIF files from a remote directory with resume capability and corruption detection"""

    target_path = Path(target_dir).expanduser().resolve()
    if not check_only:
        target_path.mkdir(parents=True, exist_ok=True)

    if not source_url.endswith('/'):
        source_url += '/'

    mode_text = "[bold yellow]SCAN MODE[/bold yellow]" if check_only else "[bold blue]DOWNLOAD MODE[/bold blue]"
    console.print(Panel(
        f"{mode_text} - TIF File Downloader\n[green]Source:[/green] {source_url}\n[green]Target:[/green] {target_path}",expand=False))

    tif_files = get_tif_files_from_directory(source_url)

    if not tif_files:
        console.print("[red]No TIF files found in the directory[/red]")
        return

    # Analyze existing files and detect corruption by size comparison
    existing_files, missing_files, corrupted_files, size_info = analyze_existing_files(target_path, tif_files)

    # Handle corrupted files FIRST (before showing final status)
    files_to_redownload = []
    if not check_only and corrupted_files:
        files_to_redownload = handle_corrupted_files(corrupted_files, target_path)

    # Calculate final download queue
    all_files_to_download = missing_files + files_to_redownload

    # Display status table with accurate counts
    table = Table(title="Download Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Size", justify="right", style="yellow")

    table.add_row("Total files found", str(len(tif_files)),
                  format_size(size_info['estimated_total_size']) if size_info[
                                                                        'estimated_total_size'] > 0 else "Unknown")
    table.add_row("Already downloaded", str(len(existing_files)),
                  format_size(size_info['total_existing_size']))

    # Show corruption info
    if corrupted_files:
        corrupted_size = sum(actual_size for _, _, actual_size, _ in corrupted_files)
        if check_only:
            table.add_row("[red]Corrupted files[/red]", f"[red]{len(corrupted_files)}[/red]",
                          f"[red]{format_size(corrupted_size)}[/red]")
        else:
            # Show what happened to corrupted files
            skipped_corrupted = len(corrupted_files) - len(files_to_redownload)
            if files_to_redownload:
                table.add_row("[yellow]Corrupted (to redownload)[/yellow]",
                              f"[yellow]{len(files_to_redownload)}[/yellow]", "")
            if skipped_corrupted > 0:
                table.add_row("[red]Corrupted (skipped)[/red]", f"[red]{skipped_corrupted}[/red]", "")

    # Show accurate download count
    table.add_row("To download", str(len(all_files_to_download)),
                  format_size(size_info['estimated_remaining_size']) if size_info[
                                                                            'estimated_remaining_size'] > 0 else "Unknown")

    if size_info['avg_file_size'] > 0:
        table.add_row("Average file size", "", format_size(size_info['avg_file_size']))
        completion_pct = (len(existing_files) / len(tif_files)) * 100 if tif_files else 0
        table.add_row("Progress", f"{completion_pct:.1f}%", "")

    console.print(table)

    if check_only:
        corruption_status = f"Corrupted: {len(corrupted_files)}" if corrupted_files else "No corruption detected"
        console.print(Panel(
            f"[bold yellow]SCAN COMPLETE[/bold yellow]\n"
            f"Files found: {len(tif_files)}\n"
            f"Already downloaded: {len(existing_files)}\n"
            f"Missing: {len(missing_files)}\n"
            f"{corruption_status}\n"
            f"Ready to download: {len(missing_files) + len(corrupted_files)}\n"
            f"Storage used: {format_size(size_info['total_existing_size'])}\n"
            f"Estimated remaining: {format_size(size_info['estimated_remaining_size']) if size_info['estimated_remaining_size'] > 0 else 'Unknown'}\n"
            f"Estimated total: {format_size(size_info['estimated_total_size']) if size_info['estimated_total_size'] > 0 else 'Unknown'}\n"
            f"Expected file size: {format_size(size_info['expected_file_size']) if size_info['expected_file_size'] else 'Unknown'}\n\n"
            f"[dim]Run without --check to start downloading[/dim]",
            title="Scan Results",
            expand=False
        ))
        return

    if not all_files_to_download:
        console.print("[green]All files already downloaded and verified![/green]")
        return

    # Show which file we're resuming from
    if existing_files:
        last_downloaded = max(existing_files, key=lambda x: x[0])
        console.print(f"[yellow]Resuming after: {last_downloaded[0]}[/yellow]")

    if files_to_redownload:
        console.print(f"[yellow]Re-downloading {len(files_to_redownload)} corrupted files[/yellow]")

    console.print(f"\n[bold green]Starting download of {len(all_files_to_download)} files...[/bold green]")
    console.print(
        f"[dim]Download queue: {len(missing_files)} missing + {len(files_to_redownload)} corrupted = {len(all_files_to_download)} total[/dim]")

    successful_downloads = 0

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
    ) as progress:

        for i, (filename, file_url) in enumerate(all_files_to_download, 1):
            target_file_path = target_path / filename

            task_id = progress.add_task(f"[cyan]{filename}[/cyan] ({i}/{len(all_files_to_download)})", total=0)

            if download_file(file_url, target_file_path, progress, task_id):
                successful_downloads += 1
                console.print(f"[green]✓[/green] {filename}")
            else:
                if target_file_path.exists():
                    target_file_path.unlink()
                console.print(f"[red]✗[/red] {filename}")

            progress.remove_task(task_id)

    final_existing = len(existing_files) + successful_downloads
    final_completion = (final_existing / len(tif_files)) * 100 if tif_files else 0

    console.print(Panel(
        f"[bold green]Download Complete![/bold green]\n"
        f"Successfully downloaded: {successful_downloads}/{len(all_files_to_download)} files\n"
        f"Total files in directory: {final_existing}/{len(tif_files)} ({final_completion:.1f}%)\n"
        f"Files saved to: {target_path}",
        title="Summary",
        expand=False
    ))

    successful_downloads = 0

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
    ) as progress:

        for i, (filename, file_url) in enumerate(all_files_to_download, 1):
            target_file_path = target_path / filename

            task_id = progress.add_task(f"[cyan]{filename}[/cyan] ({i}/{len(all_files_to_download)})", total=0)

            if download_file(file_url, target_file_path, progress, task_id):
                successful_downloads += 1
                console.print(f"[green]✓[/green] {filename}")
            else:
                if target_file_path.exists():
                    target_file_path.unlink()
                console.print(f"[red]✗[/red] {filename}")

            progress.remove_task(task_id)

    final_existing = len(existing_files) + successful_downloads
    final_completion = (final_existing / len(tif_files)) * 100 if tif_files else 0

    console.print(Panel(
        f"[bold green]Download Complete![/bold green]\n"
        f"Successfully downloaded: {successful_downloads}/{len(all_files_to_download)} files\n"
        f"Total files in directory: {final_existing}/{len(tif_files)} ({final_completion:.1f}%)\n"
        f"Files saved to: {target_path}",
        title="Summary",
        expand=False
    ))


def main():
    parser = argparse.ArgumentParser(
        description='Interactive TIF file downloader with smart path inference and corruption detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Mode (Recommended):
  python download_tifs.py
  python download_tifs.py --data-dir ~/my_data
  python download_tifs.py --check

Non-Interactive Mode:
  python download_tifs.py --source https://example.com/tifs/ --target ./local/path/
  python download_tifs.py --check --source https://example.com/tifs/ --target ./local/path/

The script will intelligently suggest target directories based on the source URL
and detect corrupted files by comparing local file sizes (assumes all files should be same size).
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
        if args.source and args.target:
            if not args.source.startswith(('http://', 'https://')):
                console.print("[red]Error: source URL must be a valid HTTP/HTTPS URL[/red]")
                sys.exit(1)
            download_tifs(args.source, args.target, check_only=args.check)

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
