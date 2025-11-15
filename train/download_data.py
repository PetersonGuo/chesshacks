"""
Download Lichess dataset and evaluate positions with Stockfish (Simplified Workflow)

This script downloads chess games from the Lichess database, extracts positions,
and evaluates them using Stockfish to create a training dataset for NNUE models.

Simplified workflow:
    1. Auto-detects and downloads the latest Lichess database
    2. Downloads exactly 10,000 games (configurable)
    3. Extracts 10 random positions from each game (no filtering)
    4. Evaluates all positions with Stockfish
    5. Outputs to JSONL format for training

Requirements:
    - Stockfish installed and available in PATH (or provide path with --stockfish-path)
    - Python packages: requests, zstandard, python-chess, tqdm

Example usage:
    # Download and process with defaults (10k games, latest database)
    python download_data.py

    # Download from specific month/year
    python download_data.py --year 2024 --month 10

    # Use custom Stockfish path and higher depth
    python download_data.py --stockfish-path /usr/local/bin/stockfish --depth 15

    # Process more games with more workers
    python download_data.py --max-games 50000 --num-workers 8

    # Save as CSV format
    python download_data.py --output-format csv
"""

import os
import json
import csv
import argparse
import io
import random
import requests
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, Iterator
import chess
import chess.pgn
import chess.engine
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import TrainingConfig, get_config


def _resolve_config_params(config: Optional[TrainingConfig], **kwargs) -> Tuple[TrainingConfig, Dict[str, Any]]:
    """Helper to resolve parameters from config or kwargs"""
    if config is None:
        config = get_config('default')

    resolved = {}
    for key, value in kwargs.items():
        if value is not None:
            resolved[key] = value
        elif hasattr(config, f'download_{key}'):
            resolved[key] = getattr(config, f'download_{key}')
        elif hasattr(config, key):
            resolved[key] = getattr(config, key)
        else:
            resolved[key] = None

    return config, resolved


def _get_latest_lichess_database() -> Tuple[int, int]:
    """
    Auto-detect the latest available Lichess database month/year

    Returns:
        Tuple of (year, month) for the latest available database
    """
    from datetime import datetime

    # Start with current month and work backwards
    current = datetime.now()
    year = current.year
    month = current.month

    # Try current month and previous 3 months
    for attempt in range(4):
        month_str = f"{month:02d}"
        date_str = f"{year}-{month_str}"
        filename = f"lichess_db_standard_rated_{date_str}.pgn.zst"
        url = f"https://database.lichess.org/standard/{filename}"

        # Check if file exists with HEAD request
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                print(f"Auto-detected latest database: {date_str}")
                return year, month
        except:
            pass

        # Go back one month
        month -= 1
        if month == 0:
            month = 12
            year -= 1

    # If nothing found, default to October 2025 (fallback)
    print("Warning: Could not auto-detect latest database, using fallback: 2025-10")
    return 2025, 10


def stream_lichess_database(output_dir: Optional[str] = None, year: Optional[int] = None,
                            month: Optional[int] = None, rated_only: Optional[bool] = None,
                            max_games: Optional[int] = None, max_games_searched: Optional[int] = None,
                            download_mode: Optional[str] = None,
                            skip_filter: Optional[bool] = None,
                            skip_redownload: Optional[bool] = None,
                            config: Optional[TrainingConfig] = None) -> str:
    """
    Download Lichess database (simplified - no filtering)

    Args:
        output_dir: Directory to save games (defaults to config.download_output_dir or 'data')
        year: Year of the database (defaults to None = auto-detect latest)
        month: Month of the database (defaults to None = auto-detect latest)
        rated_only: Download rated games only (defaults to config.download_rated_only or True)
        max_games: Maximum games to download (used by extract_positions_from_pgn, not enforced here)
        max_games_searched: Maximum total games to search (not used in simplified workflow)
        download_mode: Download mode - 'streaming' or 'direct' (defaults to 'streaming')
        skip_filter: Skip filtering entirely - always True in simplified workflow
        skip_redownload: Skip re-downloading if file already exists (defaults to True)
        config: TrainingConfig instance (optional, for defaults)

    Returns:
        Path to downloaded PGN file (.zst compressed)
    """
    config, params = _resolve_config_params(
        config,
        output_dir=output_dir,
        year=year,
        month=month,
        rated_only=rated_only,
        mode=download_mode,
        skip_filter=skip_filter,
        skip_redownload=skip_redownload
    )

    output_dir = params['output_dir']
    year = params['year']
    month = params['month']
    rated_only = params['rated_only']
    download_mode = params.get('mode', 'streaming')
    skip_redownload = params.get('skip_redownload', True)

    # Auto-detect latest database if year/month not specified
    if year is None or month is None:
        year, month = _get_latest_lichess_database()

    os.makedirs(output_dir, exist_ok=True)

    month_str = f"{month:02d}"
    date_str = f"{year}-{month_str}"

    if rated_only:
        filename = f"lichess_db_standard_rated_{date_str}.pgn.zst"
    else:
        filename = f"lichess_db_standard_{date_str}.pgn.zst"

    url = f"https://database.lichess.org/standard/{filename}"
    compressed_path = os.path.join(output_dir, filename)

    # Check if we should skip re-downloading (only for direct mode)
    if download_mode == 'direct' and skip_redownload and os.path.exists(compressed_path):
        print(f"Found existing download: {compressed_path}")
        print(f"Skipping re-download (file size: {os.path.getsize(compressed_path) / (1024**3):.2f} GB)")
        print("Set skip_redownload=False or delete the file to re-download")
        return compressed_path

    # Use streaming mode to download only needed games
    if download_mode == 'streaming':
        # Check if we already have a streaming file with enough games
        if max_games:
            streaming_filename = f"lichess_games_streamed_{max_games}.pgn"
            streaming_path = os.path.join(output_dir, streaming_filename)
            if skip_redownload and os.path.exists(streaming_path):
                # Count games in existing file to verify it has enough
                try:
                    with open(streaming_path, 'r', encoding='utf-8') as f:
                        existing_games = sum(1 for line in f if line.startswith('[Event'))
                    if existing_games >= max_games:
                        print(f"Found existing streaming file: {streaming_path}")
                        print(f"  Games: {existing_games}")
                        print(f"  File size: {os.path.getsize(streaming_path) / (1024**2):.2f} MB")
                        print("Skipping re-download. Set skip_redownload=False to re-download")
                        return streaming_path
                except Exception as e:
                    print(f"Warning: Could not verify existing streaming file: {e}")
                    print("Will re-download...")
        
        print(f"Streaming Lichess database: {filename}")
        print(f"URL: {url}")
        print(f"Mode: Streaming (will only download games needed)")
        return _stream_download_games(url, output_dir, max_games)
    else:
        # Download the full database file
        print(f"Downloading Lichess database: {filename}")
        print(f"URL: {url}")
        print(f"Downloading to: {compressed_path}")
        return _direct_download_only(url, compressed_path)


def _stream_download_games(url: str, output_dir: str, max_games: Optional[int] = None) -> str:
    """
    Stream download from URL, decompress on-the-fly, and save only the games we need
    
    Args:
        url: URL to download from
        output_dir: Directory to save the output PGN file
        max_games: Maximum number of games to download (None = all)
    
    Returns:
        Path to the saved PGN file
    """
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard package is required for streaming downloads. "
            "Install it with: pip install zstandard"
        )
    
    # Create output filename
    output_filename = f"lichess_games_streamed_{max_games or 'all'}.pgn"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Streaming from URL and extracting games...")
    print(f"Max games to extract: {max_games if max_games else 'All'}")
    
    # Stream download and decompress on-the-fly
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    dctx = zstd.ZstdDecompressor()
    decompressor_reader = dctx.stream_reader(response.raw)
    pgn_stream = io.TextIOWrapper(decompressor_reader, encoding='utf-8')
    
    game_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        with tqdm(desc="Streaming games", unit=" games") as pbar:
            while True:
                if max_games and game_count >= max_games:
                    break
                
                # Read one game using chess.pgn (handles multi-line games properly)
                game = chess.pgn.read_game(pgn_stream)
                if game is None:
                    break
                
                # Write the game to output file
                print(game, file=out_file)
                print(file=out_file)  # Empty line between games
                
                game_count += 1
                pbar.update(1)
                
                if max_games and game_count >= max_games:
                    break
    
    # Close the stream
    try:
        pgn_stream.close()
    except:
        pass
    try:
        decompressor_reader.close()
    except:
        pass
    
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"\nStreaming complete!")
    print(f"  Games extracted: {game_count}")
    print(f"  Output file: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return output_path


def _direct_download_only(url: str, compressed_path: str) -> str:
    """
    Download the compressed file only, without filtering
    
    Args:
        url: URL to download from
        compressed_path: Path to save the compressed file
    
    Returns:
        Path to the compressed file
    """
    print(f"Downloading to: {compressed_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    with open(compressed_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"\nDownload complete. File size: {os.path.getsize(compressed_path) / (1024**3):.2f} GB")
    print(f"Saved to: {compressed_path}")
    print("Skipping filtering - will use compressed file directly for position extraction")
    
    return compressed_path


def extract_positions_from_pgn(pgn_path: str, max_games: Optional[int] = None,
                                positions_per_game: Optional[int] = None,
                                config: Optional[TrainingConfig] = None) -> Iterator[Tuple[str, Dict[str, str]]]:
    """
    Extract positions from PGN file (simplified - no filtering)

    Args:
        pgn_path: Path to PGN file (can be .zst compressed)
        max_games: Maximum number of games to process (None = all)
        positions_per_game: Number of positions to sample per game (defaults to config.download_positions_per_game or 10)
        config: TrainingConfig instance (optional, for defaults)

    Yields:
        Tuple of (fen, game_info) for each position
    """
    config, params = _resolve_config_params(
        config,
        positions_per_game=positions_per_game
    )

    positions_per_game = params['positions_per_game']

    print(f"Reading from: {pgn_path}")
    print(f"Configuration:")
    print(f"  Positions per game: {positions_per_game}")
    print(f"  Max games: {max_games if max_games else 'All'}")
    print(f"  No filtering - using all positions from games")
    print()
    
    # Track file handles for proper cleanup
    pgn_file = None
    file_handle = None
    decompressor_reader = None
    
    if pgn_path.endswith('.zst'):
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError(
                "zstandard package is required for .zst files. "
                "Install it with: pip install zstandard"
            )
        
        print("Opening compressed file (.zst)...")
        dctx = zstd.ZstdDecompressor()
        file_handle = open(pgn_path, 'rb')
        decompressor_reader = dctx.stream_reader(file_handle)
        pgn_file = io.TextIOWrapper(decompressor_reader, encoding='utf-8')
        print("Compressed file opened successfully")
    else:
        print("Opening PGN file...")
        pgn_file = open(pgn_path, 'r', encoding='utf-8', errors='ignore')
        print("PGN file opened successfully")
    
    game_count = 0
    games_processed = 0
    total_positions_extracted = 0
    games_with_no_positions = 0

    try:
        with tqdm(total=max_games, desc="Extracting positions", unit=" games") as pbar:
            while True:
                if max_games and game_count >= max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                game_count += 1
                games_processed += 1

                game_info = {
                    'white': game.headers.get('White', ''),
                    'black': game.headers.get('Black', ''),
                    'result': game.headers.get('Result', ''),
                    'elo_white': game.headers.get('WhiteElo', ''),
                    'elo_black': game.headers.get('BlackElo', ''),
                }

                board = game.board()

                # Collect all positions from this game
                all_positions = []
                for move in game.mainline_moves():
                    board.push(move)
                    all_positions.append(board.fen())

                # Randomly sample positions_per_game from all positions
                num_to_sample = min(positions_per_game, len(all_positions))
                if num_to_sample > 0:
                    sampled = random.sample(all_positions, num_to_sample)
                    for fen in sampled:
                        total_positions_extracted += 1
                        yield fen, game_info
                else:
                    games_with_no_positions += 1
                
                # Update progress bar with positions count
                pbar.set_postfix({'positions': total_positions_extracted})
                pbar.update(1)

    finally:
        # Print summary
        print("=" * 80)
        print("Position extraction summary:")
        print(f"  Total games read: {game_count}")
        print(f"  Games processed: {games_processed}")
        print(f"  Games with no positions: {games_with_no_positions}")
        print(f"  Total positions extracted: {total_positions_extracted}")
        if games_processed > 0:
            avg_positions = total_positions_extracted / games_processed
            print(f"  Average positions per processed game: {avg_positions:.2f}")
        print("=" * 80)
        
        # Close file handles in reverse order of opening
        if pgn_file:
            try:
                pgn_file.close()
            except:
                pass
        if decompressor_reader:
            try:
                decompressor_reader.close()
            except:
                pass
        if file_handle:
            try:
                file_handle.close()
            except:
                pass


def evaluate_position_with_stockfish(fen: str, engine: chess.engine.SimpleEngine, 
                                     depth: int = 10) -> Optional[int]:
    """
    Evaluate a position using Stockfish
    
    Args:
        fen: FEN string of the position
        engine: chess.engine.SimpleEngine instance
        depth: Search depth for evaluation
    
    Returns:
        Evaluation score in centipawns (positive = white advantage)
    """
    try:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info['score']
        
        score_white = score.pov(chess.WHITE)
        
        if score_white.is_mate():
            mate_score = score_white.mate()
            return 10000 if mate_score > 0 else -10000
        else:
            return int(score_white.score())
    
    except Exception as e:
        print(f"Error evaluating position {fen}: {e}")
        return None


def process_positions_batch(positions_batch: list, engine_path: str, depth: int = 10) -> list:
    """
    Process a batch of positions with Stockfish

    Args:
        positions_batch: List of (fen, game_info) tuples
        engine_path: Path to Stockfish executable
        depth: Search depth for evaluation

    Returns:
        List of (fen, eval) tuples
    """
    results = []

    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            for fen, game_info in tqdm(positions_batch, desc="Evaluating positions", leave=False):
                eval_score = evaluate_position_with_stockfish(fen, engine, depth)
                if eval_score is not None:
                    results.append((fen, eval_score))
    except Exception as e:
        print(f"Error in batch processing: {e}")

    return results


def download_and_process_lichess_data(
    output_dir=None,
    year=None,
    month=None,
    rated_only=None,
    stockfish_path=None,
    depth=None,
    max_games=None,
    max_games_searched=None,
    positions_per_game=None,
    max_positions=None,
    output_format=None,
    num_workers=None,
    batch_size=None,
    download_mode=None,
    skip_filter=None,
    skip_redownload=None,
    config=None
):
    """
    Download Lichess database and process with Stockfish evaluations (simplified workflow)

    Args:
        output_dir: Directory to save output files (defaults to config.download_output_dir or 'data')
        year: Year of database to download (defaults to None = auto-detect latest)
        month: Month of database to download (defaults to None = auto-detect latest)
        rated_only: Download rated games only (defaults to config.download_rated_only or True)
        stockfish_path: Path to Stockfish executable (defaults to config.stockfish_path or 'stockfish')
        depth: Stockfish search depth (defaults to config.download_depth or 10)
        max_games: Maximum games to download (defaults to config.download_max_games or 10000)
        max_games_searched: Maximum total games to search (None = unlimited)
        positions_per_game: Positions to sample per game (defaults to config.download_positions_per_game or 10)
        max_positions: Maximum positions to evaluate (None = all)
        output_format: Output format ('csv', 'json', or 'jsonl') (defaults to config.download_output_format or 'jsonl')
        num_workers: Number of parallel workers for evaluation (defaults to config.download_num_workers or 4)
        batch_size: Batch size for parallel processing (defaults to config.download_batch_size or 100)
        download_mode: Download mode - 'streaming' or 'direct' (defaults to config.download_mode or 'streaming')
        skip_filter: Skip filtering entirely - use all games (defaults to config.download_skip_filter or True)
        skip_redownload: Skip re-downloading if file already exists (defaults to config.download_skip_redownload or True)
        config: TrainingConfig instance (optional, for defaults)
    """
    config, params = _resolve_config_params(
        config,
        output_dir=output_dir,
        year=year,
        month=month,
        rated_only=rated_only,
        stockfish_path=stockfish_path,
        depth=depth,
        max_games=max_games,
        positions_per_game=positions_per_game,
        output_format=output_format,
        num_workers=num_workers,
        batch_size=batch_size,
        mode=download_mode,
        skip_filter=skip_filter,
        skip_redownload=skip_redownload
    )

    # Extract resolved parameters
    output_dir = params['output_dir']
    year = params['year']
    month = params['month']
    rated_only = params['rated_only']
    stockfish_path = params['stockfish_path']
    depth = params['depth']
    max_games = params['max_games']
    positions_per_game = params['positions_per_game']
    output_format = params['output_format']
    num_workers = params['num_workers']
    batch_size = params['batch_size']
    download_mode = params['mode']
    skip_filter = params.get('skip_filter', True)
    skip_redownload = params.get('skip_redownload', True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Downloading Lichess database (simplified workflow)")
    print("=" * 80)
    pgn_path = stream_lichess_database(
        output_dir=output_dir,
        year=year,
        month=month,
        rated_only=rated_only,
        max_games=max_games,
        max_games_searched=max_games_searched,
        download_mode=download_mode,
        skip_filter=skip_filter,
        skip_redownload=skip_redownload,
        config=config
    )

    print("\n" + "=" * 80)
    print("Extracting positions from games")
    print("=" * 80)

    positions = []
    position_count = 0

    for fen, game_info in extract_positions_from_pgn(
        pgn_path,
        max_games=max_games,
        positions_per_game=positions_per_game,
        config=config
    ):
        positions.append((fen, game_info))
        position_count += 1
        
        if max_positions and position_count >= max_positions:
            break
    
    print(f"Extracted {len(positions)} positions from {max_games or 'all'} games")
    
    print("\n" + "=" * 80)
    print("Evaluating positions with Stockfish")
    print("=" * 80)
    
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            print(f"Stockfish found at: {stockfish_path}")
    except Exception as e:
        print(f"Error: Could not start Stockfish at {stockfish_path}")
        print(f"Please install Stockfish and provide the correct path.")
        print(f"Error: {e}")
        return
    
    evaluated_positions = []
    batches = []
    for i in range(0, len(positions), batch_size):
        batch = positions[i:i+batch_size]
        batches.append(batch)
    
    print(f"Processing {len(batches)} batches with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(process_positions_batch, batch, stockfish_path, depth)
            futures.append(future)

        with tqdm(total=len(futures), desc="Batch progress", unit="batch") as pbar:
            for future in as_completed(futures):
                results = future.result()
                evaluated_positions.extend(results)
                pbar.update(1)

    print(f"\nEvaluated {len(evaluated_positions)} positions")
    
    print("\n" + "=" * 80)
    print("Saving results")
    print("=" * 80)
    
    # Extract year/month from filename if not already set
    if year is None or month is None:
        import re
        filename = os.path.basename(pgn_path)
        match = re.search(r'(\d{4})-(\d{2})', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
        else:
            # Fallback: use current date or generic name
            from datetime import datetime
            now = datetime.now()
            year = year or now.year
            month = month or now.month
    
    if output_format == 'csv':
        output_path = os.path.join(output_dir, f'lichess_evaluated_{year}-{month:02d}.csv')
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['fen', 'eval'])
            for fen, eval_score in evaluated_positions:
                writer.writerow([fen, eval_score])
    
    elif output_format == 'json':
        output_path = os.path.join(output_dir, f'lichess_evaluated_{year}-{month:02d}.json')
        data = [{'fen': fen, 'eval': eval_score} for fen, eval_score in evaluated_positions]
        with open(output_path, 'w') as f:
            json.dump(data, f)
    
    elif output_format == 'jsonl':
        output_path = os.path.join(output_dir, f'lichess_evaluated_{year}-{month:02d}.jsonl')
        with open(output_path, 'w') as f:
            for fen, eval_score in evaluated_positions:
                json.dump({'fen': fen, 'eval': eval_score}, f)
                f.write('\n')
    
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    
    print(f"Saved {len(evaluated_positions)} positions to: {output_path}")
    print(f"Output format: {output_format}")
    
    return output_path


def main():
    # Load config for defaults
    config = get_config('default')
    
    parser = argparse.ArgumentParser(
        description='Download Lichess dataset and evaluate positions with Stockfish'
    )
    
    parser.add_argument('--output-dir', type=str, default=config.download_output_dir,
                       help=f'Directory to save output files (default: {config.download_output_dir})')
    parser.add_argument('--year', type=int, default=None,
                       help='Year of database to download (default: auto-detect latest)')
    parser.add_argument('--month', type=int, default=None,
                       help='Month of database to download (1-12, default: auto-detect latest)')
    parser.add_argument('--all-games', action='store_true',
                       help='Download all games (not just rated)')
    parser.add_argument('--stockfish-path', type=str, default=config.stockfish_path,
                       help=f'Path to Stockfish executable (default: {config.stockfish_path})')
    parser.add_argument('--depth', type=int, default=config.download_depth,
                       help=f'Stockfish search depth (default: {config.download_depth})')
    parser.add_argument('--max-games', type=int, default=config.download_max_games,
                       help=f'Maximum games to download (default: {config.download_max_games})')
    parser.add_argument('--positions-per-game', type=int, default=config.download_positions_per_game,
                       help=f'Positions to sample per game (default: {config.download_positions_per_game})')
    parser.add_argument('--max-positions', type=int, default=None,
                       help='Maximum positions to evaluate (default: all)')
    parser.add_argument('--output-format', type=str, default=config.download_output_format,
                       choices=['csv', 'json', 'jsonl'],
                       help=f'Output format (default: {config.download_output_format})')
    parser.add_argument('--num-workers', type=int, default=config.download_num_workers,
                       help=f'Number of parallel workers (default: {config.download_num_workers})')
    parser.add_argument('--batch-size', type=int, default=config.download_batch_size,
                       help=f'Batch size for parallel processing (default: {config.download_batch_size})')
    parser.add_argument('--download-mode', type=str, default='streaming',
                       choices=['streaming', 'direct'],
                       help='Download mode: streaming or direct (default: streaming)')
    parser.add_argument('--skip-filter', action='store_true', default=config.download_skip_filter,
                       help='Skip filtering entirely - use all games from the database (default: True)')
    parser.add_argument('--skip-redownload', action='store_true', default=config.download_skip_redownload,
                       help='Skip re-downloading if file already exists (default: True)')

    args = parser.parse_args()

    download_and_process_lichess_data(
        output_dir=args.output_dir,
        year=args.year,
        month=args.month,
        rated_only=not args.all_games,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        max_games=args.max_games,
        positions_per_game=args.positions_per_game,
        max_positions=args.max_positions,
        output_format=args.output_format,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        download_mode=args.download_mode,
        skip_filter=args.skip_filter,
        skip_redownload=args.skip_redownload,
        config=config
    )


if __name__ == '__main__':
    main()
