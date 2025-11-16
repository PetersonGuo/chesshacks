"""
Download Lichess dataset and evaluate positions with Stockfish

This module downloads chess games from the Lichess database, extracts positions,
and evaluates them using Stockfish to create a training dataset for NNUE models.

Simplified workflow:
    1. Auto-detects and downloads the latest Lichess database
    2. Downloads exactly 10,000 games (configurable)
    3. Extracts 10 random positions from each game (no filtering)
    4. Evaluates all positions with Stockfish
    5. Outputs to JSONL format for training
"""

import os
import json
import io
import random
import requests
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, Iterator
from contextlib import contextmanager
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

    # Try current month and previous 12 months (full year)
    for attempt in range(13):
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
        except Exception as e:
            # Silently continue to next month
            pass

        # Go back one month
        month -= 1
        if month == 0:
            month = 12
            year -= 1

    # If nothing found, try a known good recent month (October 2024)
    # Lichess databases are typically available with a 1-2 month delay
    print("Warning: Could not auto-detect latest database, trying known recent months...")
    for test_year, test_month in [(2024, 12), (2024, 11), (2024, 10), (2024, 9)]:
        month_str = f"{test_month:02d}"
        date_str = f"{test_year}-{month_str}"
        filename = f"lichess_db_standard_rated_{date_str}.pgn.zst"
        url = f"https://database.lichess.org/standard/{filename}"
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                print(f"Using fallback database: {date_str}")
                return test_year, test_month
        except:
            pass

    # Final fallback
    print("Warning: All auto-detection failed, using final fallback: 2024-10")
    return 2024, 10


def stream_lichess_database(output_dir: Optional[str] = None, year: Optional[int] = None,
                            month: Optional[int] = None, rated_only: Optional[bool] = None,
                            max_games: Optional[int] = None,
                            download_mode: Optional[str] = None,
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
        download_mode: Download mode - 'streaming' or 'direct' (defaults to 'streaming')
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
        max_games=max_games,
        mode=download_mode,
        skip_redownload=skip_redownload
    )

    output_dir = params['output_dir']
    year = params['year']
    month = params['month']
    rated_only = params['rated_only']
    download_mode = params.get('mode', 'streaming')
    skip_redownload = params.get('skip_redownload', True)
    max_games = params.get('max_games')

    os.makedirs(output_dir, exist_ok=True)

    # Check for existing streaming file BEFORE any download attempts
    if max_games and download_mode == 'streaming':
        streaming_filename = f"lichess_games_streamed_{max_games}.pgn"
        streaming_path = os.path.join(output_dir, streaming_filename)
        if os.path.exists(streaming_path):
            if skip_redownload:
                # Count games in existing file to verify it has enough
                try:
                    with open(streaming_path, 'r', encoding='utf-8') as f:
                        existing_games = sum(1 for line in f if line.startswith('[Event'))
                    if existing_games >= max_games:
                        print(f"✓ Found existing streaming file: {streaming_path}")
                        print(f"  Games: {existing_games}")
                        print(f"  File size: {os.path.getsize(streaming_path) / (1024**2):.2f} MB")
                        print("  Skipping download. Set skip_redownload=False to re-download")
                        return streaming_path
                    else:
                        print(f"⚠ Existing file has only {existing_games} games, need {max_games}. Will re-download...")
                except Exception as e:
                    print(f"⚠ Warning: Could not verify existing streaming file: {e}")
                    print("  Will re-download...")
            else:
                print(f"Found existing streaming file: {streaming_path}")
                print("  skip_redownload=False, will re-download...")

    # Auto-detect latest database if year/month not specified
    if year is None or month is None:
        year, month = _get_latest_lichess_database()

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
                                config: Optional[TrainingConfig] = None) -> Iterator[str]:
    """
    Extract positions from PGN file (simplified - no filtering)

    Args:
        pgn_path: Path to PGN file (can be .zst compressed)
        max_games: Maximum number of games to process (None = all)
        positions_per_game: Number of positions to sample per game (defaults to config.download_positions_per_game or 10)
        config: TrainingConfig instance (optional, for defaults)

    Yields:
        FEN string for each position
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
    print()
    
    @contextmanager
    def open_pgn_file(path):
        """Context manager for opening PGN files (compressed or plain)"""
        if path.endswith('.zst'):
            try:
                import zstandard as zstd
            except ImportError:
                raise ImportError(
                    "zstandard package is required for .zst files. "
                    "Install it with: pip install zstandard"
                )
            file_handle = open(path, 'rb')
            try:
                dctx = zstd.ZstdDecompressor()
                reader = dctx.stream_reader(file_handle)
                pgn_file = io.TextIOWrapper(reader, encoding='utf-8')
                try:
                    yield pgn_file
                finally:
                    try:
                        pgn_file.close()
                    except:
                        pass
                    try:
                        reader.close()
                    except:
                        pass
            finally:
                file_handle.close()
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                yield f
    
    game_count = 0
    total_positions_extracted = 0
    games_with_no_positions = 0

    with open_pgn_file(pgn_path) as pgn_file:
        with tqdm(total=max_games, desc="Extracting positions", unit=" games") as pbar:
            while True:
                if max_games and game_count >= max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                game_count += 1
                board = game.board()

                # Use reservoir sampling to sample positions on-the-fly
                # This avoids collecting all positions in memory first
                sampled_positions = []
                move_count = 0
                
                for move in game.mainline_moves():
                    board.push(move)
                    move_count += 1
                    fen = board.fen()
                    
                    # Reservoir sampling: for first k positions, add directly
                    if len(sampled_positions) < positions_per_game:
                        sampled_positions.append(fen)
                    else:
                        # For position i, replace a random existing position with probability k/i
                        # move_count is 0-indexed, so we've seen move_count+1 positions total
                        j = random.randint(0, move_count)
                        if j < positions_per_game:
                            sampled_positions[j] = fen

                # Yield all sampled positions
                if len(sampled_positions) > 0:
                    for fen in sampled_positions:
                        total_positions_extracted += 1
                        yield fen
                else:
                    games_with_no_positions += 1
                
                # Update progress bar with positions count
                pbar.set_postfix({'positions': total_positions_extracted})
                pbar.update(1)

    # Print summary
    print("=" * 80)
    print("Position extraction summary:")
    print(f"  Total games read: {game_count}")
    print(f"  Games with no positions: {games_with_no_positions}")
    print(f"  Total positions extracted: {total_positions_extracted}")
    if game_count > 0:
        avg_positions = total_positions_extracted / game_count
        print(f"  Average positions per game: {avg_positions:.2f}")
    print("=" * 80)


def evaluate_position_with_stockfish(fen: str, engine: chess.engine.SimpleEngine, 
                                     depth: int = 10, 
                                     score_min: int = -10000,
                                     score_max: int = 10000) -> Optional[int]:
    """
    Evaluate a position using Stockfish
    
    Args:
        fen: FEN string of the position
        engine: chess.engine.SimpleEngine instance
        depth: Search depth for evaluation
        score_min: Minimum score to clip to (in centipawns)
        score_max: Maximum score to clip to (in centipawns)
    
    Returns:
        Evaluation score in centipawns (positive = white advantage), clipped to [score_min, score_max]
    """
    try:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info['score']
        
        score_white = score.pov(chess.WHITE)
        
        if score_white.is_mate():
            mate_score = score_white.mate()
            raw_score = score_max if mate_score > 0 else score_min
        else:
            raw_score = int(score_white.score())
        
        # Clip the score to the specified range
        clipped_score = max(score_min, min(score_max, raw_score))
        return clipped_score
    
    except Exception as e:
        print(f"Error evaluating position {fen}: {e}")
        return None


def process_positions_batch(positions_batch: list, engine_path: str, depth: int = 10,
                            score_min: int = -10000, score_max: int = 10000) -> list:
    """
    Process a batch of positions with Stockfish

    Args:
        positions_batch: List of FEN strings
        engine_path: Path to Stockfish executable
        depth: Search depth for evaluation
        score_min: Minimum score to clip to (in centipawns)
        score_max: Maximum score to clip to (in centipawns)

    Returns:
        List of (fen, eval) tuples
    """
    results = []
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            for fen in positions_batch:
                eval_score = evaluate_position_with_stockfish(fen, engine, depth, score_min, score_max)
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
    positions_per_game=None,
    max_positions=None,
    num_workers=None,
    batch_size=None,
    download_mode=None,
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
        positions_per_game: Positions to sample per game (defaults to config.download_positions_per_game or 10)
        max_positions: Maximum positions to evaluate (None = all)
        num_workers: Number of parallel workers for evaluation (defaults to config.download_num_workers or 4)
        batch_size: Batch size for parallel processing (defaults to config.download_batch_size or 100)
        download_mode: Download mode - 'streaming' or 'direct' (defaults to config.download_mode or 'streaming')
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
        num_workers=num_workers,
        batch_size=batch_size,
        mode=download_mode,
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
    num_workers = params['num_workers']
    batch_size = params['batch_size']
    download_mode = params['mode']
    skip_redownload = params.get('skip_redownload', True)

    os.makedirs(output_dir, exist_ok=True)

    # Extract year/month early if possible, otherwise will be determined later
    if year is None or month is None:
        import re
        # Try to detect from existing files first
        for filename in os.listdir(output_dir):
            match = re.search(r'lichess_(?:positions|evaluated)_(\d{4})-(\d{2})\.jsonl', filename)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                break
        # If not found in existing files, use auto-detection
        if year is None or month is None:
            year, month = _get_latest_lichess_database()

    # Check if evaluated file already exists
    evaluated_path = os.path.join(output_dir, f'lichess_evaluated_{year}-{month:02d}.jsonl')
    if os.path.exists(evaluated_path):
        print(f"✓ Found existing evaluated file: {evaluated_path}")
        print(f"  Skipping extraction and evaluation. Using existing file.")
        return evaluated_path

    # Check if raw positions file exists
    raw_positions_path = os.path.join(output_dir, f'lichess_positions_{year}-{month:02d}.jsonl')
    if os.path.exists(raw_positions_path):
        print(f"✓ Found existing raw positions file: {raw_positions_path}")
        print(f"  Loading positions from file (skipping extraction)...")
        positions = []
        with open(raw_positions_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                positions.append(data['fen'])
        print(f"  Loaded {len(positions)} positions from file")
    else:
        # Need to extract positions from PGN
        print("=" * 80)
        print("Downloading Lichess database (simplified workflow)")
        print("=" * 80)
        pgn_path = stream_lichess_database(
            output_dir=output_dir,
            year=year,
            month=month,
            rated_only=rated_only,
            max_games=max_games,
            download_mode=download_mode,
            skip_redownload=skip_redownload,
            config=config
        )

        print("\n" + "=" * 80)
        print("Extracting positions from games")
        print("=" * 80)

        positions = []
        for fen in extract_positions_from_pgn(
            pgn_path,
            max_games=max_games,
            positions_per_game=positions_per_game,
            config=config
        ):
            positions.append(fen)
            if max_positions and len(positions) >= max_positions:
                break
        
        print(f"Extracted {len(positions)} positions from {max_games or 'all'} games")
        
        # Save extracted positions (before evaluation)
        print("\n" + "=" * 80)
        print("Saving extracted positions")
        print("=" * 80)
        
        # Extract year/month from filename if not already set
        if year is None or month is None:
            match = re.search(r'(\d{4})-(\d{2})', os.path.basename(pgn_path))
            if match:
                year, month = int(match.group(1)), int(match.group(2))
            else:
                # Use auto-detection if can't extract from filename
                year, month = _get_latest_lichess_database()
        
        # Update raw_positions_path with correct year/month
        raw_positions_path = os.path.join(output_dir, f'lichess_positions_{year}-{month:02d}.jsonl')
        with open(raw_positions_path, 'w') as f:
            for fen in positions:
                json.dump({'fen': fen}, f)
                f.write('\n')
        
        print(f"Saved {len(positions)} raw positions to: {raw_positions_path}")
        print(f"Output format: JSONL")
    
    print("\n" + "=" * 80)
    print("Evaluating positions with Stockfish")
    print("=" * 80)
    
    evaluated_positions = []
    batches = []
    for i in range(0, len(positions), batch_size):
        batch = positions[i:i+batch_size]
        batches.append(batch)
    
    print(f"Processing {len(batches)} batches with {num_workers} workers...")
    
    # Get score clipping parameters from config
    score_min = config.eval_score_min
    score_max = config.eval_score_max

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(process_positions_batch, batch, stockfish_path, depth, score_min, score_max)
            futures.append(future)

        with tqdm(total=len(futures), desc="Batch progress", unit="batch") as pbar:
            for future in as_completed(futures):
                results = future.result()
                evaluated_positions.extend(results)
                pbar.update(1)

    print(f"\nEvaluated {len(evaluated_positions)} positions")
    
    print("\n" + "=" * 80)
    print("Saving evaluated results")
    print("=" * 80)
    
    # Save evaluated positions as JSONL format
    output_path = os.path.join(output_dir, f'lichess_evaluated_{year}-{month:02d}.jsonl')
    with open(output_path, 'w') as f:
        for fen, eval_score in evaluated_positions:
            json.dump({'fen': fen, 'eval': eval_score}, f)
            f.write('\n')
    
    print(f"Saved {len(evaluated_positions)} positions to: {output_path}")
    print(f"Output format: JSONL")
    
    return output_path


