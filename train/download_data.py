"""
Download Lichess dataset and evaluate positions with Stockfish

This script downloads chess games from the Lichess database, extracts positions,
and evaluates them using Stockfish to create a training dataset for NNUE models.

Requirements:
    - Stockfish installed and available in PATH (or provide path with --stockfish-path)
    - Python packages: requests, zstandard, python-chess, tqdm

Example usage:
    # Download and process 1000 games from January 2024 (streaming mode - default)
    python download_data.py --max-games 1000 --year 2024 --month 1
    
    # Use direct download mode (downloads full file first)
    python download_data.py --max-games 1000 --year 2024 --month 1 --download-mode direct
    
    # Use custom Stockfish path and higher depth
    python download_data.py --stockfish-path /usr/local/bin/stockfish --depth 15
    
    # Process more positions with more workers
    python download_data.py --max-games 5000 --positions-per-game 20 --num-workers 8
    
    # Save as CSV format
    python download_data.py --output-format csv --max-games 1000
    
Download Modes:
    - streaming (default): Filters games on-the-fly without saving full database file.
                          Saves bandwidth and disk space but cannot be interrupted/resumed.
    - direct: Downloads complete database file first, then filters it.
             Uses more disk space but allows resuming and keeps full database for reuse.
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


def stream_lichess_database(output_dir: Optional[str] = None, year: Optional[int] = None, 
                            month: Optional[int] = None, rated_only: Optional[bool] = None,
                            max_games: Optional[int] = None, max_games_searched: Optional[int] = None, 
                            min_elo: Optional[int] = None, max_elo: Optional[int] = None, 
                            start_date: Optional[str] = None, end_date: Optional[str] = None, 
                            time_control: Optional[str] = None, min_moves: Optional[int] = None, 
                            max_moves: Optional[int] = None, result_filter: Optional[str] = None, 
                            download_mode: Optional[str] = None,
                            skip_filter: Optional[bool] = None,
                            skip_redownload: Optional[bool] = None,
                            config: Optional[TrainingConfig] = None) -> str:
    """
    Download Lichess database and filter games
    
    Args:
        output_dir: Directory to save filtered games (defaults to config.download_output_dir or 'data')
        year: Year of the database (defaults to config.download_year or 2024)
        month: Month of the database (defaults to config.download_month or 1)
        rated_only: Download rated games only (defaults to config.download_rated_only or True)
        max_games: Maximum matching games to save (stops when reached)
        max_games_searched: Maximum total games to search (stops regardless of matches)
        min_elo, max_elo, start_date, end_date, time_control, min_moves, max_moves, result_filter: Filters
        download_mode: Download mode - 'streaming' (filter on-the-fly) or 'direct' (download full file first)
        skip_filter: Skip filtering entirely - use all games (defaults to config.download_skip_filter or False)
        skip_redownload: Skip re-downloading if file already exists (defaults to config.download_skip_redownload or False)
        config: TrainingConfig instance (optional, for defaults)
    
    Returns:
        Path to filtered PGN file
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
    download_mode = params['mode']
    skip_filter = params.get('skip_filter', False)
    skip_redownload = params.get('skip_redownload', False)
    
    # Default to streaming mode if not specified
    if download_mode is None:
        download_mode = 'streaming'
    
    if download_mode not in ['streaming', 'direct']:
        raise ValueError(f"Invalid download_mode: {download_mode}. Must be 'streaming' or 'direct'")
    
    os.makedirs(output_dir, exist_ok=True)
    
    month_str = f"{month:02d}"
    date_str = f"{year}-{month_str}"
    
    if rated_only:
        filename = f"lichess_db_standard_rated_{date_str}.pgn.zst"
    else:
        filename = f"lichess_db_standard_{date_str}.pgn.zst"
    
    url = f"https://database.lichess.org/standard/{filename}"
    output_path = os.path.join(output_dir, f"filtered_{date_str}.pgn")
    compressed_path = os.path.join(output_dir, filename)
    
    # Check if we should skip re-downloading
    if skip_redownload:
        if download_mode == 'direct' and os.path.exists(compressed_path):
            print(f"Found existing download: {compressed_path}")
            print(f"Skipping re-download (file size: {os.path.getsize(compressed_path) / (1024**3):.2f} GB)")
            print("Set skip_redownload=False or delete the file to re-download")
            # Skip to filtering only
            return _filter_existing_download(compressed_path, output_path, max_games, max_games_searched,
                                            min_elo, max_elo, start_date, end_date,
                                            time_control, min_moves, max_moves, result_filter,
                                            skip_filter=skip_filter)
        elif os.path.exists(output_path):
            print(f"Found existing filtered file: {output_path}")
            print("Skipping download and filtering")
            print("Set skip_redownload=False or delete the file to re-process")
            return output_path
    
    if skip_filter:
        print(f"Skipping filtering - will use all games from {filename}")
        # Set filters to None when skipping
        min_elo = max_elo = start_date = end_date = None
        time_control = min_moves = max_moves = result_filter = None
    
    if download_mode == 'direct':
        print(f"Downloading full database: {filename}...")
        print(f"URL: {url}")
        print("This will download the complete file before filtering")
        return _direct_download_and_filter(url, filename, output_path, output_dir,
                                          max_games, max_games_searched,
                                          min_elo, max_elo, start_date, end_date,
                                          time_control, min_moves, max_moves, result_filter)
    else:
        print(f"Streaming and filtering games from {filename}...")
        print(f"URL: {url}")
        print("This will only save games matching your filters (no full download)")
        return _streaming_download_and_filter(url, output_path,
                                             max_games, max_games_searched,
                                             min_elo, max_elo, start_date, end_date,
                                             time_control, min_moves, max_moves, result_filter)


def _filter_existing_download(compressed_path: str, output_path: str,
                              max_games: Optional[int], max_games_searched: Optional[int],
                              min_elo: Optional[int], max_elo: Optional[int],
                              start_date: Optional[str], end_date: Optional[str],
                              time_control: Optional[str], min_moves: Optional[int],
                              max_moves: Optional[int], result_filter: Optional[str],
                              skip_filter: bool = False) -> str:
    """
    Filter an already downloaded compressed file
    """
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard package is required. Install it with: pip install zstandard"
        )
    
    if skip_filter:
        print("Skipping filtering - copying all games from existing downloaded file...")
        # When skipping filter, just copy all games without filtering
        min_elo = max_elo = start_date = end_date = None
        time_control = min_moves = max_moves = result_filter = None
    else:
        print("Filtering games from existing downloaded file...")
    
    # Filter the downloaded file
    dctx = zstd.ZstdDecompressor()
    games_found = 0
    games_written = 0
    
    with open(compressed_path, 'rb') as compressed_file:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            pbar = tqdm(desc="Filtering games", unit=' games', unit_scale=False)
            
            try:
                with dctx.stream_reader(compressed_file) as decompressor:
                    pgn_file = io.TextIOWrapper(decompressor, encoding='utf-8', errors='ignore')
                    
                    while True:
                        if max_games and games_written >= max_games:
                            print(f"\nFound {games_written} matching games, stopping")
                            break
                        
                        if max_games_searched and games_found >= max_games_searched:
                            print(f"\nSearched {games_found} games (limit reached), stopping")
                            break
                        
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                        
                        games_found += 1
                        pbar.update(1)
                        
                        # If skip_filter is True, include all games; otherwise check filters
                        if skip_filter or should_include_game(game, min_elo, max_elo, start_date,
                                              end_date, time_control, min_moves,
                                              max_moves, result_filter):
                            exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                            game_text = game.accept(exporter)
                            out_file.write(game_text)
                            out_file.write('\n\n')
                            games_written += 1
                            pbar.set_postfix({'matched': games_written})
            
            finally:
                pbar.close()
    
    print(f"\nProcessed {games_found} total games, saved {games_written} matching games")
    print(f"Saved to: {output_path}")
    
    return output_path


def _direct_download_and_filter(url: str, filename: str, output_path: str, output_dir: str,
                                 max_games: Optional[int], max_games_searched: Optional[int],
                                 min_elo: Optional[int], max_elo: Optional[int],
                                 start_date: Optional[str], end_date: Optional[str],
                                 time_control: Optional[str], min_moves: Optional[int],
                                 max_moves: Optional[int], result_filter: Optional[str]) -> str:
    """
    Direct download mode: Download complete file first, then filter
    """
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard package is required. Install it with: pip install zstandard"
        )
    
    # Download the full file
    compressed_path = os.path.join(output_dir, filename)
    
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
    print("Filtering games from downloaded file...")
    
    # Now filter the downloaded file
    dctx = zstd.ZstdDecompressor()
    games_found = 0
    games_written = 0
    
    with open(compressed_path, 'rb') as compressed_file:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            pbar = tqdm(desc="Filtering games", unit=' games', unit_scale=False)
            
            try:
                with dctx.stream_reader(compressed_file) as decompressor:
                    pgn_file = io.TextIOWrapper(decompressor, encoding='utf-8', errors='ignore')
                    
                    while True:
                        if max_games and games_written >= max_games:
                            print(f"\nFound {games_written} matching games, stopping")
                            break
                        
                        if max_games_searched and games_found >= max_games_searched:
                            print(f"\nSearched {games_found} games (limit reached), stopping")
                            break
                        
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                        
                        games_found += 1
                        pbar.update(1)
                        
                        if should_include_game(game, min_elo, max_elo, start_date,
                                              end_date, time_control, min_moves,
                                              max_moves, result_filter):
                            exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                            game_text = game.accept(exporter)
                            out_file.write(game_text)
                            out_file.write('\n\n')
                            games_written += 1
                            pbar.set_postfix({'matched': games_written})
            
            finally:
                pbar.close()
    
    print(f"\nProcessed {games_found} total games, saved {games_written} matching games")
    print(f"Saved to: {output_path}")
    print(f"\nNote: Full database file kept at: {compressed_path}")
    print(f"You can delete it to save space if no longer needed.")
    
    return output_path


def _streaming_download_and_filter(url: str, output_path: str,
                                   max_games: Optional[int], max_games_searched: Optional[int],
                                   min_elo: Optional[int], max_elo: Optional[int],
                                   start_date: Optional[str], end_date: Optional[str],
                                   time_control: Optional[str], min_moves: Optional[int],
                                   max_moves: Optional[int], result_filter: Optional[str]) -> str:
    """
    Streaming download mode: Filter games on-the-fly without saving full file
    """
    try:
        import zstandard as zstd
    except ImportError:
        raise ImportError(
            "zstandard package is required. Install it with: pip install zstandard"
        )
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    dctx = zstd.ZstdDecompressor()
    
    games_found = 0
    games_written = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        pbar = tqdm(
            desc="Streaming games",
            unit=' games',
            unit_scale=False,
        )
        
        try:
            decompressor = dctx.stream_reader(response.raw)
            pgn_file = io.TextIOWrapper(decompressor, encoding='utf-8', errors='ignore')
            
            while True:
                if max_games and games_written >= max_games:
                    print(f"\nFound {games_written} matching games, stopping download")
                    break
                
                if max_games_searched and games_found >= max_games_searched:
                    print(f"\nSearched {games_found} games (limit reached), stopping download")
                    break
                
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                games_found += 1
                pbar.update(1)
                
                if should_include_game(game, min_elo, max_elo, start_date,
                                      end_date, time_control, min_moves,
                                      max_moves, result_filter):
                    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                    game_text = game.accept(exporter)
                    out_file.write(game_text)
                    out_file.write('\n\n')
                    games_written += 1
                    pbar.set_postfix({'matched': games_written})
        
        finally:
            pbar.close()
    
    print(f"\nProcessed {games_found} total games, saved {games_written} matching games")
    print(f"Saved to: {output_path}")
    return output_path


def should_include_game(game: chess.pgn.Game, min_elo: Optional[int] = None, 
                        max_elo: Optional[int] = None, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None, time_control: Optional[str] = None, 
                        min_moves: Optional[int] = None, max_moves: Optional[int] = None, 
                        result_filter: Optional[str] = None) -> bool:
    """
    Check if a game should be included based on filters
    
    Args:
        game: chess.pgn.Game object
        min_elo: Minimum ELO rating (either player)
        max_elo: Maximum ELO rating (either player)
        start_date: Start date filter (YYYY-MM-DD or YYYY.MM.DD format)
        end_date: End date filter (YYYY-MM-DD or YYYY.MM.DD format)
        time_control: Time control filter (e.g., "180+2")
        min_moves: Minimum number of moves
        max_moves: Maximum number of moves
        result_filter: Result filter ('1-0', '0-1', '1/2-1/2', or None for all)
    
    Returns:
        True if game should be included, False otherwise
    """
    try:
        white_elo = game.headers.get('WhiteElo', '')
        black_elo = game.headers.get('BlackElo', '')
        
        if white_elo and white_elo.isdigit():
            white_elo = int(white_elo)
        else:
            white_elo = None
        
        if black_elo and black_elo.isdigit():
            black_elo = int(black_elo)
        else:
            black_elo = None
        
        if min_elo is not None:
            if white_elo is None or black_elo is None:
                return False
            if white_elo < min_elo and black_elo < min_elo:
                return False
        
        if max_elo is not None:
            if white_elo is None or black_elo is None:
                return False
            if white_elo > max_elo or black_elo > max_elo:
                return False
        
        date_str = game.headers.get('Date', '')
        if date_str and (start_date or end_date):
            try:
                if start_date:
                    start_normalized = start_date.replace('-', '.')
                    if date_str < start_normalized:
                        return False
                if end_date:
                    end_normalized = end_date.replace('-', '.')
                    if date_str > end_normalized:
                        return False
            except Exception:
                # Skip games with invalid date formats
                pass
        
        if time_control:
            game_time_control = game.headers.get('TimeControl', '')
            if game_time_control != time_control:
                return False
        
        if min_moves is not None or max_moves is not None:
            move_count = len(list(game.mainline_moves()))
            if min_moves is not None and move_count < min_moves:
                return False
            if max_moves is not None and move_count > max_moves:
                return False
        
        if result_filter:
            result = game.headers.get('Result', '')
            if result != result_filter:
                return False
        
        return True

    except Exception as e:
        # Log parsing errors but don't crash - just skip the game
        print(f"Warning: Error checking game filters: {e}")
        return False


def extract_positions_from_pgn(pgn_path: str, max_games: Optional[int] = None, 
                                positions_per_game: Optional[int] = None, 
                                min_ply: Optional[int] = None, max_ply: Optional[int] = None, 
                                min_elo: Optional[int] = None, max_elo: Optional[int] = None,
                                start_date: Optional[str] = None, end_date: Optional[str] = None, 
                                time_control: Optional[str] = None,
                                min_moves: Optional[int] = None, max_moves: Optional[int] = None, 
                                result_filter: Optional[str] = None,
                                subset_ratio: Optional[float] = None,
                                config: Optional[TrainingConfig] = None) -> Iterator[Tuple[str, Dict[str, str]]]:
    """
    Extract positions from PGN file with filtering
    
    Args:
        pgn_path: Path to PGN file (can be .zst compressed)
        max_games: Maximum number of games to process (None = all)
        positions_per_game: Number of positions to sample per game (defaults to config.download_positions_per_game or 10)
        min_ply: Minimum ply (half-move) to sample from (defaults to config.download_min_ply or 10)
        max_ply: Maximum ply to sample from (defaults to config.download_max_ply or 100)
        min_elo: Minimum ELO rating filter
        max_elo: Maximum ELO rating filter
        start_date: Start date filter (YYYY-MM-DD format)
        end_date: End date filter (YYYY-MM-DD format)
        time_control: Time control filter
        min_moves: Minimum moves filter
        max_moves: Maximum moves filter
        result_filter: Result filter
        subset_ratio: Read only a subset of games (e.g., 0.1 for 10%) (defaults to config.download_subset_ratio)
        config: TrainingConfig instance (optional, for defaults)
    
    Yields:
        Tuple of (fen, game_info) for each position
    """
    config, params = _resolve_config_params(
        config,
        positions_per_game=positions_per_game,
        min_ply=min_ply,
        max_ply=max_ply,
        subset_ratio=subset_ratio
    )
    
    positions_per_game = params['positions_per_game']
    min_ply = params['min_ply']
    max_ply = params['max_ply']
    subset_ratio = params.get('subset_ratio', None)
    if pgn_path.endswith('.zst'):
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError(
                "zstandard package is required for .zst files. "
                "Install it with: pip install zstandard"
            )
        
        dctx = zstd.ZstdDecompressor()
        with open(pgn_path, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                pgn_file = io.TextIOWrapper(reader, encoding='utf-8')
    else:
        pgn_file = open(pgn_path, 'r', encoding='utf-8', errors='ignore')
    
    game_count = 0
    
    try:
        while True:
            if max_games and game_count >= max_games:
                break
            
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            # Apply subset ratio - randomly skip games if ratio specified
            if subset_ratio is not None:
                if random.random() > subset_ratio:
                    continue
            
            if not should_include_game(game, min_elo, max_elo, start_date, end_date,
                                     time_control, min_moves, max_moves, result_filter):
                continue
            
            game_count += 1
            
            game_info = {
                'white': game.headers.get('White', ''),
                'black': game.headers.get('Black', ''),
                'result': game.headers.get('Result', ''),
                'elo_white': game.headers.get('WhiteElo', ''),
                'elo_black': game.headers.get('BlackElo', ''),
            }
            
            board = game.board()
            ply_count = 0

            # Collect all valid positions from this game
            valid_positions = []
            for move in game.mainline_moves():
                board.push(move)
                ply_count += 1

                if min_ply <= ply_count <= max_ply:
                    valid_positions.append((ply_count, board.fen()))

            # Randomly sample positions_per_game from valid positions
            # This ensures consistent sampling regardless of game length
            num_to_sample = min(positions_per_game, len(valid_positions))
            if num_to_sample > 0:
                sampled = random.sample(valid_positions, num_to_sample)
                for _, fen in sampled:
                    yield fen, game_info
            
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games...")
    
    finally:
        if not pgn_path.endswith('.zst'):
            pgn_file.close()


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
            for fen, game_info in positions_batch:
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
    min_elo=None,
    max_elo=None,
    start_date=None,
    end_date=None,
    time_control=None,
    min_moves=None,
    max_moves=None,
    result_filter=None,
    min_ply=None,
    max_ply=None,
    download_mode=None,
    skip_filter=None,
    skip_redownload=None,
    subset_ratio=None,
    config=None
):
    """
    Download Lichess database and process with Stockfish evaluations
    
    Args:
        output_dir: Directory to save output files (defaults to config.download_output_dir or 'data')
        year: Year of database to download (defaults to config.download_year or 2024)
        month: Month of database to download (defaults to config.download_month or 1)
        rated_only: Download rated games only (defaults to config.download_rated_only or True)
        stockfish_path: Path to Stockfish executable (defaults to config.stockfish_path or 'stockfish')
        depth: Stockfish search depth (defaults to config.download_depth or 10)
        max_games: Maximum games to process (defaults to config.download_max_games or 10000)
        positions_per_game: Positions to sample per game (defaults to config.download_positions_per_game or 10)
        max_positions: Maximum positions to evaluate (None = all)
        output_format: Output format ('csv', 'json', or 'jsonl') (defaults to config.download_output_format or 'jsonl')
        num_workers: Number of parallel workers for evaluation (defaults to config.download_num_workers or 4)
        batch_size: Batch size for parallel processing (defaults to config.download_batch_size or 100)
        min_ply: Minimum ply to sample from (defaults to config.download_min_ply or 10)
        max_ply: Maximum ply to sample from (defaults to config.download_max_ply or 100)
        download_mode: Download mode - 'streaming' (filter on-the-fly, default) or 'direct' (download full file first)
        skip_filter: Skip filtering entirely - use all games (defaults to config.download_skip_filter or False)
        skip_redownload: Skip re-downloading if file already exists (defaults to config.download_skip_redownload or False)
        subset_ratio: Read only a subset of games (e.g., 0.1 for 10%) (defaults to config.download_subset_ratio)
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
        min_ply=min_ply,
        max_ply=max_ply,
        mode=download_mode,
        skip_filter=skip_filter,
        skip_redownload=skip_redownload,
        subset_ratio=subset_ratio
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
    min_ply = params['min_ply']
    max_ply = params['max_ply']
    download_mode = params['mode']
    skip_filter = params.get('skip_filter', False)
    skip_redownload = params.get('skip_redownload', False)
    subset_ratio = params.get('subset_ratio', None)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Downloading Lichess database")
    print("=" * 80)
    pgn_path = stream_lichess_database(
        output_dir, year, month, rated_only, max_games, max_games_searched,
        min_elo, max_elo, start_date, end_date, time_control,
        min_moves, max_moves, result_filter, download_mode, skip_filter, skip_redownload, config
    )
    
    print("\n" + "=" * 80)
    print("Extracting positions from games")
    print("=" * 80)
    
    positions = []
    position_count = 0
    
    if subset_ratio is not None:
        print(f"Using subset ratio: {subset_ratio:.1%} of games")
    
    for fen, game_info in extract_positions_from_pgn(
        pgn_path, 
        max_games=None,
        positions_per_game=positions_per_game,
        min_ply=min_ply,
        max_ply=max_ply,
        min_elo=None,
        max_elo=None,
        start_date=None,
        end_date=None,
        time_control=None,
        min_moves=None,
        max_moves=None,
        result_filter=None,
        subset_ratio=subset_ratio,
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
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results = future.result()
            evaluated_positions.extend(results)
    
    print(f"Evaluated {len(evaluated_positions)} positions")
    
    print("\n" + "=" * 80)
    print("Saving results")
    print("=" * 80)
    
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
    parser.add_argument('--year', type=int, default=config.download_year,
                       help=f'Year of database to download (default: {config.download_year})')
    parser.add_argument('--month', type=int, default=config.download_month,
                       help=f'Month of database to download (1-12, default: {config.download_month})')
    parser.add_argument('--all-games', action='store_true',
                       help='Download all games (not just rated)')
    parser.add_argument('--stockfish-path', type=str, default=config.stockfish_path,
                       help=f'Path to Stockfish executable (default: {config.stockfish_path})')
    parser.add_argument('--depth', type=int, default=config.download_depth,
                       help=f'Stockfish search depth (default: {config.download_depth})')
    parser.add_argument('--max-games', type=int, default=config.download_max_games,
                       help=f'Maximum games to process (default: {config.download_max_games})')
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
                       help='Download mode: streaming (filter on-the-fly, saves bandwidth) or direct (download full file first) (default: streaming)')
    parser.add_argument('--skip-filter', action='store_true',
                       help='Skip filtering entirely - use all games from the database')
    parser.add_argument('--skip-redownload', action='store_true',
                       help='Skip re-downloading if file already exists')
    parser.add_argument('--subset-ratio', type=float, default=None,
                       help='Read only a subset of games (e.g., 0.1 for 10%%). None = all games')
    
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
        subset_ratio=args.subset_ratio,
        config=config
    )


if __name__ == '__main__':
    main()
