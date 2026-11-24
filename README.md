# OrangeOil

> Investing always involves risk, as market fluctuations can lead to potential losses of capital.

## Installation

```bash
git clone https://github.com/Narthanan2000/OrangeOil.git
cd OrangeOil
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

| API | LINK |
|---|---|
| `EIA_API_KEY` | https://www.eia.gov/opendata/register.php |
| `FIRMS_MAP_KEY` | https://firms.modaps.eosdis.nasa.gov/api/map_key/ |
| `AISSTREAM_API_KEY` | https://aisstream.io |

## Usage

```bash
python main.py firms
python main.py ais --seconds 0
python main.py map
python main.py train
python main.py forecast
```

## Credits
Code developed with the assistance of Claude (Anthropic).


## License

MIT

