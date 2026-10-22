from pathlib import Path
import pytest
from visionlab.platform_support import model_install_command


@pytest.mark.parametrize('system,machine,torch,vision,cpu_index', [
    ('Windows','AMD64','2.7.0','0.22.0',True),
    ('Darwin','arm64','2.7.0','0.22.0',False),
    ('Darwin','x86_64','2.2.2','0.17.2',False),
    ('Linux','x86_64','2.7.0','0.22.0',True),
])
def test_platform_model_pair(system,machine,torch,vision,cpu_index):
    command=model_install_command(system,machine)
    assert f'torch=={torch}' in command
    assert f'torchvision=={vision}' in command
    assert ('--index-url' in command)==cpu_index


@pytest.mark.parametrize('system,machine',[('Windows','ARM64'),('Windows','x86'),('Other','x86_64')])
def test_unsupported_architecture_is_clear(system,machine):
    with pytest.raises(ValueError):model_install_command(system,machine)


def make_dataset(tmp_path, duplicate=False):
    yaml=pytest.importorskip('yaml')
    from PIL import Image
    for split,color in [('train','red'),('val','red' if duplicate else 'blue')]:
        folder=tmp_path/split;folder.mkdir()
        Image.new('RGB',(8,8),color).save(folder/'sample.png')
    path=tmp_path/'dataset.yaml'
    path.write_text(yaml.safe_dump({'train':'train','val':'val','names':['object']}))
    return path


def test_training_rejects_duplicate_split(tmp_path):
    from visionlab.train import validate_dataset
    path=make_dataset(tmp_path,True)
    with pytest.raises(ValueError,match='duplicate'):validate_dataset(path)


def test_training_dataset_paths_and_counts(tmp_path):
    from visionlab.train import validate_dataset
    config,counts=validate_dataset(make_dataset(tmp_path))
    assert counts=={'train':1,'val':1}
    assert Path(config['train']).is_absolute()


def test_dataset_cannot_execute_download_code(tmp_path):
    from visionlab.train import validate_dataset
    path=make_dataset(tmp_path)
    with path.open('a') as stream:stream.write('\ndownload: print(123)\n')
    with pytest.raises(ValueError,match='Download scripts'):validate_dataset(path)
