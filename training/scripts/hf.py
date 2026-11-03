
import click
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm
from pathlib import Path, PurePosixPath
import time
import os.path as osp


@click.group()
@click.version_option()
def cli():
    """
    hf script
    """

@cli.command('upload')
@click.argument('directory_or_path')
@click.option('--token', default=None)
@click.option('--repo_id')
@click.option('--suffix', default='.7z,.zip,.zip.*,.tar.zst.*,.json,.safetensors,.pt,.pkl,.bin')
@click.option('--repo_type', default='dataset')
def dump_framedir_list(directory_or_path, token, repo_id, suffix, repo_type):

    api = HfApi()
    suffix_lst = suffix.split(',')
    files = []
    if osp.isdir(directory_or_path):
        for suffix in suffix_lst:
            files += [str(p) for p in Path(directory_or_path).rglob('*'+suffix)]
    else:
        assert(osp.isfile(directory_or_path))
        files = [directory_or_path]
        directory_or_path = osp.dirname(directory_or_path)

    root_dir = str(PurePosixPath(directory_or_path))
    skip = True
    for p in tqdm(files):
        path_in_repo = str(PurePosixPath(p)).replace(root_dir+'/', '')
        # if osp.basename(p) == 'chunk01.zip.007':
        #     skip = False
        #     continue
        # if skip:
        #     continue
        while True:
            try:
                api.upload_file(
                    repo_id=repo_id,
                    path_or_fileobj=p,
                    path_in_repo=path_in_repo,
                    # run_as_future=True,
                    token=token,
                    repo_type=repo_type
                )
                break
            except HfHubHTTPError as e:
                print(e)
                time.sleep(180)


@cli.command('download')
@click.option('--token', default=None)
@click.option('--repo_id')
@click.option('--files')
@click.option('--save_dir')
@click.option('--suffix', default='.7z,.zip,.zip.*,.tar.zst.*')
@click.option('--repo_type', default='dataset')
def dump_framedir_list(token, repo_id, files, save_dir, suffix, repo_type):

    api = HfApi()
    suffix_lst = suffix.split(',')
    files = files.split(',')

    for p in tqdm(files):
        subfolder = osp.dirname(p)
        if subfolder == '':
            subfolder = None
        while True:
            try:
                api.hf_hub_download(
                    repo_id=repo_id,
                    local_dir=save_dir,
                    filename=p,
                    subfolder=subfolder,
                    token=token,
                    repo_type=repo_type
                )
                break
            except HfHubHTTPError as e:
                print(e)
                time.sleep(180)

if __name__ == '__main__':
    cli()