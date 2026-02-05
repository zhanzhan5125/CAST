import requests
import base64
from util.remove_comments import remove_comments_and_docstrings


def get_file_content(lang:str, repo: str, path: str, sha: str = None):
    github_token = "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"
    headers = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    owner, repo_name = repo.split('/')

    if sha is None:
        # 获取最新版本的内容
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            content = resp.json()
            file_content = base64.b64decode(content['content']).decode('utf-8')
            return file_content
        else:
            raise Exception(f"Failed to fetch file: {resp.status_code} - {resp.text}")
    else:
        # 先查找 blob sha
        tree_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{sha}?recursive=1"
        tree_resp = requests.get(tree_url, headers=headers)
        if tree_resp.status_code != 200:
            raise Exception(f"Failed to get tree: {tree_resp.status_code} - {tree_resp.text}")

        tree_data = tree_resp.json().get('tree', [])
        file_blob_sha = None
        for item in tree_data:
            if item['path'] == path and item['type'] == 'blob':
                file_blob_sha = item['sha']
                break

        if not file_blob_sha:
            raise Exception("File not found in tree.")

        # 获取 blob 内容
        blob_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/blobs/{file_blob_sha}"
        blob_resp = requests.get(blob_url, headers=headers)
        if blob_resp.status_code != 200:
            raise Exception(f"Failed to get blob: {blob_resp.status_code} - {blob_resp.text}")

        blob_data = blob_resp.json()
        file_content = base64.b64decode(blob_data['content']).decode('utf-8')
        return remove_comments_and_docstrings(file_content, lang)


# 示例使用
if __name__ == "__main__":
    data = {
        "repo": "Qiskit/qiskit-terra",
        "path": "qiskit/tools/qcvv/fitters.py",
        "sha": "d4f58d903bc96341b816f7c35df936d6421267d1"
    }
    content = get_file_content(lang='python', repo=data["repo"], path=data["path"], sha=data['sha'])
    # print(content)
    print(remove_comments_and_docstrings(content, "python"))
