/** Vessel archive management with multiple description prototypes per hull number. */
const API = '/api/ships';
let allShips = [];
let editingPrototypeId = '';
let uploadFile = null;

function showToast(message, type = 'success') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = 'toast ' + type + ' show';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('show'), 3000);
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '请求失败');
  return data;
}

function escHtml(value) {
  const div = document.createElement('div');
  div.textContent = String(value || '');
  return div.innerHTML;
}

async function loadShips() {
  try {
    const [shipsData, statsData] = await Promise.all([apiFetch(API), apiFetch(API + '/stats')]);
    allShips = shipsData.ships || [];
    document.getElementById('totalCount').textContent = statsData.total_ships;
    document.getElementById('prototypeCount').textContent = statsData.total_prototypes;
    document.getElementById('backendType').textContent = String(statsData.backend || '').toUpperCase();
    renderTable(allShips);
  } catch (error) {
    showToast('加载失败: ' + error.message, 'error');
  }
}

function renderTable(ships) {
  const tbody = document.getElementById('shipTable');
  if (!ships.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-msg">暂无结构原型</td></tr>';
    return;
  }
  tbody.innerHTML = ships.map((ship, index) =>
    '<tr>' +
      '<td><code>' + escHtml(ship.prototype_id) + '</code></td>' +
      '<td><span class="hull-num">' + escHtml(ship.hull_number) + '</span></td>' +
      '<td>' + escHtml(ship.description) + '</td>' +
      '<td><div class="actions">' +
        '<button class="btn btn-outline btn-sm" data-action="edit" data-index="' + index + '">编辑</button>' +
        '<button class="btn btn-danger btn-sm" data-action="delete" data-index="' + index + '">删除</button>' +
      '</div></td>' +
    '</tr>'
  ).join('');
  tbody.querySelectorAll('button[data-action]').forEach(button => {
    const ship = ships[Number(button.dataset.index)];
    if (button.dataset.action === 'edit') {
      button.addEventListener('click', () => openEditModal(ship.prototype_id, ship.hull_number, ship.description));
    } else {
      button.addEventListener('click', () => deletePrototype(ship.prototype_id));
    }
  });
}

document.getElementById('searchInput').addEventListener('input', function () {
  const query = this.value.toLowerCase().trim();
  renderTable(query ? allShips.filter(ship =>
    ship.prototype_id.toLowerCase().includes(query) ||
    ship.hull_number.toLowerCase().includes(query) ||
    ship.description.toLowerCase().includes(query)
  ) : allShips);
});

function openAddModal() {
  editingPrototypeId = '';
  document.getElementById('modalTitle').textContent = '新增结构原型';
  document.getElementById('modalPrototypeId').value = '';
  document.getElementById('modalHullNumber').value = '';
  document.getElementById('modalHullNumber').disabled = false;
  document.getElementById('modalDescription').value = '';
  document.getElementById('shipModal').classList.add('active');
}

function openEditModal(prototypeId, hullNumber, description) {
  editingPrototypeId = prototypeId;
  document.getElementById('modalTitle').textContent = '编辑结构原型 ' + prototypeId;
  document.getElementById('modalPrototypeId').value = prototypeId;
  document.getElementById('modalHullNumber').value = hullNumber;
  document.getElementById('modalHullNumber').disabled = true;
  document.getElementById('modalDescription').value = description;
  document.getElementById('shipModal').classList.add('active');
}

function closeModal() { document.getElementById('shipModal').classList.remove('active'); }

async function submitShip() {
  const hullNumber = document.getElementById('modalHullNumber').value.trim();
  const description = document.getElementById('modalDescription').value.trim();
  if (!hullNumber || !description) { showToast('舷号和结构描述不能为空', 'error'); return; }
  try {
    if (editingPrototypeId) {
      await apiFetch(API + '/prototypes/' + encodeURIComponent(editingPrototypeId), {
        method: 'PUT', body: JSON.stringify({ description }),
      });
      showToast('原型更新成功');
    } else {
      const result = await apiFetch(API, {
        method: 'POST', body: JSON.stringify({ hull_number: hullNumber, description }),
      });
      showToast(result.message);
    }
    closeModal();
    loadShips();
  } catch (error) { showToast(error.message, 'error'); }
}

async function deletePrototype(prototypeId) {
  if (!confirm('确定删除结构原型 ' + prototypeId + '？')) return;
  try {
    await apiFetch(API + '/prototypes/' + encodeURIComponent(prototypeId), { method: 'DELETE' });
    showToast('原型删除成功');
    loadShips();
  } catch (error) { showToast(error.message, 'error'); }
}

function openBulkModal() {
  document.getElementById('bulkInput').value = '';
  document.getElementById('bulkModal').classList.add('active');
}
function closeBulkModal() { document.getElementById('bulkModal').classList.remove('active'); }
async function submitBulk() {
  const raw = document.getElementById('bulkInput').value.trim();
  if (!raw) { showToast('请输入数据', 'error'); return; }
  let ships;
  try { ships = JSON.parse(raw); } catch { showToast('JSON 格式错误', 'error'); return; }
  try {
    const result = await apiFetch(API + '/bulk', { method: 'POST', body: JSON.stringify({ ships }) });
    showToast(result.message);
    closeBulkModal();
    loadShips();
  } catch (error) { showToast(error.message, 'error'); }
}

function openUploadModal() {
  uploadFile = null;
  document.getElementById('uploadFilename').textContent = '';
  document.getElementById('uploadPreview').style.display = 'none';
  document.getElementById('uploadPreview').src = '';
  document.getElementById('recognizeResult').classList.remove('show');
  document.getElementById('btnRecognize').style.display = '';
  document.getElementById('btnConfirmAdd').style.display = 'none';
  document.getElementById('btnRecognize').disabled = false;
  document.getElementById('btnConfirmAdd').disabled = false;
  document.getElementById('recHullNumber').value = '';
  document.getElementById('recDescription').value = '';
  document.getElementById('recExistsWarn').style.display = 'none';
  document.getElementById('uploadModal').classList.add('active');
}

function closeUploadModal() {
  document.getElementById('uploadModal').classList.remove('active');
  uploadFile = null;
}

function handleUploadFile(file) {
  if (!file.type.startsWith('image/')) { showToast('请选择图片文件', 'error'); return; }
  if (file.size > 20 * 1024 * 1024) { showToast('文件过大，请上传 20MB 以内的图片', 'error'); return; }
  uploadFile = file;
  document.getElementById('uploadFilename').textContent = file.name;
  const reader = new FileReader();
  reader.onload = event => {
    const preview = document.getElementById('uploadPreview');
    preview.src = event.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
  document.getElementById('recognizeResult').classList.remove('show');
  document.getElementById('btnRecognize').style.display = '';
  document.getElementById('btnConfirmAdd').style.display = 'none';
}

document.getElementById('uploadFileInput').addEventListener('change', event => {
  if (event.target.files.length) handleUploadFile(event.target.files[0]);
});
const uploadZone = document.getElementById('uploadZone');
uploadZone.addEventListener('dragover', event => { event.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', event => { event.preventDefault(); uploadZone.classList.remove('dragover'); });
uploadZone.addEventListener('drop', event => {
  event.preventDefault();
  uploadZone.classList.remove('dragover');
  if (event.dataTransfer.files.length) handleUploadFile(event.dataTransfer.files[0]);
});

async function doRecognize() {
  if (!uploadFile) { showToast('请先选择图片', 'error'); return; }
  const button = document.getElementById('btnRecognize');
  const original = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<span class="loading-spinner"></span> 识别中…';
  try {
    const formData = new FormData();
    formData.append('file', uploadFile);
    const response = await fetch(API + '/recognize', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '识别失败');
    const result = data.data;
    document.getElementById('recHullNumber').value = result.hull_number || '';
    document.getElementById('recDescription').value = result.description || '';
    const warning = document.getElementById('recExistsWarn');
    if (result.already_exists) {
      warning.style.display = 'block';
      warning.textContent = '该舷号已有 ' + result.existing_prototype_count + ' 条原型；确认后将追加新原型，不覆盖旧数据。';
    } else {
      warning.style.display = 'none';
    }
    document.getElementById('recognizeResult').classList.add('show');
    document.getElementById('btnConfirmAdd').style.display = '';
  } catch (error) {
    showToast('识别失败: ' + error.message, 'error');
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

async function doConfirmAdd() {
  const hullNumber = document.getElementById('recHullNumber').value.trim();
  const description = document.getElementById('recDescription').value.trim();
  if (!hullNumber) { showToast('舷号不能为空', 'error'); return; }
  if (!description) { showToast('结构描述不能为空', 'error'); return; }
  const button = document.getElementById('btnConfirmAdd');
  const original = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<span class="loading-spinner"></span> 提交中…';
  try {
    const result = await apiFetch(API, {
      method: 'POST', body: JSON.stringify({ hull_number: hullNumber, description }),
    });
    showToast(result.message);
    closeUploadModal();
    loadShips();
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

loadShips();
