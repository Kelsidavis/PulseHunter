async function loadDetections() {
  const response = await fetch('reports/');
  const text = await response.text();
  const files = [...text.matchAll(/href="(.*?\.json)"/g)].map(m => m[1]);

  const detections = [];
  for (let file of files) {
    try {
      const res = await fetch('reports/' + file);
      const json = await res.json();
      detections.push(...json.detections);
    } catch (e) { console.warn('Bad file:', file); }
  }

  const aladin = A.aladin('#aladin-lite-div', { survey: 'P/DSS2/color', fov: 180 });
  detections.forEach(d => {
    if (!d.ra_deg || !d.dec_deg) return;
    const c = d.confidence || 0;
    const color = c > 0.8 ? 'green' : c > 0.5 ? 'yellow' : 'red';
    aladin.addCatalog(A.catalog({ sourceSize: 8, color, shape: 'circle' }));
    aladin.catalogs[0].addSources([A.source(d.ra_deg, d.dec_deg, { name: d.observer || 'Unknown' })]);

    const row = document.createElement('tr');
    row.innerHTML = `<td>${d.observer || 'Unknown'}</td><td>${d.ra_deg}</td><td>${d.dec_deg}</td>
                     <td>${d.timestamp_utc || ''}</td><td>${(d.confidence * 100).toFixed(0)}%</td>
                     <td>${d.match_name || '—'}</td>`;
    document.querySelector('#detectionTable tbody').appendChild(row);
  });
}
loadDetections();
