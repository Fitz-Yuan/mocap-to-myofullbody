// Links into the former one-page documentation (documentation/#<id>) go to the new page.
(function () {
  var map = {
 "allmodels": "validation/public-data/#allmodels",
 "bottom-line": "validation/summary/",
 "c3d-export": "pipeline/gap-filling/#c3d-export",
 "code": "reference/code/",
 "effort": "validation/case-study/#effort",
 "gaps": "pipeline/gap-filling/",
 "header": "",
 "kin-validation": "validation/inverse-dynamics/",
 "kinetics": "pipeline/kinetics/",
 "lomo": "validation/markers/",
 "models": "pipeline/models/",
 "next": "validation/summary/#known-limits",
 "opensim": "validation/opensim/",
 "opensim-setup": "reference/validation-scripts/#opensim-setup",
 "own-data": "getting-started/your-data/",
 "pig": "validation/public-data/#pig",
 "pipeline": "pipeline/overview/",
 "posture": "validation/joint-angles/#posture",
 "problems": "validation/case-study/#problems",
 "public": "validation/public-data/#public",
 "reading": "reference/reading/",
 "recovery": "pipeline/gap-filling/#recovery",
 "report": "pipeline/qa/",
 "routes": "validation/case-study/#routes",
 "run": "reference/cli/",
 "settings": "reference/settings/",
 "sources": "validation/case-study/#sources",
 "stage-ik": "pipeline/ik/",
 "stage-io": "reference/outputs/#stage-io",
 "stage-markers": "pipeline/markers/",
 "stage-qa": "pipeline/box-lift/#stage-qa",
 "stage-read": "pipeline/reading/",
 "stage-ref": "pipeline/box-lift/#stage-ref",
 "static": "validation/joint-angles/#static",
 "status": "validation/summary/",
 "steps": "pipeline/overview/",
 "terms": "reference/glossary/",
 "validation": "validation/joint-angles/#validation",
 "validation-scripts": "reference/validation-scripts/#validation-scripts",
 "viewer-files": "reference/outputs/#viewer-files",
 "xsens": "validation/xsens/"
};
  var id = decodeURIComponent((location.hash || "").slice(1));
  var root = document.documentElement.getAttribute("data-content_root") || "./";
  if (id && map.hasOwnProperty(id) && !document.getElementById(id)) {
    location.replace(root + map[id]);
  }
})();
