const BASE_URL = "http://127.0.0.1:8000";

export async function healthCheck() {
  const res = await fetch(BASE_URL + "/");
  return res.json();
}

export async function recognizeFace(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(BASE_URL + "/recognize", {
    method: "POST",
    body: formData,
  });

  return res.json();
}

export async function registerStart(name) {
  const res = await fetch(
    BASE_URL + "/register/start?name=" + name,
    { method: "POST" }
  );

  return res.json();
}

export async function registerFrame(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(BASE_URL + "/register/frame", {
    method: "POST",
    body: formData,
  });

  return res.json();
}

export async function registerSave() {
  const res = await fetch(BASE_URL + "/register/save", {
    method: "POST",
  });

  return res.json();
}