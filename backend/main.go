// backend/main.go
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/api/confirm-received", confirmReceivedHandler)

	log.Printf("Backend listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "OK")
}

func confirmReceivedHandler(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "Missing id parameter", http.StatusBadRequest)
		return
	}

	sheetID := os.Getenv("GOOGLE_SHEET_ID")
	credFile := os.Getenv("GOOGLE_CREDENTIALS_FILE")
	if credFile == "" {
		credFile = "credentials.json"
	}
	if sheetID == "" {
		http.Error(w, "Server misconfiguration: GOOGLE_SHEET_ID not set", http.StatusInternalServerError)
		return
	}

	if err := updateSheetStatus(r.Context(), sheetID, credFile, id, "Received"); err != nil {
		log.Printf("ERROR updating sheet for %s: %v", id, err)
		http.Error(w, "Failed to update status", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprintf(w, `<!DOCTYPE html>
<html><body style="font-family:sans-serif;text-align:center;padding:60px">
<h2 style="color:#4CAF50">&#x2713; Đã xác nhận nhận bưu phẩm</h2>
<p>Mã vận đơn: <strong>%s</strong></p>
<p>Trạng thái đã được cập nhật thành <strong>Received</strong>.</p>
</body></html>`, id)
}
