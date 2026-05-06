// backend/sheets.go
package main

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/api/option"
	"google.golang.org/api/sheets/v4"
)

// updateSheetStatus tìm dòng có ma_van_don trong cột A,
// sau đó cập nhật cột K (trang_thai) và L (ngay_nhan).
func updateSheetStatus(
	ctx context.Context,
	sheetID string,
	credFile string,
	maVanDon string,
	newStatus string,
) error {
	srv, err := sheets.NewService(ctx, option.WithCredentialsFile(credFile))
	if err != nil {
		return fmt.Errorf("create sheets service: %w", err)
	}

	resp, err := srv.Spreadsheets.Values.Get(sheetID, "Sheet1!A:A").Do()
	if err != nil {
		return fmt.Errorf("read column A: %w", err)
	}

	rowIndex := -1
	for i, row := range resp.Values {
		if len(row) > 0 && row[0] == maVanDon {
			rowIndex = i + 1 // Sheets API dùng 1-indexed
			break
		}
	}
	if rowIndex == -1 {
		return fmt.Errorf("ma_van_don %q not found in sheet", maVanDon)
	}

	// Cột K = trang_thai (index 10), Cột L = ngay_nhan (index 11)
	updateRange := fmt.Sprintf("Sheet1!K%d:L%d", rowIndex, rowIndex)
	timestamp := time.Now().Format("02/01/2006 15:04:05")
	values := &sheets.ValueRange{
		Values: [][]interface{}{{newStatus, timestamp}},
	}

	_, err = srv.Spreadsheets.Values.
		Update(sheetID, updateRange, values).
		ValueInputOption("USER_ENTERED").
		Do()
	if err != nil {
		return fmt.Errorf("update cells %s: %w", updateRange, err)
	}

	return nil
}
