package oauth2

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"sync"
	"time"
)

type CallbackServer struct {
	server     *http.Server
	resultChan chan *AuthCodeResult
	state      string
	timeout    time.Duration
	mutex      sync.Mutex
	started    bool
	path       string
}

func NewCallbackServer(path, state string, timeout time.Duration) *CallbackServer {
	resultChan := make(chan *AuthCodeResult, 1)

	mux := http.NewServeMux()
	server := &http.Server{
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	cbServer := &CallbackServer{
		server:     server,
		resultChan: resultChan,
		state:      state,
		timeout:    timeout,
		path:       path, // ← СОХРАНЯЕМ путь
	}

	// ИЗМЕНЕНО: используем переданный путь
	mux.HandleFunc(path, cbServer.handleCallback)
	mux.HandleFunc("/health", cbServer.handleHealth)

	return cbServer
}

func (s *CallbackServer) Start(port int) (string, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.started {
		return "", fmt.Errorf("server already started")
	}

	if port <= 0 || port > 65535 {
		return "", fmt.Errorf("invalid port: %d", port)
	}

	addr := fmt.Sprintf(":%d", port)
	s.server.Addr = addr

	go func() {
		if err := s.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("Server error: %v", err)
		}
	}()

	if err := s.waitForServer(addr); err != nil {
		return "", err
	}

	s.started = true
	return fmt.Sprintf("http://localhost:%d", port), nil
}

func (s *CallbackServer) WaitForResult() (*AuthCodeResult, error) {
	select {
	case result := <-s.resultChan:
		return result, nil
	case <-time.After(s.timeout):
		return nil, fmt.Errorf("timeout waiting for authorization callback")
	}
}

func (s *CallbackServer) Stop() error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.started {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	s.started = false
	return s.server.Shutdown(ctx)
}

func (s *CallbackServer) handleCallback(w http.ResponseWriter, r *http.Request) {
	callbackURL := getFullRequestURL(r)

	result, err := ParseAuthCodeCallback(callbackURL, s.state)
	if err != nil {
		http.Error(w, "Authorization failed: "+err.Error(), http.StatusBadRequest)
		s.resultChan <- &AuthCodeResult{Error: err.Error()}
		return
	}

	s.resultChan <- result

	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(`
		<html>
			<body>
				<h2>Authorization Successful!</h2>
				<p>You can close this window.</p>
			</body>
		</html>
	`))
}

func (s *CallbackServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func (s *CallbackServer) waitForServer(addr string) error {
	for i := 0; i < 10; i++ {
		conn, err := net.Dial("tcp", addr)
		if err == nil {
			conn.Close()
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("server failed to start in time")
}

func getFullRequestURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return fmt.Sprintf("%s://%s%s", scheme, r.Host, r.RequestURI)
}
