package phonenumbers

// PhoneNumberCurrent checks if a given string can be fully divided into 11-digit blocks, each starting with '79'.
// It returns true if the string can be fully divided, and false otherwise.
func PhoneNumberCurrent(input string) bool {

	if len(input)%11 != 0 {
		return false
	}

	for i := 0; i < len(input); i += 11 {
		if input[i] != '7' || input[i+1] != '9' {
			return false
		}
	}

	return true
}
