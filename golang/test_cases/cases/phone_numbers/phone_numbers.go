package phonenumbers

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
