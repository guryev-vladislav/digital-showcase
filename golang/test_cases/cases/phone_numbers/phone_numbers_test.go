package phonenumbers

import "testing"

func TestPhoneNumberCurrent(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"79035095678", true},
		{"7903509567879067853256", true},
		{"7903509567", false},
		{"7929397979797979797979", false},
		{"7979797979797979797979", false},
		{"79797979797979797979", false},
	}

	for _, test := range tests {
		actual := PhoneNumberCurrent(test.input)
		if actual != test.expected {
			t.Errorf("PhoneNumberCurrent(%q) = %t, but got %t", test.input, test.expected, actual)
		}
	}
}
