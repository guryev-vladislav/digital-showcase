package internal

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func insert(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val: val}
	}
	if val < root.Val {
		root.Left = insert(root.Left, val)
	} else if val > root.Val {
		root.Right = insert(root.Right, val)
	}
	return root
}

func height(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftHeight := height(root.Left)
	rightHeight := height(root.Right)
	if leftHeight > rightHeight {
		return leftHeight + 1
	}
	return rightHeight + 1
}

func TreeHeight() {
	scanner := bufio.NewScanner(os.Stdin)
	var root *TreeNode

	for scanner.Scan() {
		val, _ := strconv.Atoi(scanner.Text())
		if val == 0 {
			break
		}
		root = insert(root, val) // Обновляем root
	}

	fmt.Println(height(root))
}
