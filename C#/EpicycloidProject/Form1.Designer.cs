namespace EpicycloidProject
{
    partial class Form1
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Требуемый метод для поддержки конструктора — не изменяйте 
        /// содержимое этого метода с помощью редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.pictureBoxGraph = new System.Windows.Forms.PictureBox();
            this.label1 = new System.Windows.Forms.Label();
            this.textBoxA = new System.Windows.Forms.TextBox();
            this.buttonDraw = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.textBoxB = new System.Windows.Forms.TextBox();
            this.trackBarScale = new System.Windows.Forms.TrackBar();
            this.labelScale = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxGraph)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarScale)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBoxGraph
            // 
            this.pictureBoxGraph.Location = new System.Drawing.Point(-6, 0);
            this.pictureBoxGraph.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.pictureBoxGraph.Name = "pictureBoxGraph";
            this.pictureBoxGraph.Size = new System.Drawing.Size(1608, 719);
            this.pictureBoxGraph.TabIndex = 0;
            this.pictureBoxGraph.TabStop = false;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(174, 758);
            this.label1.Margin = new System.Windows.Forms.Padding(6, 0, 6, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(136, 25);
            this.label1.TabIndex = 1;
            this.label1.Text = "Параметр a:";
            // 
            // textBoxA
            // 
            this.textBoxA.Location = new System.Drawing.Point(326, 758);
            this.textBoxA.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.textBoxA.Name = "textBoxA";
            this.textBoxA.Size = new System.Drawing.Size(196, 31);
            this.textBoxA.TabIndex = 2;
            // 
            // buttonDraw
            // 
            this.buttonDraw.Location = new System.Drawing.Point(1282, 790);
            this.buttonDraw.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.buttonDraw.Name = "buttonDraw";
            this.buttonDraw.Size = new System.Drawing.Size(294, 44);
            this.buttonDraw.TabIndex = 3;
            this.buttonDraw.Text = "Построить график";
            this.buttonDraw.UseVisualStyleBackColor = true;
            this.buttonDraw.Click += new System.EventHandler(this.buttonDraw_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(174, 810);
            this.label2.Margin = new System.Windows.Forms.Padding(6, 0, 6, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(136, 25);
            this.label2.TabIndex = 4;
            this.label2.Text = "Параметр b:";
            // 
            // textBoxB
            // 
            this.textBoxB.Location = new System.Drawing.Point(326, 808);
            this.textBoxB.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.textBoxB.Name = "textBoxB";
            this.textBoxB.Size = new System.Drawing.Size(196, 31);
            this.textBoxB.TabIndex = 5;
            // 
            // trackBarScale
            // 
            this.trackBarScale.LargeChange = 25;
            this.trackBarScale.Location = new System.Drawing.Point(928, 758);
            this.trackBarScale.Maximum = 200;
            this.trackBarScale.Minimum = 10;
            this.trackBarScale.Name = "trackBarScale";
            this.trackBarScale.Size = new System.Drawing.Size(225, 90);
            this.trackBarScale.SmallChange = 5;
            this.trackBarScale.TabIndex = 6;
            this.trackBarScale.Value = 100;
            // 
            // labelScale
            // 
            this.labelScale.AutoSize = true;
            this.labelScale.Location = new System.Drawing.Point(733, 758);
            this.labelScale.Name = "labelScale";
            this.labelScale.Size = new System.Drawing.Size(171, 25);
            this.labelScale.TabIndex = 7;
            this.labelScale.Text = "Масштаб: 100%";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(12F, 25F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ActiveCaption;
            this.ClientSize = new System.Drawing.Size(1600, 865);
            this.Controls.Add(this.labelScale);
            this.Controls.Add(this.trackBarScale);
            this.Controls.Add(this.textBoxB);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.buttonDraw);
            this.Controls.Add(this.textBoxA);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.pictureBoxGraph);
            this.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxGraph)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarScale)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBoxGraph;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textBoxA;
        private System.Windows.Forms.Button buttonDraw;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textBoxB;
        private System.Windows.Forms.TrackBar trackBarScale;
        private System.Windows.Forms.Label labelScale;
    }
}

