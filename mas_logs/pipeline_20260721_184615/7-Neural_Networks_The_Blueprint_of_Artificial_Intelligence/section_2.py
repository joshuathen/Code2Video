from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section2Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_text = "The Perceptron: The Single Neuron"
        lecture_lines = [
            "A neuron is the basic unit of neural networks.",
            "Weights adjust the importance of each incoming signal.",
            "The bias sets a threshold for the neuron's activation."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors (Hexadecimal as per L008)
        NEURON_COLOR = "#FFFFFF"
        WEIGHT_COLOR = "#00FFFF"
        BIAS_COLOR = "#FFCC00"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(NEURON_COLOR))
        
        # Create a central neuron using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg]
        neuron_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        neuron_asset.set_color(NEURON_COLOR)
        
        # Position neuron in the center-right (D5)
        self.place_at_grid(neuron_asset, "D5", scale_factor=0.8)
        
        # Grid positions for inputs
        pos_in1 = self.grid["B3"]
        pos_in2 = self.grid["D3"]
        pos_in3 = self.grid["F3"]
        
        # Lines connecting to neuron. 
        # Since SVGMobject doesn't have point_at_angle, we use boundaries or offsets from center.
        # Neuron is at D5.
        target_center = self.grid["D5"]
        
        line1 = Line(pos_in1, target_center + 0.3*UL, color=NEURON_COLOR)
        line2 = Line(pos_in2, target_center + 0.3*LEFT, color=NEURON_COLOR)
        line3 = Line(pos_in3, target_center + 0.3*DL, color=NEURON_COLOR)

        self.play(
            FadeIn(neuron_asset), 
            Create(line1), 
            Create(line2), 
            Create(line3),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WEIGHT_COLOR)
        )
        
        # Label input lines with 'w1', 'w2', 'w3' in cyan (#00FFFF)
        # Using simple MathTex (L022 fallback to Text if needed, but simple MathTex is usually fine)
        w1 = MathTex(r"\mathbf{w}_1", color=WEIGHT_COLOR)
        w2 = MathTex(r"w_2", color=WEIGHT_COLOR)
        w3 = MathTex(r"w_3", color=WEIGHT_COLOR)
        
        # Positioning to avoid overlaps (as per issues 25, 26, 27 and 40)
        # Fix: Move 'w1' label to 'B4' (scale 0.8).
        # Fix: Move 'w2' label to 'D3' (scale 0.8).
        # Fix: Move 'w3' label to 'F4' (scale 0.8).
        self.place_at_grid(w1, "B4", scale_factor=0.8)
        self.place_at_grid(w2, "D3", scale_factor=0.8)
        self.place_at_grid(w3, "F4", scale_factor=0.8)
        
        # Since w2 is at D3 (start of line2), let's offset it slightly left or use area to avoid exact overlap
        # Actually, VideoCritic said D3. Let's follow exactly.
        
        self.play(
            Write(w1),
            Write(w2),
            Write(w3)
        )
        
        # Highlight w1's importance using Indicate (L004)
        self.play(Indicate(w1, color=WEIGHT_COLOR, scale_factor=1.2))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BIAS_COLOR)
        )
        
        # Display a yellow '+ b' (#FFCC00) inside the neuron circle to represent bias.
        # Positioned inside the neuron at D5. 
        # Using self.place_at_grid for precise positioning.
        bias_label = MathTex("+ b", color=BIAS_COLOR)
        self.place_at_grid(bias_label, "D5", scale_factor=0.8)
        
        self.play(FadeIn(bias_label))
        self.wait(1.5)
        
        # Reset lecture colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2.0)
