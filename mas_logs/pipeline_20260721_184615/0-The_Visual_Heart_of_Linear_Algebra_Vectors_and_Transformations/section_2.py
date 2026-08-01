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
        # Initial Setup
        title = "Prerequisite: The Basis Vectors"
        lines = [
            "Basis vectors i-hat and j-hat are fundamental building blocks.",
            "Every vector is a scaled combination of these two.",
            "This recipe defines any point in the 2D plane."
        ]
        self.setup_layout(title, lines)
        
        # Define grid points for convenience
        # We set our origin at D2 (Column 2) to stay clear of the lecture text (L002)
        origin_pos = self.grid["D2"]
        i_end_pos = self.grid["D3"]
        j_end_pos = self.grid["C2"]
        
        # Colors for highlights and vectors
        I_COLOR = "#FF0000"   # Red as per storyboard
        J_COLOR = "#00FF00"   # Green as per storyboard
        H1_COLOR = "#00FFFF"  # Highlight color for line 1
        H2_COLOR = "#FFFF00"  # Highlight color for line 2
        H3_COLOR = "#FF00FF"  # Highlight color for line 3
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture text
        self.lecture[0].set_color(H1_COLOR)
        
        # Create i-hat and j-hat arrows
        i_hat = Arrow(origin_pos, i_end_pos, color=I_COLOR, buff=0, tip_length=0.2)
        j_hat = Arrow(origin_pos, j_end_pos, color=J_COLOR, buff=0, tip_length=0.2)
        
        # Create labels
        i_label = MathTex(r"\hat{i}", color=I_COLOR)
        j_label = MathTex(r"\hat{j}", color=J_COLOR)
        
        # Position labels within 1 grid unit (L002)
        # Fixes from Issue 33: i_label to E4, j_label to C4
        self.place_at_grid(i_label, "E4", scale_factor=0.7) 
        self.place_at_grid(j_label, "C4", scale_factor=0.7) 
        
        self.wait(2.0) # Absorption time
        self.play(FadeIn(i_hat), FadeIn(j_hat))
        self.play(Write(i_label), Write(j_label))
        self.play(Indicate(i_hat), Indicate(j_hat)) # Use Indicate for highlighting (L004)
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Update lecture highlight
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color(H2_COLOR)
        
        # "Magnetic tracks" visual metaphor
        x_axis = DashedLine(self.grid["D1"], self.grid["D6"], color="#555555", stroke_opacity=0.6)
        y_axis = DashedLine(self.grid["F2"], self.grid["A2"], color="#555555", stroke_opacity=0.6)
        
        self.wait(2.0)
        self.play(Create(x_axis), Create(y_axis))
        
        # Scaled targets: 3 * i_hat and 2 * j_hat
        i_scaled_end = self.grid["D5"]
        j_scaled_end = self.grid["B2"]
        
        i_scaled_arrow = Arrow(origin_pos, i_scaled_end, color=I_COLOR, buff=0, tip_length=0.2)
        j_scaled_arrow = Arrow(origin_pos, j_scaled_end, color=J_COLOR, buff=0, tip_length=0.2)
        
        i_scale_label = MathTex(r"3\hat{i}", color=I_COLOR)
        j_scale_label = MathTex(r"2\hat{j}", color=J_COLOR)
        
        # Fixes from Issue 33: j_scale_label to B4
        self.place_at_grid(i_scale_label, "E5", scale_factor=0.7)
        self.place_at_grid(j_scale_label, "B4", scale_factor=0.7)

        # Perform the stretching animation
        self.play(
            Transform(i_hat, i_scaled_arrow),
            Transform(j_hat, j_scaled_arrow),
            Transform(i_label, i_scale_label),
            Transform(j_label, j_scale_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Update lecture highlight
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color(H3_COLOR)
        
        # Tip-to-tail addition: Move scaled j-hat to the end of scaled i-hat
        j_moved_end = self.grid["B5"]
        j_moved_arrow = Arrow(i_scaled_end, j_moved_end, color=J_COLOR, buff=0, tip_length=0.2)
        
        # Result vector: origin (D2) to destination (B5)
        res_vector = Arrow(origin_pos, j_moved_end, color=H3_COLOR, buff=0, tip_length=0.2)
        res_label = MathTex(r"3\hat{i} + 2\hat{j}", color=H3_COLOR)
        self.place_at_grid(res_label, "A5", scale_factor=0.7)
        
        # Shifted j-label position
        j_label_moved = MathTex(r"2\hat{j}", color=J_COLOR)
        self.place_at_grid(j_label_moved, "C5", scale_factor=0.7)

        self.wait(2.0)
        
        # Morph the layout to show addition
        self.play(
            Transform(j_hat, j_moved_arrow),
            Transform(j_label, j_label_moved),
            run_time=2
        )
        
        # Final vector creation
        self.play(Create(res_vector), Write(res_label))
        self.play(Indicate(res_vector))
        self.wait(2.0)
