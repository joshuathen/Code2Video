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
        # Initialize layout with section details from storyboard
        title = "Prerequisite: The Slope of a Line"
        lines = [
            "- A straight line has a constant slope.",
            "- We calculate slope as rise over run.",
            "- This value represents a steady rate of change."
        ]
        self.setup_layout(title, lines)
        
        # Define colors from storyboard
        COLOR_BLUE = "#0000FF"
        COLOR_GREEN = "#00FF00"
        COLOR_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Color transition for the first lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_BLUE))
        
        # Line between (1,1) and (3,5) mapped to grid E2 and A4
        # E2: x=1.5, y=-1.8 | A4: x=3.5, y=2.2 | Slope = (2.2 - -1.8) / (3.5 - 1.5) = 2.0
        p1 = self.grid['E2']
        p2 = self.grid['A4']
        
        blue_line = Line(p1, p2, color=COLOR_BLUE)
        dot1 = Dot(p1, color=COLOR_BLUE)
        dot2 = Dot(p2, color=COLOR_BLUE)
        
        # Point labels (1,1) and (3,5) positioned using grid
        # Fix: Move label1 to F1 to avoid overlap (Issue 30)
        label1 = MathTex("(1, 1)", font_size=20, color=COLOR_BLUE)
        self.place_at_grid(label1, "F1", scale_factor=1.0)
        
        # Fix: Move label2 to A6 to avoid overlap (Issue 30)
        label2 = MathTex("(3, 5)", font_size=20, color=COLOR_BLUE)
        self.place_at_grid(label2, "A6", scale_factor=1.0)
        
        self.play(Create(blue_line), FadeIn(dot1, dot2))
        self.play(Write(label1), Write(label2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color transition for the second lecture line
        self.play(self.lecture[1].animate.set_color(COLOR_GREEN))
        
        # Rise and Run visualization
        p_corner = self.grid['E4'] # (3.5, -1.8)
        
        run_seg = Line(p1, p_corner, color=COLOR_GREEN)
        rise_seg = Line(p_corner, p2, color=COLOR_GREEN)
        
        # Labels for Rise and Run segments
        # Fix: Adjust run_text and rise_text positions (Issue 30)
        run_text = Text("Run", font_size=18, color=COLOR_GREEN)
        self.place_at_grid(run_text, "E3", scale_factor=1.0)
        
        rise_text = Text("Rise", font_size=18, color=COLOR_GREEN)
        self.place_at_grid(rise_text, "D6", scale_factor=1.0)
        
        self.play(Create(run_seg))
        self.play(FadeIn(run_text))
        self.play(Create(rise_seg))
        self.play(FadeIn(rise_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color transition for the third lecture line
        self.play(self.lecture[2].animate.set_color(COLOR_WHITE))
        
        # Final slope value label
        # Fix: Use place_at_grid for slope_label at B6 with scale 0.8 (Issue 30)
        slope_label = MathTex(r"\text{Slope} = \frac{\text{Rise}}{\text{Run}} = 2", font_size=24, color=COLOR_WHITE)
        self.place_at_grid(slope_label, "B6", scale_factor=0.8)
        
        self.play(Write(slope_label))
        self.wait(2)
