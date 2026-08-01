from manim import *
import os

# Pre-emptively create the media/texts directory
os.makedirs(os.path.join("media", "texts"), exist_ok=True)

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
        # Data from shared state
        title = "Basis Vectors: The DNA of the Grid"
        lecture_lines = [
            "Every vector starts from two fundamental units.",
            "i-hat points one unit to the right.",
            "j-hat points one unit straight up.",
            "Any point is reached by scaling and adding these two.",
            "They are the building blocks of our entire coordinate system."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_I = "#FF0000"  # Red
        COLOR_J = "#00FF00"  # Green
        COLOR_V = "#FFFFFF"  # White
        
        # === Animation for Lecture Line 1 ===
        # Show a grid. Animate a red arrow (#FF0000) from (0,0) to (1,0) labeled 'i-hat' 
        # and a green arrow (#00FF00) from (0,0) to (0,1) labeled 'j-hat'.
        self.play(self.lecture[0].animate.set_color(COLOR_V))
        
        # Create plane
        plane = NumberPlane(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'A1', 'F6')
        origin = plane.coords_to_point(0, 0)
        
        # Vectors
        i_hat = Arrow(origin, plane.coords_to_point(1, 0), buff=0, color=COLOR_I)
        j_hat = Arrow(origin, plane.coords_to_point(0, 1), buff=0, color=COLOR_J)
        
        # Labels (Applying fixes from Issue 21 and 22)
        i_label = MathTex("\\hat{i}", color=COLOR_I)
        self.place_at_grid(i_label, 'F2', scale_factor=0.8) # Fix Issue 21
        
        j_label = MathTex("\\hat{j}", color=COLOR_J)
        self.place_at_grid(j_label, 'E1', scale_factor=0.8) # Fix Issue 22
        
        self.play(Create(plane))
        self.play(GrowArrow(i_hat), FadeIn(i_label))
        self.play(GrowArrow(j_hat), FadeIn(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Flash the red i-hat (#FF0000) three times to emphasize its horizontal direction.
        self.play(self.lecture[1].animate.set_color(COLOR_I))
        for _ in range(3):
            self.play(Flash(i_hat, color=COLOR_I, line_length=0.3, flash_radius=0.5), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash the green j-hat (#00FF00) three times to emphasize its vertical direction.
        self.play(self.lecture[2].animate.set_color(COLOR_J))
        for _ in range(3):
            self.play(Flash(j_hat, color=COLOR_J, line_length=0.3, flash_radius=0.5), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Animate 4 red i-hat arrows and 3 green j-hat arrows appearing tip-to-tail to reach point (4,3). 
        # Draw a white vector (#FFFFFF) from origin to (4,3).
        self.play(self.lecture[3].animate.set_color(COLOR_V))
        
        steps = VGroup()
        # 4 i-hat steps
        curr_pos = origin
        for i in range(4):
            next_pos = plane.coords_to_point(i+1, 0)
            step = Arrow(curr_pos, next_pos, buff=0, color=COLOR_I, stroke_width=4)
            steps.add(step)
            curr_pos = next_pos
        
        # 3 j-hat steps (starting from (4,0))
        for j in range(3):
            next_pos = plane.coords_to_point(4, j+1)
            step = Arrow(curr_pos, next_pos, buff=0, color=COLOR_J, stroke_width=4)
            steps.add(step)
            curr_pos = next_pos
            
        vector_43 = Arrow(origin, plane.coords_to_point(4, 3), buff=0, color=COLOR_V)
        
        # Applying fix from Issue 23
        coord_label = MathTex("(4, 3)", color=COLOR_V)
        self.place_at_grid(coord_label, 'B6', scale_factor=0.9)
        
        self.play(Create(steps), run_time=3)
        self.play(GrowArrow(vector_43), FadeIn(coord_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the original i-hat, j-hat, and the new vector (4,3) together, then fade out the construction segments.
        self.play(self.lecture[4].animate.set_color(COLOR_V))
        
        highlight = SurroundingRectangle(VGroup(i_hat, j_hat, vector_43), color=YELLOW, buff=0.2)
        self.play(Create(highlight))
        self.play(FadeOut(steps), FadeOut(highlight))
        self.wait(2)
